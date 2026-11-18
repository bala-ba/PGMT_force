"""Display a PGMT Stage1 policy on MJWarp and record raw foot contacts.

Uses the GPU solver's contact arrays, never the CPU viewer's reconstructed
contacts. Samples at the control rate after the environment's final forward,
before manual resets. Run --help for playback/recording options.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(ROOT / "logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt"))
    p.add_argument("--motion", nargs="+", default=[str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz")])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--duration", type=float, default=60., help="Total simulated seconds to record; pauses at completion")
    p.add_argument("--output", default=str(ROOT / "logs/pgmt_mjwarp_contacts" / datetime.now().strftime("%Y%m%d_%H%M%S")))
    p.add_argument("--force-threshold", type=float, default=.001, help="Normal force threshold in N for loaded contacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", help="Short verification runs without a window")
    p.add_argument("--exit-on-complete", action="store_true", help="Close the window when recording completes")
    a = p.parse_args()
    if a.duration <= 0 or a.force_threshold < 0:
        p.error("duration must be positive and force-threshold must be nonnegative")
    return a


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


class ContactRecorder:
    def __init__(self, env, output, threshold):
        import mujoco
        import numpy as np
        import warp as wp
        self.env, self.output, self.threshold = env, output, threshold
        self.model, self.data = env.sim.mj_model, env.sim.wp_data
        m, d = self.model, self.data
        self.body_ids = {}
        self.foot_geoms = {}
        self.names = {i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or str(i) for i in range(m.ngeom)}
        for foot in ("left", "right"):
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{foot}_ankle_roll_link")
            if body < 0:
                raise ValueError(f"Missing {foot} foot body")
            self.body_ids[foot] = body
            geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == body
                     and (m.geom_contype[i] or m.geom_conaffinity[i])]
            if not geoms:
                raise ValueError(f"Missing collision geometries on {foot} foot")
            self.foot_geoms.update({i: foot for i in geoms})
        # mjlab's plane belongs to a fixed child body, not body 0 itself.
        self.terrain_geoms = {i for i in range(m.ngeom)
                              if m.body_weldid[m.geom_bodyid[i]] == 0
                              and (m.geom_contype[i] or m.geom_conaffinity[i])}
        if not self.terrain_geoms:
            raise ValueError("No static terrain collision geometry found")
        self.ids = wp.array(np.arange(d.naconmax, dtype=np.int32), device=d.nacon.device)
        self.forces = wp.zeros(d.naconmax, dtype=wp.spatial_vector, device=d.nacon.device)
        self.count_file = (output / "contact_counts.csv").open("w", newline="")
        self.point_file = (output / "contact_points.csv").open("w", newline="")
        common = ["sample", "time_s", "episode", "episode_step", "motion", "action_ref_frame", "state_ref_frame"]
        count_fields = common + ["terminal", "termination", "all_detected", "constraint_count"]
        for foot in self.body_ids:
            count_fields += [f"{foot}_{key}" for key in ("detected", "constrained", "loaded", "normal_force_N")]
        point_fields = common + ["foot", "contact_index", "foot_geom", "other_geom", "distance_m",
            "constrained", "loaded", "normal_force_N", "world_x", "world_y", "world_z",
            "foot_x", "foot_y", "foot_z", "force_world_x", "force_world_y", "force_world_z",
            "force_foot_x", "force_foot_y", "force_foot_z"]
        self.count_csv = csv.DictWriter(self.count_file, count_fields)
        self.point_csv = csv.DictWriter(self.point_file, point_fields)
        self.count_csv.writeheader()
        self.point_csv.writeheader()
        self.rows, self.latest = [], []
        self.max_local_roundtrip_error = 0.
        self.max_force_roundtrip_error = 0.
        self.max_contacts = 0
        self.max_constraints = 0
        self.negative_force_samples = 0

    def capture(self, common, terminated, reasons):
        import numpy as np
        import warp as wp
        import mujoco_warp as mjwarp
        d = self.data
        n = int(d.nacon.numpy()[0])
        nefc = int(d.nefc.numpy()[0])
        if n >= d.naconmax or nefc >= d.njmax:
            raise RuntimeError(f"Contact/constraint capacity reached: {n}/{d.naconmax}, {nefc}/{d.njmax}")
        self.max_contacts = max(self.max_contacts, n)
        self.max_constraints = max(self.max_constraints, nefc)
        with wp.ScopedDevice(d.nacon.device):
            self.forces.zero_()
            mjwarp.contact_force(self.env.sim.wp_model, d, self.ids, False, self.forces)
        local_forces = self.forces.numpy()[:n, :3]
        pairs = d.contact.geom.numpy()[:n]
        positions = d.contact.pos.numpy()[:n]
        frames = d.contact.frame.numpy()[:n]
        addresses = d.contact.efc_address.numpy()[:n, 0]
        distances = d.contact.dist.numpy()[:n]
        worlds = d.contact.worldid.numpy()[:n]
        body_pos, body_rot = d.xpos.numpy()[0], d.xmat.numpy()[0]
        counts = dict(common, terminal=int(terminated), termination="|".join(reasons),
                      all_detected=n, constraint_count=nefc)
        for foot in self.body_ids:
            counts.update({f"{foot}_{key}": 0 for key in ("detected", "constrained", "loaded", "normal_force_N")})
        self.latest = []
        for i, (g0, g1) in enumerate(pairs):
            if worlds[i] != 0:
                continue
            if g0 in self.foot_geoms and g1 in self.terrain_geoms:
                foot_geom, other, sign = int(g0), int(g1), -1.
            elif g1 in self.foot_geoms and g0 in self.terrain_geoms:
                foot_geom, other, sign = int(g1), int(g0), 1.
            else:
                continue
            foot = self.foot_geoms[foot_geom]
            body = self.body_ids[foot]
            rotation = body_rot[body]
            pos = positions[i]
            pos_local = rotation.T @ (pos - body_pos[body])
            # Raw MuJoCo contact frame: row 0 is the normal geom0 -> geom1.
            # contact_force reports force on geom1; choose environment-on-foot.
            force_world = sign * (frames[i].T @ local_forces[i])
            force_local = rotation.T @ force_world
            normal = float(local_forces[i, 0])
            constrained = bool(addresses[i] >= 0)
            loaded = bool(constrained and normal > self.threshold)
            self.max_local_roundtrip_error = max(self.max_local_roundtrip_error,
                float(np.linalg.norm(rotation @ pos_local + body_pos[body] - pos)))
            self.max_force_roundtrip_error = max(self.max_force_roundtrip_error,
                float(np.linalg.norm(rotation @ force_local - force_world)))
            if not np.isfinite(np.r_[pos, pos_local, force_world, force_local, normal]).all():
                raise ValueError("Nonfinite contact data")
            self.negative_force_samples += int(normal < -1e-5)
            counts[f"{foot}_detected"] += 1
            counts[f"{foot}_constrained"] += int(constrained)
            counts[f"{foot}_loaded"] += int(loaded)
            counts[f"{foot}_normal_force_N"] += normal
            row = dict(common, foot=foot, contact_index=i, foot_geom=self.names[foot_geom],
                other_geom=self.names[other], distance_m=float(distances[i]), constrained=int(constrained),
                loaded=int(loaded), normal_force_N=normal)
            for prefix, values in (("world", pos), ("foot", pos_local),
                                   ("force_world", force_world), ("force_foot", force_local)):
                row.update({f"{prefix}_{axis}": float(value) for axis, value in zip("xyz", values)})
            self.point_csv.writerow(row)
            self.latest.append(row)
        self.count_csv.writerow(counts)
        self.rows.append(counts)
        if len(self.rows) % 50 == 0:
            self.flush()
        return counts

    def flush(self):
        self.count_file.flush()
        self.point_file.flush()

    def summary(self):
        import numpy as np
        result = dict(samples=len(self.rows), max_detected_contacts_all_bodies=self.max_contacts,
            contact_capacity=self.data.naconmax, max_constraints=self.max_constraints,
            constraint_capacity=self.data.njmax,
            max_position_roundtrip_error_m=self.max_local_roundtrip_error,
            max_force_roundtrip_error_N=self.max_force_roundtrip_error,
            negative_normal_force_records=self.negative_force_samples)
        for foot in self.body_ids:
            result[foot] = {}
            for kind in ("detected", "constrained", "loaded"):
                values = np.array([r[f"{foot}_{kind}"] for r in self.rows], dtype=int)
                counts = {str(v): int((values == v).sum()) for v in np.unique(values)}
                result[foot][kind] = dict(min=int(values.min()) if len(values) else None,
                    max=int(values.max()) if len(values) else None,
                    mean=float(values.mean()) if len(values) else None, histogram=counts)
        return result

    def close(self):
        self.count_file.close()
        self.point_file.close()


def main():
    args = arguments()
    os.environ["PMT_BACKEND"] = "mjlab"
    import mujoco
    import numpy as np
    import torch
    import warp as wp
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from pmt_tasks.builder import build_env_cfg
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic
    from motion_tracking_rl.networks.pgmt_force_actor_critic import PGMTForceActorCritic

    torch.set_num_threads(4)
    wp.config.kernel_cache_dir = os.environ.get("PMT_WARP_CACHE", "/tmp/pgmt-mjlab-warp")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; launch with access to host NVIDIA devices")
    if not args.headless and not os.environ.get("DISPLAY"):
        raise RuntimeError("Native display requested but DISPLAY is unset")
    checkpoint = Path(args.checkpoint).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["policy_metadata"]
    force_v1 = metadata.get("policy_family") == "pgmt_stage1_force_v1"
    if metadata.get("policy_family") not in ("pgmt_stage1", "pgmt_stage1_force_v1"):
        raise ValueError("This viewer requires a PGMT Stage1 checkpoint")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f).exists() for f in ("metadata.json", "contact_points.csv")):
        raise FileExistsError(f"Refusing to overwrite run {output}")
    cfg = build_env_cfg("pgmt_stage1", backend="mjlab")
    cfg.scene.num_envs, cfg.seed = 1, args.seed
    cfg.auto_reset = False
    cfg.events = {}
    cfg.commands["motion"].motion_files = [str(Path(p).resolve()) for p in args.motion]
    cfg.commands["motion"].chunk_length = 0
    cfg.commands["motion"].evaluation = True
    cfg.commands["motion"].noise = False
    cfg.terminations.pop("time_out", None)
    if force_v1:
        from pmt_tasks.backends.mjlab_pgmt.force_feedback import enable_force_feedback_v1
        enable_force_feedback_v1(cfg, {"policy": {"class_name": "PGMTActorCritic"}},
                                 metadata["policy_kwargs"]["force_feedback_cfg"])
    status = dict(status="initializing", pid=os.getpid(), arguments=vars(args),
        backend="mujoco_warp", device=args.device, checkpoint=str(checkpoint), checkpoint_sha256=digest,
        checkpoint_iteration=saved.get("iter"), display=os.environ.get("DISPLAY"),
        physics_dt=cfg.sim.mujoco.timestep, sample_dt=cfg.decimation * cfg.sim.mujoco.timestep,
        packages={p: version(p) for p in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang")},
        sampling="After final MJWarp forward at 50 Hz, before any episode reset; not CPU viewer contacts",
        point_definition="One raw foot-terrain contact record, not deduplicated, not a fixed tactile sensor",
        loaded_definition=f"Constraint included and normal_force_N > {args.force_threshold}",
        coordinates="world XYZ and ankle_roll_link-local XYZ, metres; forces environment-on-foot, newtons",
        contact_index="Frame-local packed index only; not a persistent contact ID", training=False)
    write_json(output / "metadata.json", status)
    env = recorder = viewer = None
    controls = dict(stop=False, paused=False, reset=False)
    episodes = []
    sample = episode = episode_step = 0
    complete = False
    def stop_signal(*_):
        controls["stop"] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_signal)
    try:
        env = ManagerBasedRlEnv(cfg, device=args.device)
        command = env.command_manager.get_term("motion")
        obs, _ = env.reset(seed=args.seed)
        policy_class = PGMTForceActorCritic if force_v1 else PGMTActorCritic
        policy = policy_class(TensorDict(obs, batch_size=[1]), metadata["obs_groups"],
                              metadata["num_actions"], **metadata["policy_kwargs"]).to(args.device)
        policy.load_state_dict(saved["model_state_dict"], strict=True)
        policy.eval()
        if policy.get_checkpoint_metadata() != metadata:
            raise ValueError("Checkpoint observation/action metadata mismatch")
        recorder = ContactRecorder(env, output, args.force_threshold)
        status.update(status="running", motion_records=command.data_store.records,
            terrain_collision_geoms=[recorder.names[g] for g in sorted(recorder.terrain_geoms)],
            foot_collision_geoms={side: [recorder.names[g] for g, f in recorder.foot_geoms.items() if f == side]
                                  for side in recorder.body_ids})
        write_json(output / "metadata.json", status)
        render_data = mujoco.MjData(env.sim.mj_model)
        if not args.headless:
            import mujoco.viewer
            def keypress(key):
                if key == 32:
                    controls["paused"] = not controls["paused"]
                elif key in (ord("R"), ord("r")):
                    controls["reset"] = True
                elif key in (ord("Q"), ord("q")):
                    controls["stop"] = True
            viewer = mujoco.viewer.launch_passive(env.sim.mj_model, render_data,
                key_callback=keypress, show_left_ui=False, show_right_ui=False)
            with viewer.lock():
                viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135., -20., 3.0
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        print(f"[ready] {args.device}, checkpoint iter={saved.get('iter')}, output={output}", flush=True)
        print("[keys] SPACE pause/resume, R reset, Q finish. Green=left, magenta=right; dim=unloaded.", flush=True)
        pending_reset = False
        latest = None
        next_log = 1.
        while not controls["stop"] and (viewer is None or viewer.is_running()):
            started = time.monotonic()
            if (pending_reset or controls["reset"]) and not complete:
                if controls["reset"] and not pending_reset:
                    episodes.append(dict(episode=episode, samples=episode_step, reason=["manual_reset"]))
                episode += 1
                episode_step = 0
                command.evaluation_motion_ids = torch.tensor([episode % command.data_store.num_motions], device=env.device)
                obs, _ = env.reset()
                pending_reset = controls["reset"] = False
            if not controls["paused"] and not complete:
                action_frame = int(command.frame_ids[0])
                motion_id = int(command.motion_ids[0])
                with torch.inference_mode():
                    actions = policy.act_inference(TensorDict(obs, batch_size=[1]))
                    if not torch.isfinite(actions).all():
                        raise ValueError("Nonfinite policy action")
                    obs, _, terminated, truncated, _ = env.step(actions)
                sample += 1
                episode_step += 1
                reasons = [name for name in cfg.terminations if bool(env.termination_manager.get_term(name)[0])]
                done = bool((terminated | truncated)[0])
                common = dict(sample=sample, time_s=sample * env.step_dt, episode=episode,
                    episode_step=episode_step, motion=Path(command.data_store.records[motion_id]["file"]).name,
                    action_ref_frame=action_frame, state_ref_frame=int(command.frame_ids[0]))
                latest = recorder.capture(common, done, reasons)
                if done:
                    episodes.append(dict(episode=episode, samples=episode_step, reason=reasons,
                                         final_frame=int(command.frame_ids[0])))
                    pending_reset = True
                if sample * env.step_dt >= next_log:
                    print(f"[contacts] t={sample * env.step_dt:.2f}s episode={episode} "
                          f"L={latest['left_loaded']}/{latest['left_detected']} "
                          f"R={latest['right_loaded']}/{latest['right_detected']} (loaded/detected)", flush=True)
                    next_log += 1.
                complete = sample * env.step_dt >= args.duration - 1e-9
                if complete:
                    if not done:
                        episodes.append(dict(episode=episode, samples=episode_step, reason=["recording_limit"],
                                             final_frame=int(command.frame_ids[0])))
                    recorder.flush()
                    write_json(output / "summary.json", dict(recorder.summary(), episodes=episodes,
                        recorded_time_s=sample * env.step_dt, completed=True))
                    status.update(status="complete", samples=sample, recorded_time_s=sample * env.step_dt)
                    write_json(output / "metadata.json", status)
                    print(f"[complete] Recorded {sample} samples; summary.json saved. Viewer paused.", flush=True)
                    if args.headless or args.exit_on_complete:
                        break
            if viewer is not None:
                d = env.sim.wp_data
                render_data.qpos[:] = d.qpos.numpy()[0]
                render_data.qvel[:] = d.qvel.numpy()[0]
                render_data.ctrl[:] = d.ctrl.numpy()[0]
                mujoco.mj_forward(env.sim.mj_model, render_data)
                with viewer.lock():
                    viewer.cam.lookat[:] = render_data.qpos[:3]
                    viewer.user_scn.ngeom = 0
                    for point in recorder.latest:
                        i = viewer.user_scn.ngeom
                        color = np.array([.1, 1., .35, 1.] if point["foot"] == "left" else [1., .15, .75, 1.])
                        if not point["loaded"]:
                            color[:3] *= .45
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.009, .009, .009]), np.array([point[f"world_{a}"] for a in "xyz"]),
                            np.eye(3).flatten(), color)
                        viewer.user_scn.ngeom += 1
                if latest:
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                        "Backend\nState\nTime\nEpisode\nLeft loaded / detected\nRight loaded / detected\nControls",
                        f"MuJoCo Warp {args.device}\n{'COMPLETE (paused)' if complete else 'PAUSED' if controls['paused'] else 'RECORDING'}"
                        f"\n{sample * env.step_dt:.2f} s\n{episode}"
                        f"\n{latest['left_loaded']} / {latest['left_detected']}"
                        f"\n{latest['right_loaded']} / {latest['right_detected']}\nSPACE pause | R reset | Q quit"))
                viewer.sync()
                time.sleep(max(0., (0.05 if complete or controls["paused"] else env.step_dt) - (time.monotonic() - started)))
    except Exception as exc:
        status.update(status="error", error=repr(exc))
        raise
    finally:
        if recorder is not None:
            recorder.flush()
            write_json(output / "summary.json", dict(recorder.summary(), episodes=episodes,
                recorded_time_s=sample * env.step_dt, completed=complete))
            recorder.close()
        status.update(checkpoint_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest,
                      samples=sample)
        if status["status"] == "running":
            status["status"] = "stopped"
        write_json(output / "metadata.json", status)
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
