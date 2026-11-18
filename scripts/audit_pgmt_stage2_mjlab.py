"""Bounded Stage2 CUDA rollout audit across every configured terrain family."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--terrain-asset", type=Path)
    parser.add_argument("--trace-after-seconds", type=float, default=0., help="Optional diagnostic stack dump interval")
    parser.add_argument("--broadphase", choices=("auto", "nxn"), default="auto", help="Compare the native prefiltered-pair broadphase")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.trace_after_seconds > 0:
        import faulthandler
        faulthandler.dump_traceback_later(args.trace_after_seconds, repeat=True)
    import mujoco
    import numpy as np
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_agent
    from pmt_tasks.backends.mjlab_pgmt.force_feedback import enable_force_feedback_v1
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    from motion_tracking_rl.networks import PGMTForceActorCritic
    from motion_tracking_rl.utils.pgmt_force_checkpoint import migrate_instanttrack_force

    torch.set_num_threads(4)
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    if args.terrain_asset:
        from functools import partial
        from pmt_tasks.backends.mjlab_pgmt.terrain import add_native_terrain
        cfg.commands["motion"].terrain_asset = str(args.terrain_asset.resolve())
        cfg.scene.spec_fn = partial(add_native_terrain, asset=cfg.commands["motion"].terrain_asset)
    from pmt_tasks.backends.mjlab_pgmt.terrain import load_manifest
    manifest = load_manifest(cfg.commands["motion"].terrain_asset)
    family_count = len(manifest["spec"]["families"])
    levels = [0] * family_count + [manifest["spec"]["num_rows"] - 1] * family_count
    cfg.scene.num_envs, cfg.seed = 2 * family_count, 42
    cfg.commands["motion"].motion_files = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.commands["motion"].fixed_start_frame = 0
    agent = build_agent(build_task_config("pgmt_stage2"), stage=2)
    enable_force_feedback_v1(cfg, agent)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("[Audit] Building terrain environment", flush=True)
    env = PGMTVectorEnv(ManagerBasedRlEnv(cfg, device=args.device))
    try:
        raw, command = env.env, env.command
        if args.broadphase == "nxn":
            from mujoco_warp._src.types import BroadphaseType
            raw.sim.wp_model.opt.broadphase = BroadphaseType.NXN
            raw.sim.create_graph()
        terrain = command.terrain
        terrain.terrain_types[:] = torch.arange(family_count, device=raw.device).repeat(2)
        terrain.terrain_levels[:] = torch.tensor(levels, device=raw.device)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        command.started.zero_()
        print("[Audit] Resetting at L0/L9", flush=True)
        obs, _ = env.reset()
        policy_cfg = dict(agent["policy"])
        policy_cfg.pop("class_name")
        policy = PGMTForceActorCritic(obs, agent["obs_groups"], 29, **policy_cfg).to(raw.device).eval()
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format") == "instanttrack.pgmt_force_cop.v1":
            migrate_instanttrack_force(policy, args.checkpoint)
        else:
            if saved["policy_metadata"] != policy.get_checkpoint_metadata():
                raise ValueError("Stage2 policy metadata differs")
            policy.load_state_dict(saved["model_state_dict"], strict=True)
        initial = {k: v.cpu().numpy().copy() for k, v in obs.items()}
        model = raw.sim.mj_model
        feet = np.array([model.body("robot/" + side + "_ankle_roll_link").id for side in ("left", "right")])
        foot_geom = np.isin(model.geom_bodyid, feet) & ((model.geom_contype | model.geom_conaffinity) != 0)
        static_geom = (model.body_weldid[model.geom_bodyid] == 0) & ((model.geom_contype | model.geom_conaffinity) != 0)
        friction = wp.to_torch(raw.sim.wp_model.geom_friction).cpu().numpy()
        report = dict(device=args.device, num_envs=raw.num_envs, control_steps=args.steps,
            broadphase=str(raw.sim.wp_model.opt.broadphase), broadphase_override=args.broadphase,
            initialization_s=time.monotonic() - started, model_geoms=model.ngeom,
            initial_levels=levels, families=terrain.manifest["spec"]["families"],
            terrain_asset=cfg.commands["motion"].terrain_asset,
            allowed_spawn_patch_ids=terrain.allowed_patch_ids.cpu().tolist(),
            observation_shapes={k: list(v.shape) for k, v in obs.items()},
            physics_dt=raw.physics_dt, control_dt=raw.step_dt,
            foot_geom_friction_min=float(friction[:, foot_geom, 0].min()),
            foot_geom_friction_max=float(friction[:, foot_geom, 0].max()),
            terrain_friction_values=np.unique(friction[:, static_geom, 0]).tolist(),
            foot_priority_values=np.unique(model.geom_priority[foot_geom]).tolist(),
            terrain_priority_values=np.unique(model.geom_priority[static_geom]).tolist(),
            packed_contact_capacity=raw.sim.wp_data.naconmax, constraint_capacity=raw.sim.wp_data.njmax,
            all_finite=True, max_contacts=0, max_constraints=0, reward_group_sum_max_error=0.)
        resets = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
        feedback, contact_frictions = [], []
        rollout_started = time.monotonic()
        print("[Audit] Starting rollout", flush=True)
        for step in range(args.steps):
            with torch.inference_mode():
                action = policy.act_inference(obs)
                obs, reward, done, info = env.step(action)
            assert torch.isfinite(action).all() and torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            resets += done.long()
            feedback.append(obs["proprio_history"][:, -1, 96:].cpu().numpy().copy())
            d = raw.sim.wp_data
            count = int(d.nacon.numpy()[0])
            report["max_contacts"] = max(report["max_contacts"], count)
            report["max_constraints"] = max(report["max_constraints"], int(d.nefc.numpy().max()))
            report["reward_group_sum_max_error"] = max(report["reward_group_sum_max_error"],
                float((info["reward_groups"].sum(-1) - reward).abs().max()))
            pairs = d.contact.geom.numpy()[:count]
            select = ((foot_geom[pairs[:, 0]] & static_geom[pairs[:, 1]])
                      | (foot_geom[pairs[:, 1]] & static_geom[pairs[:, 0]]))
            contact_frictions.extend(d.contact.friction.numpy()[:count, 0][select].tolist())
            if (step + 1) % 10 == 0:
                print(f"[Audit] {step + 1}/{args.steps} control steps, {time.monotonic() - rollout_started:.1f}s", flush=True)
        feedback = np.stack(feedback)
        report.update(rollout_s=time.monotonic() - rollout_started,
            resets_per_env=resets.cpu().tolist(), cop_valid_fraction=feedback[..., [5, 11]].mean((0, 1)).tolist(),
            max_abs_force_n=float(np.abs(feedback[..., [0, 1, 2, 6, 7, 8]]).max()),
            foot_terrain_contact_friction_min=min(contact_frictions),
            foot_terrain_contact_friction_max=max(contact_frictions),
            contact_samples=len(contact_frictions),
            note="Capability audit with automatic resets; no completion-rate or convergence claim. Initial levels may change on reset.")
        assert report["max_contacts"] < report["packed_contact_capacity"]
        assert report["max_constraints"] < report["constraint_capacity"]
        np.savez_compressed(args.output / "observations.npz", **initial, feedback=feedback)
        (args.output / "rollout_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, family_count, figsize=(3 * family_count, 6), constrained_layout=True, squeeze=False)
        for i, axis in enumerate(axes.flat):
            value = np.ma.masked_where(initial["terrain_valid"][i] < .5, initial["terrain_map"][i])
            plot = axis.imshow(value, origin="lower", extent=(-1, 1, -1, 1), vmin=-.5, vmax=1.5, cmap="viridis")
            axis.set_title(f"L{report['initial_levels'][i]} {report['families'][i % family_count]}")
            axis.set_xlabel("Torso yaw X (m)")
            axis.set_ylabel("Y (m)")
        figure.colorbar(plot, ax=axes, label="Actual policy input: torso Z - ground Z (m); white = invalid/dropout")
        figure.suptitle("Stage2 actual 21 x 21 observations | 10% dropout and +/- 6 cm height noise enabled")
        figure.savefig(args.output / "terrain_observations.png", dpi=140)
        plt.close(figure)
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
