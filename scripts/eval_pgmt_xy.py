"""Full-clip native Isaac evaluation of XY tracking over the checkpoint terrain.

Each compatible (motion, terrain, level) is evaluated once, from frame zero.
Metrics are captured after physics and BEFORE native reset/command advancement.
Only the first episode in each batch slot contributes; failures are retained.
"""
import argparse
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("PMT_PGMT_PHYSICS_WORKER") != "1":
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_checkpoint", help="Optional actor checkpoint, e.g. original Stage1, under the same evaluation environment")
parser.add_argument("--record_rewards", action="store_true", help="Retain native rewards and capture contact/reward diagnostics including physics substeps")
parser.add_argument("--disable_terrain_tokens", action="store_true", help="Explicit inference ablation: omit terrain tokens after strictly loading the actor")
parser.add_argument("--families", nargs="+", help="Evaluate only these terrain families")
parser.add_argument("--flat_plane", action="store_true", help="Use a large isolated flat plane, eliminating adjacent-terrain crossings")
parser.add_argument("--log_dir", required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--levels", nargs="+", type=int, default=list(range(10)))
parser.add_argument("--seed", type=int, default=1729)
parser.add_argument("--yaw", type=float, default=0., help="Initial world heading, radians")
parser.add_argument("--motions", nargs="+", help="Subset of the checkpoint motion library")
parser.add_argument("--terrain_asset", help="Explicit common terrain for cross-checkpoint comparisons")
parser.add_argument("--spawn_offset_x", type=float, default=0., help="Offset from the assigned tile center")
parser.add_argument("--placement_search_radius", type=float, default=0., help="Find a shared collision-valid X offset near the requested one")
parser.add_argument("--align_path_x", action="store_true", help="Rotate a single motion's endpoint direction onto world +X")
parser.add_argument("--record_all_states", action="store_true")
parser.add_argument("--max_motions_per_family", type=int, default=0,
                    help="0 evaluates all compatible clips; positive values are for smoke tests")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless or args.num_envs < 1 or any(x not in range(10) for x in args.levels):
    parser.error("Require --headless, positive num_envs and levels 0..9")
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_path = output / "training_status.json"
if status_path.exists():
    parser.error("Refusing to overwrite an existing run")
status = dict(status="launching", pid=os.getpid(), training=False,
              run_id=os.environ.get("PMT_PGMT_RUN_ID"), started_at=datetime.now().isoformat())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_status(**values):
    status.update(values, updated_at=datetime.now().isoformat())
    write_json(status_path, status)


save_status()
stopping = False


def request_stop(*_):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic
    from motion_tracking_rl.utils.pgmt_checkpoint import canonical
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.mdp.commands.multi_motion_command import SamplingResult
    from pmt_tasks.pgmt_performance import install_performance
    from pmt_tasks.pgmt_physx import shared_terrain_scene
    from pmt_tasks.pgmt_sampling import load_terrain_compatibility
    from pmt_tasks.pgmt_terrain_layout import training_manifest, compatibility_families
    from pgmt_xy_report import trajectory_metrics, write_reports

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy_checkpoint = Path(args.policy_checkpoint).resolve() if args.policy_checkpoint else checkpoint
    policy_saved = torch.load(policy_checkpoint, map_location="cpu", weights_only=False) if args.policy_checkpoint else saved
    metadata = policy_saved["policy_metadata"]
    archived = saved["training_contract"]["runtime"]["environment"]
    if saved["policy_metadata"]["policy_family"] != "pgmt_terrain_tracking":
        raise ValueError("Requires a native PGMT terrain checkpoint")
    files = archived["commands"]["motion"]["motion_files"]
    if args.motions:
        selected_files = [str(Path(p).resolve()) for p in args.motions]
        if len(set(selected_files)) != len(selected_files) or not set(selected_files).issubset(files):
            raise ValueError("Selected motions must be unique members of the checkpoint library")
        files = selected_files
    asset = Path(args.terrain_asset or archived["pgmt_terrain_asset"]).resolve()
    if args.flat_plane:
        if args.levels != [0]:
            raise ValueError("The flat-plane baseline requires --levels 0")
        from pmt_tasks.pgmt_emd_flat_asset import build_flat_eval_asset
        asset = Path(build_flat_eval_asset(output / "flat_terrain", min(args.num_envs, len(files))))
    straightness = None
    if args.align_path_x:
        if len(files) != 1:
            raise ValueError("--align_path_x requires exactly one --motions clip")
        with np.load(files[0]) as motion:
            xy = motion["body_pos_w"][:, 0, :2].astype(float)
            q = motion["body_quat_w"][0, 0].astype(float)
        delta = xy[-1]-xy[0]
        distance = float(np.linalg.norm(delta))
        if distance < .1:
            raise ValueError("Motion endpoint displacement is too small for alignment")
        heading = float(np.arctan2(delta[1], delta[0]))
        source_yaw = float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
        args.yaw = source_yaw-heading
        path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
        lateral = np.abs((xy-xy[0])@np.array([-delta[1], delta[0]])/distance)
        straightness = dict(endpoint_distance_m=distance, path_length_m=path_length,
            distance_over_path=distance/path_length, max_lateral_deviation_m=float(lateral.max()),
            rotation_about_world_z_rad=-heading, source_initial_yaw_rad=source_yaw)
    manifest = training_manifest(json.loads((asset / "manifest.json").read_text()))
    families = manifest["spec"]["families"]
    if args.families and not set(args.families).issubset(families):
        raise ValueError("Unknown requested terrain family")
    compatibility_path = Path(archived["commands"]["motion"]["motion_terrain_compatibility"])
    allowed = load_terrain_compatibility(compatibility_path, files,
                                       families=compatibility_families(manifest)).numpy()
    # Interleave all terrain classes in each batch, avoiding batch/family confounding.
    trials = []
    selected = [np.flatnonzero(row).tolist() for row in allowed]
    if args.max_motions_per_family:
        selected = [indices[:args.max_motions_per_family] for indices in selected]
    for level in args.levels:
        for index in range(max(map(len, selected))):
            for family_id, mids in enumerate(selected):
                if args.families and families[family_id] not in args.families:
                    continue
                if index < len(mids):
                    mid = mids[index]
                    trials.append(dict(trial_id=len(trials), level=level, family=families[family_id],
                                       family_id=family_id, file_index=mid,
                                       motion=Path(files[mid]).stem, yaw_rad=args.yaw))
    if not trials:
        raise ValueError("Empty evaluation matrix")
    cfg = build_env_cfg("pgmt_stage2_terrain", overrides=[
        'terrain_reward_mode=' + archived.get('pgmt_terrain_reward_mode', 'legacy')])
    for field in ('pgmt_contact_events', 'pgmt_geometry_contacts', 'pgmt_support_risk', 'pgmt_touchdown_quality',
                  'pgmt_reference_contact_threshold_n'):
        if field in archived:
            setattr(cfg, field, deepcopy(archived[field]))
    cfg.pgmt_terrain_asset = str(asset)
    cfg.pmt_motion_paths = files
    cfg.pmt_decimation, cfg.pmt_sim_dt = archived["decimation"], archived["sim"]["dt"]
    cfg.pgmt_tracking_relaxation = deepcopy(archived["pgmt_tracking_relaxation"])
    cfg.__post_init__()
    if cfg.pgmt_touchdown_quality['settlement'] == 'touchdown_event_v1':
        from pmt_tasks.pgmt_touchdown import configure_touchdown_event
        configure_touchdown_event(cfg)
    if archived.get('pgmt_contact_switching_mode'):
        from pmt_tasks.pgmt_contact_geometry import configure_contact_switching
        configure_contact_switching(cfg, archived['pgmt_contact_switching_mode'])
    if args.flat_plane:
        cfg.scene.terrain.family_proportions = None
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.commands.motion.placement_mode = "origin"
        cfg.commands.motion.placement_patch_grid = None
    for name in ("terminations", "actions"):
        if canonical(getattr(cfg, name).to_dict()) != archived[name]:
            raise ValueError(f"Current {name} differs from the checkpoint contract")
    if canonical(cfg.scene.robot.to_dict()["actuators"]) != archived["scene"]["robot"]["actuators"]:
        raise ValueError("Robot actuator contract changed")
    cfg.scene.num_envs, cfg.seed = min(args.num_envs, len(trials)), args.seed
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = args.device != "cpu"
    cfg.sim.enable_scene_query_support = True
    cfg.curriculum.terrain_progress = None
    cfg.commands.motion.fixed_start_frame = 0
    cfg.commands.motion.placement_xy_mode = "center"
    cfg.commands.motion.placement_yaw_range = (args.yaw, args.yaw)
    if args.spawn_offset_x:
        cfg.commands.motion.placement_mode = "origin"
        cfg.commands.motion.placement_patch_grid = None
    cfg.commands.motion.motion_terrain_compatibility = str(compatibility_path)
    cfg.pgmt_terrain_observation = deepcopy(saved["policy_metadata"]["policy_kwargs"]["terrain_observation_cfg"])
    cfg.pgmt_terrain_observation["augmentation_enabled"] = False
    cfg.observations.proprio.enable_corruption = cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    # Rewards cannot affect deterministic inference; omit their expensive diagnostic kernels.
    if args.record_rewards:
        if canonical(cfg.rewards.to_dict()) != archived["rewards"]:
            raise ValueError("Current rewards differ from the checkpoint contract")
    else:
        cfg.rewards = {}
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    robot_usd = archived["scene"]["robot"]["spawn"]["usd_path"]
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=robot_usd, **shared)
    protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        policy_checkpoint=str(policy_checkpoint), policy_checkpoint_sha256=hashlib.sha256(policy_checkpoint.read_bytes()).hexdigest(),
        native_rewards_enabled=args.record_rewards,
        disable_terrain_tokens=args.disable_terrain_tokens,
        checkpoint_iteration=saved["iter"], physics="Isaac Sim / PhysX", torch_version=torch.__version__,
        terrain_asset=str(asset), terrain_manifest_sha256=hashlib.sha256((asset/"manifest.json").read_bytes()).hexdigest(),
        compatibility_sha256=hashlib.sha256(compatibility_path.read_bytes()).hexdigest(),
        motion_files=files, compatible_motion_counts=dict(zip(families, allowed.sum(-1).tolist())),
        motion_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        levels=args.levels, trials=trials, seed=args.seed, initial_world_yaw_rad=args.yaw,
        checkpoint_training_terrain=archived["pgmt_terrain_asset"],
        cross_terrain_evaluation=str(asset) != str(Path(archived["pgmt_terrain_asset"]).resolve()),
        spawn_offset_x_m=args.spawn_offset_x, straightness=straightness,
        num_envs=cfg.scene.num_envs, physics_dt=cfg.sim.dt, control_dt=cfg.sim.dt*cfg.decimation,
        actor="deterministic mean", observation_noise=False, random_pushes=False, random_dynamics=False,
        start_frame=0, placement="native collision-checked placement at tile center plus declared X offset",
        success="motion_finished AND NOT terminated AND NOT map_out_of_bounds; coincident time_out allowed",
        xy_metric="pelvis link origin minus episode-placed reference pelvis XY in world frame; after physics, before command advance/reset",
        aggregation="time-sample pooled XY metrics; success denominator includes every scheduled compatible trial",
        trajectory_layout="xy[control_step, batch_slot, (actual_x,actual_y,reference_x,reference_y)]; valid prefix length=control_steps",
        terrain_attribution="spawn family and level; native map allows crossing neighboring tiles; occupancy fractions reported",
        checkpoint_contract_checks=["terminations", "actions", "robot.actuators"],
        original_terminations=archived["terminations"], training=False)
    write_json(output / "protocol.json", protocol)
    dump_yaml(str(output / "env.yaml"), cfg)
    save_status(status="building_environment", total_trials=len(trials), completed_trials=0)
    with shared_terrain_scene(0) as physics_setup:
        raw = ManagerBasedRLEnv(cfg)
    try:
        performance = install_performance(raw, "placement,reset_stats,contact_sensor")
        command, terrain, robot = raw.command_manager.get_term("motion"), raw.scene.terrain, raw.scene["robot"]
        actual_files = [str(Path(m.source_file).resolve()) for m in command.data_store.motions]
        if set(actual_files) != set(files):
            raise ValueError("Loaded motion library differs from checkpoint")
        lookup = {p: i for i, p in enumerate(actual_files)}
        assigned = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
        command._draw_samples = lambda ids: SamplingResult(assigned[ids].clone(), torch.zeros_like(assigned[ids]))
        if args.placement_search_radius:
            if len(files) != 1 or len(args.levels) != 1 or len(trials) != raw.num_envs or not args.align_path_x:
                raise ValueError("Placement search requires one aligned clip and one simultaneous terrain batch")
            from pmt_tasks.pgmt_reference import initial_transform, place_on_surface
            terrain.terrain_types[:] = torch.tensor([t["family_id"] for t in trials], device=raw.device)
            terrain.terrain_levels.fill_(args.levels[0])
            centers = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            ids = torch.arange(raw.num_envs, device=raw.device)
            reference = command._source_reference(assigned, torch.zeros_like(assigned))
            candidates = np.arange(-args.placement_search_radius, args.placement_search_radius+.001, .025)
            candidates = sorted((float(args.spawn_offset_x+d) for d in candidates), key=lambda x: abs(x-args.spawn_offset_x))
            preflight = []
            selected_offset = None
            for offset in candidates:
                spawn = centers[:, :2].clone()
                spawn[:, 0] += offset
                rotation, translation = initial_transform(reference["body_pos"][:, 0], reference["body_quat"][:, 0],
                    spawn, torch.full((raw.num_envs,), args.yaw, device=raw.device))
                _, valid, gap = place_on_surface(reference, rotation, translation, terrain.bounds_for(ids), terrain,
                    command.collision_body_ids, command.collision_local_centres, command.collision_radii,
                    clearance=command.cfg.spawn_clearance, boundary_margin=command.cfg.boundary_margin, geometry_only=True)
                preflight.append(dict(offset_x_m=offset, valid=valid.cpu().tolist(), minimum_gap_m=gap.cpu().tolist()))
                if bool(valid.all()):
                    selected_offset = offset
                    break
            write_json(output/"placement_preflight.json", preflight)
            if selected_offset is None:
                raise RuntimeError("No common collision-valid start in the requested search range")
            protocol["requested_spawn_offset_x_m"] = args.spawn_offset_x
            args.spawn_offset_x = selected_offset
            protocol["spawn_offset_x_m"] = selected_offset
            write_json(output/"protocol.json", protocol)
            terrain.env_origins[:] = centers
            terrain.env_origins[:, 0] += selected_offset
            print(f"[XY] shared collision-valid spawn offset X={selected_offset:.3f} m", flush=True)
        env = RslRlVecEnvWrapper(raw)
        policy = PGMTActorCritic(env.get_observations(), metadata["obs_groups"], env.num_actions,
                                **deepcopy(metadata["policy_kwargs"])).to(args.device)
        actual = policy.get_checkpoint_metadata()
        for key in ("signature", "action_contract", "value_groups"):
            if actual[key] != metadata[key]:
                raise ValueError(f"Policy metadata mismatch: {key}")
        for key, schema in metadata["obs_schema"]["available_obs"].items():
            if actual["obs_schema"]["available_obs"][key] != schema:
                raise ValueError(f"Actor observation mismatch: {key}")
        if actual.get("perception_contract") != metadata.get("perception_contract"):
            raise ValueError("Actor perception contract mismatch")
        policy.load_state_dict(policy_saved["model_state_dict"], strict=True)
        if args.disable_terrain_tokens:
            policy.terrain_mode = "disabled"
        policy.eval().requires_grad_(False)
        terms = list(raw.termination_manager.active_terms)
        recording = dict(enabled=False)
        probe = None
        if args.record_rewards:
            from pgmt_reward_probe import RewardProbe
            probe = RewardProbe(raw, recording)
        native_compute = raw.termination_manager.compute

        def measured_compute():
            done = native_compute()
            if recording["enabled"]:
                k = recording["step"]
                active = recording["active"]
                xy = torch.cat((command.robot_body_pos_w[:, 0, :2], command.body_pos_w[:, 0, :2]), -1)
                if not torch.isfinite(xy[active]).all():
                    raise RuntimeError("Nonfinite active trajectory; evaluation invalid")
                recording["xy"][k] = xy
                located = terrain.tile_map.locate(xy[:, :2])
                recording["same_family"] += active & (located["family_index"] == terrain.terrain_types)
                recording["same_tile"] += active & (located["family_index"] == terrain.terrain_types) & (located["tile_level"] == terrain.terrain_levels)
                recording["counts"] += active
                fresh = active & done
                recording["flags"][fresh] = torch.stack([raw.termination_manager.get_term(n) for n in terms], -1)[fresh]
                recording["failed"][fresh] = raw.termination_manager.terminated[fresh]
                recording["final_frame"][fresh] = command.frame_ids[fresh]
                recording["active"] &= ~done
                # Store a small fixed set of full states for optional visual replay.
                if len(recording["visual_ids"]):
                    ids = recording["visual_ids"]
                    recording["states"][k] = torch.cat((robot.data.root_pos_w[ids], robot.data.root_quat_w[ids], robot.data.joint_pos[ids]), -1)
                    recording["body_pos"][k] = robot.data.body_pos_w[ids]
            return done

        raw.termination_manager.compute = measured_compute
        results, begin = [], time.monotonic()
        (output / "trajectories").mkdir()
        (output / "states").mkdir()
        # A reproducible locomotion clip is recorded on each terrain and level.
        candidates = [i for i, f in enumerate(files) if Path(f).stem.startswith("walk") and allowed[:, i].all()]
        visual_mid = candidates[len(candidates)//2] if candidates else int(np.flatnonzero(allowed.all(0))[0])
        for first in range(0, len(trials), raw.num_envs):
            if stopping:
                raise InterruptedError("Requested stop")
            batch = trials[first:first+raw.num_envs]
            padded = batch + [batch[-1]] * (raw.num_envs-len(batch))
            assigned[:] = torch.tensor([lookup[files[t["file_index"]]] for t in padded], device=raw.device)
            terrain.terrain_types[:] = torch.tensor([t["family_id"] for t in padded], device=raw.device)
            terrain.terrain_levels[:] = torch.tensor([t["level"] for t in padded], device=raw.device)
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            terrain.env_origins[:, 0] += args.spawn_offset_x
            raw.seed(args.seed + first)
            recording["enabled"] = False
            with torch.inference_mode():
                obs, _ = env.reset()
            if not torch.equal(command.motion_ids, assigned) or not command.frame_ids.eq(0).all():
                raise RuntimeError("Reset changed assigned clip/start frame")
            if not torch.allclose(command.placement_xy, terrain.env_origins[:, :2], atol=1e-5, rtol=0):
                raise RuntimeError("Unexpected spawn XY")
            lengths = command.data_store.motion_lengths[assigned].cpu().numpy()
            visual_ids = torch.tensor([i for i, t in enumerate(batch) if args.record_all_states or t["file_index"] == visual_mid], device=raw.device, dtype=torch.long)
            reference_rotation = command.reference_rotation.clone().cpu().numpy()
            reference_translation = command.reference_translation.clone().cpu().numpy()
            horizon = raw.max_episode_length + 2
            recording.update(enabled=True, active=torch.arange(raw.num_envs, device=raw.device) < len(batch),
                xy=torch.empty((horizon, raw.num_envs, 4), device=raw.device),
                counts=torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device),
                same_family=torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device),
                same_tile=torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device),
                flags=torch.zeros((raw.num_envs, len(terms)), dtype=torch.bool, device=raw.device),
                failed=torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device),
                final_frame=torch.full((raw.num_envs,), -1, dtype=torch.long, device=raw.device),
                visual_ids=visual_ids, states=torch.empty((horizon, len(visual_ids), 7+robot.num_joints), device=raw.device),
                body_pos=torch.empty((horizon, len(visual_ids), robot.num_bodies, 3), device=raw.device))
            save_status(status="running", completed_trials=first, total_trials=len(trials),
                        current_batch=first//raw.num_envs, elapsed_s=time.monotonic()-begin)
            last_heartbeat = time.monotonic()
            if probe is not None:
                probe.clear()
            with torch.inference_mode():
                for step in range(horizon):
                    if stopping:
                        raise InterruptedError("Requested stop")
                    recording["step"] = step
                    action = policy.act_inference(obs)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite policy action")
                    obs_dict, _, _, _, _ = raw.step(action)
                    obs = env._observation_tensordict(obs_dict)
                    if not recording["active"].any():
                        break
                    if time.monotonic()-last_heartbeat > 30:
                        save_status(current_step=step+1, elapsed_s=time.monotonic()-begin)
                        print(f"[XY] batch {first//raw.num_envs}, step {step+1}, completed {first}/{len(trials)}", flush=True)
                        last_heartbeat = time.monotonic()
            recording["enabled"] = False
            if recording["active"].any():
                raise RuntimeError("An episode ended without native termination")
            data = {k: recording[k].cpu().numpy() for k in ("counts", "flags", "failed", "final_frame", "same_family", "same_tile")}
            if probe is not None:
                probe.save(output / f"reward_probe_{first//raw.num_envs:04d}.npz", data["counts"][:len(batch)], [t["trial_id"] for t in batch])
            xy = recording["xy"][:step+1, :len(batch)].cpu().numpy()
            trajectory_file = f"trajectories/batch_{first//raw.num_envs:04d}.npz"
            # Remove inactive padding explicitly so a later reader cannot mistake resets for trajectories.
            for j, count in enumerate(data["counts"][:len(batch)]):
                xy[count:, j] = np.nan
            np.savez_compressed(output/trajectory_file, xy=xy, trial_ids=[t["trial_id"] for t in batch],
                                control_steps=data["counts"][:len(batch)], control_dt=raw.step_dt)
            if len(visual_ids):
                states = recording["states"].cpu().numpy()
                body_pos = recording["body_pos"].cpu().numpy()
                for vi, j in enumerate(visual_ids.cpu().tolist()):
                    t = batch[j]
                    n = int(data["counts"][j])
                    np.savez_compressed(output/"states"/f"trial_{t['trial_id']:05d}.npz",
                        root_pos=states[:n, vi, :3], root_quat=states[:n, vi, 3:7],
                        joint_pos=states[:n, vi, 7:], joint_names=robot.joint_names,
                        body_pos=body_pos[:n, vi], body_names=robot.body_names,
                        reference_rotation=reference_rotation[j], reference_translation=reference_translation[j],
                        xy=xy[:n, j], control_dt=raw.step_dt,
                        trial_json=json.dumps(t), terrain_asset=str(asset), robot_usd=robot_usd)
            with (output / "episodes.jsonl").open("a") as stream:
                for j, trial in enumerate(batch):
                    n = int(data["counts"][j])
                    flags = dict(zip(terms, data["flags"][j].tolist()))
                    success = bool(flags["motion_finished"] and not data["failed"][j] and not flags["map_out_of_bounds"])
                    row = dict(trial, success=success, control_steps=n, seconds=n*raw.step_dt,
                        motion_frames=int(lengths[j]), final_reference_frame=int(data["final_frame"][j]),
                        reference_completion=float(data["final_frame"][j]/max(1, lengths[j]-1)),
                        termination_flags=flags, failed=bool(data["failed"][j]),
                        same_family_fraction=float(data["same_family"][j]/n),
                        same_tile_fraction=float(data["same_tile"][j]/n),
                        trajectory_file=trajectory_file, trajectory_slot=j,
                        **trajectory_metrics(xy[:n, j]))
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                    results.append(row)
            completed = first + len(batch)
            elapsed = time.monotonic()-begin
            save_status(completed_trials=completed, total_trials=len(trials), elapsed_s=elapsed,
                        successful_trials=sum(r["success"] for r in results))
            print(f"[XY] {completed}/{len(trials)} complete; success={sum(r['success'] for r in results)/completed:.3%}; "
                  f"elapsed={elapsed:.1f}s", flush=True)
        for key, value in policy.state_dict().items():
            if not torch.equal(value.cpu(), policy_saved["model_state_dict"][key]):
                raise RuntimeError(f"Policy or normalizer mutated: {key}")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
            raise RuntimeError("Source checkpoint changed")
        write_reports(output, results, protocol)
        save_status(status="completed", completed_trials=len(results), total_trials=len(trials),
                    successful_trials=sum(r["success"] for r in results),
                    policy_state_unchanged=True, source_checkpoint_unchanged=True,
                    physics_setup=physics_setup, performance=performance, elapsed_s=time.monotonic()-begin)
    finally:
        raw.close()


def checked_main():
    try:
        main()
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise


from pmt_tasks.runtime_options import run_and_close
run_and_close(checked_main, app)
