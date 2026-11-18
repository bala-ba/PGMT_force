"""Bounded, synchronized Stage2 profiling; does not change training settings.

Profiles the real force policy and physics. Nested phase times are inclusive;
the uninstrumented step timing is reported separately. Other GPU processes are
not stopped. Optional Kineto traces expose the expensive native CUDA kernels.
"""
import argparse
from collections import defaultdict
from functools import partial
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-asset", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-trace", action="store_true")
    parser.add_argument("--query-ablation", action="store_true", help="Compare rebuilt BVHs on actual runtime query coordinates")
    args = parser.parse_args()
    import numpy as np
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_agent
    from pmt_tasks.backends.mjlab_pgmt.force_feedback import enable_force_feedback_v1
    from pmt_tasks.backends.mjlab_pgmt.terrain import add_native_terrain
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    from motion_tracking_rl.networks import PGMTForceActorCritic
    from motion_tracking_rl.utils.pgmt_force_checkpoint import migrate_instanttrack_force

    torch.set_num_threads(4)
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    cfg.scene.num_envs, cfg.seed = args.num_envs, 42
    command_cfg = cfg.commands["motion"]
    command_cfg.motion_files = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    command_cfg.fixed_start_frame = 0
    if args.terrain_asset:
        command_cfg.terrain_asset = str(args.terrain_asset.resolve())
        cfg.scene.spec_fn = partial(add_native_terrain, asset=command_cfg.terrain_asset)
    agent = build_agent(build_task_config("pgmt_stage2"), stage=2)
    enable_force_feedback_v1(cfg, agent)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("[Profile] Building environment", flush=True)
    vector = PGMTVectorEnv(ManagerBasedRlEnv(cfg, device=args.device))
    raw = vector.env
    try:
        obs = vector.get_observations()
        policy_cfg = dict(agent["policy"])
        policy_cfg.pop("class_name")
        policy = PGMTForceActorCritic(obs, agent["obs_groups"], 29, **policy_cfg).to(args.device).eval()
        migrate_instanttrack_force(policy, args.checkpoint)
        def sync():
            torch.cuda.synchronize(args.device)
        def one_step():
            nonlocal obs
            with torch.inference_mode():
                action = policy.act_inference(obs)
                obs, reward, done, _ = vector.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            return int(done.sum())
        sync()
        build_s = time.perf_counter() - started
        print(f"[Profile] Environment built in {build_s:.1f}s; warmup", flush=True)
        one_step()
        sync()
        uninstrumented, resets = [], 0
        for i in range(args.steps):
            start = time.perf_counter()
            resets += one_step()
            sync()
            uninstrumented.append(time.perf_counter() - start)
            print(f"[Profile] Baseline {i + 1}: {uninstrumented[-1]:.4f}s", flush=True)
        report = dict(num_envs=args.num_envs, device=args.device, terrain_asset=command_cfg.terrain_asset,
            initialization_s=build_s, baseline_control_step_s=uninstrumented,
            baseline_samples_per_s=args.num_envs / np.mean(uninstrumented),
            ngeom=raw.sim.mj_model.ngeom, nhfield=raw.sim.mj_model.nhfield,
            broadphase=str(raw.sim.wp_model.opt.broadphase), decimation=cfg.decimation,
            collision_pair_arrays={name: list(getattr(raw.sim.wp_model, name).shape)
                for name in ("nxn_geom_pair", "nxn_geom_pair_filtered", "nxn_pairid", "nxn_pairid_filtered")})
        (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.gpu_trace:
            print("[Profile] Capturing one real control step with Kineto", flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                with torch.profiler.record_function("PGMT_control_step"):
                    resets += one_step()
                    sync()
            profile.export_chrome_trace(str(args.output / "cuda_trace.json"))
            (args.output / "cuda_profile.txt").write_text(profile.key_averages().table(
                sort_by="self_device_time_total", row_limit=40))
            report["cuda_kernels"] = sorted([dict(name=event.key, count=event.count,
                self_device_time_us=event.self_device_time_total, self_cpu_time_us=event.self_cpu_time_total)
                for event in profile.key_averages()], key=lambda x: -x["self_device_time_us"])[:40]
            print("[Profile] CUDA trace saved", flush=True)
        timings, originals, captured = defaultdict(list), [], {}
        def wrap(obj, name, label):
            original = getattr(obj, name)
            def timed(*args, **kwargs):
                if label == "nested_terrain_height" and args:
                    captured[args[0].numel() // 2] = args[0].detach().clone()
                sync()
                start = time.perf_counter()
                result = original(*args, **kwargs)
                sync()
                timings[label].append(time.perf_counter() - start)
                return result
            originals.append((obj, name, original))
            setattr(obj, name, timed)
        for obj, name, label in (
            (policy, "act_inference", "policy"),
            (raw.action_manager, "process_action", "process_action"),
            (raw.action_manager, "apply_action", "apply_action"),
            (raw.scene, "write_data_to_sim", "write_to_sim"),
            (raw.sim, "step", "physics_step"),
            (raw.scene, "update", "scene_update"),
            (raw.sim, "forward", "physics_forward"),
            (raw.sim, "sense", "physics_sensors"),
            (raw.termination_manager, "compute", "terminations"),
            (raw.reward_manager, "compute", "rewards"),
            (raw.command_manager, "compute", "command"),
            (raw.event_manager, "apply", "events"),
            (raw.observation_manager, "compute", "observations"),
            (vector.command.terrain, "height", "nested_terrain_height"),
            (vector.command.terrain, "distance", "nested_terrain_distance"),
            (raw._pgmt_force_feedback, "compute", "nested_force_feedback"),
        ):
            wrap(obj, name, label)
        try:
            for i in range(args.steps):
                resets += one_step()
                print(f"[Profile] Instrumented step {i + 1}/{args.steps}", flush=True)
        finally:
            for obj, name, original in originals:
                setattr(obj, name, original)
        report["phases"] = {name: dict(calls=len(values), mean_ms=1000 * np.mean(values),
            total_ms_per_control_step=1000 * sum(values) / args.steps) for name, values in timings.items()}
        report.update(resets=resets, caveat="Synchronized phase measurements introduce overhead; nested_* is already included in parent phases. GPU is shared with pre-existing jobs.")
        if args.query_ablation:
            terrain = vector.command.terrain
            original_mesh = terrain.mesh
            np.savez_compressed(args.output / "runtime_query_xy.npz", **{str(k): v.cpu().numpy() for k, v in captured.items()})
            reference, comparisons = {}, {}
            for kind in ("original", "lbvh", "sah"):
                mesh = original_mesh if kind == "original" else wp.Mesh(points=original_mesh.points,
                    indices=original_mesh.indices, bvh_constructor=kind)
                terrain.mesh = mesh
                sync()
                measurements = {}
                for count, xy in captured.items():
                    values, elapsed = None, []
                    for _ in range(3):
                        sync(); start = time.perf_counter()
                        values = terrain.height(xy)
                        sync(); elapsed.append(time.perf_counter() - start)
                    if kind == "original":
                        reference[count] = values.clone()
                    measurements[count] = dict(times_s=elapsed, finite=bool(torch.isfinite(values).all()),
                        max_error_m=float((values - reference[count]).abs().max()))
                comparisons[kind] = measurements
                print(f"[Profile] Actual queries, {kind}: {measurements}", flush=True)
                # Ensure a rebuilt BVH remains fast after real physics runs,
                # rather than measuring only an isolated, freshly built tree.
                step_times = []
                for _ in range(3):
                    sync(); start = time.perf_counter()
                    resets += one_step()
                    sync(); step_times.append(time.perf_counter() - start)
                comparisons[kind]["full_control_steps_s"] = step_times
                print(f"[Profile] Full control steps, {kind}: {step_times}", flush=True)
            terrain.mesh = original_mesh
            report["runtime_query_bvh_comparison"] = comparisons
        (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        vector.close()


if __name__ == "__main__":
    main()
