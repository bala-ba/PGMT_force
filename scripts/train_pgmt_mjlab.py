"""Train PGMT Stage1/Stage2 with mjlab (MuJoCo Warp), independently of Isaac."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None, *, stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=stage or 1)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--initialize-from", "--initialize_from", dest="initialize_from")
    source.add_argument("--resume")
    parser.add_argument("--force-feedback-v1", action="store_true",
                        help="Append bilateral force/CoP/validity to 108-D history (auto-detected from checkpoints)")
    parser.add_argument("--instanttrack-weights", choices=("ema", "model"), default="ema",
                        help="InstantTrack weights-only migration: EMA (its evaluation default) or live model")
    parser.add_argument("--motion-files", "--motion_files", nargs="+")
    parser.add_argument("--terrain-asset", help="Audited native asset directory containing source.mjb and manifest.json")
    parser.add_argument("--compatibility-file", help="Stage2 motion_terrain_compatibility.csv")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--max-iterations", "--max_iterations", type=int, default=10000,
                        help="Additional PPO updates; zero performs initialization/restore and saves")
    parser.add_argument("--num-steps-per-env", "--num_steps_per_env", type=int, default=24)
    parser.add_argument("--learning-rate", type=float,
                        help="Override initial PPO learning rate; useful for weights-only post-training")
    parser.add_argument("--save-interval", "--save_interval", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", "--log_dir")
    parser.add_argument("--headless", action="store_true", help="Accepted for compatibility; this training entry is headless")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and motion assets without starting simulation")
    parser.add_argument("--max-runtime-s", type=float, default=0.)
    parser.add_argument("--stop-file", help="Stop after a complete PPO update when this file exists")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if stage is not None and args.stage != stage:
        parser.error(f"This entry point is for Stage{stage}")
    if min(args.num_envs, args.num_steps_per_env, args.save_interval, args.torch_threads) < 1:
        parser.error("Environment count, rollout length, save interval and thread count must be positive")
    if args.max_iterations < 0 or args.max_runtime_s < 0:
        parser.error("Iterations and runtime must be nonnegative")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("learning-rate must be finite and positive")
    if args.stage == 2 and not (args.initialize_from or args.resume or args.dry_run):
        parser.error("Stage2 requires --initialize-from Stage1.pt or --resume Stage2.pt")
    if args.stage == 1 and (args.terrain_asset or args.compatibility_file):
        parser.error("Terrain and compatibility options apply to Stage2")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("The PGMT mjlab entry currently supports one training process")
    return args


class GracefulStop(Exception):
    pass


def main(argv=None, *, stage=None):
    args = parse_args(argv, stage=stage)
    os.environ["PMT_BACKEND"] = "mjlab"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pgmt-mjlab-matplotlib")
    import torch
    import warp as wp
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_agent
    from pmt_tasks.backends.mjlab_pgmt.motion import MotionLibrary
    from pmt_tasks.backends.mjlab_pgmt.runner import MJLabPGMTRunner, serialize
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    from mjlab.envs import ManagerBasedRlEnv

    torch.set_num_threads(args.torch_threads)
    wp.config.kernel_cache_dir = os.environ.get("PMT_WARP_CACHE", "/tmp/pgmt-mjlab-warp")
    task_name = f"pgmt_stage{args.stage}"
    task = build_task_config(task_name)
    cfg = build_env_cfg(task_name, backend="mjlab")
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    command = cfg.commands["motion"]
    if args.motion_files:
        command.motion_files = [str(Path(path).expanduser().resolve()) for path in args.motion_files]
    if args.terrain_asset:
        from functools import partial
        from pmt_tasks.backends.mjlab_pgmt.terrain import add_native_terrain
        command.terrain_asset = str(Path(args.terrain_asset).expanduser().resolve())
        cfg.scene.spec_fn = partial(add_native_terrain, asset=command.terrain_asset)
    if args.compatibility_file:
        command.compatibility_file = str(Path(args.compatibility_file).expanduser().resolve())
    agent = build_agent(task, stage=args.stage)
    source_path = args.resume or args.initialize_from
    if source_path:
        saved = torch.load(source_path, map_location="cpu", weights_only=False)
        args.force_feedback_v1 |= (saved.get("format") == "instanttrack.pgmt_force_cop.v1"
            or saved.get("policy_metadata", {}).get("policy_family") in
            ("pgmt_stage1_force_v1", "pgmt_stage2_force_v1"))
        del saved
    if args.force_feedback_v1:
        from pmt_tasks.backends.mjlab_pgmt.force_feedback import enable_force_feedback_v1
        enable_force_feedback_v1(cfg, agent)
    agent.update(seed=args.seed, num_steps_per_env=args.num_steps_per_env, save_interval=args.save_interval)
    if args.learning_rate is not None:
        agent["algorithm"]["learning_rate"] = args.learning_rate
    # Small smoke runs still need valid minibatches; full runs retain four.
    agent["algorithm"]["num_mini_batches"] = min(4, args.num_envs * args.num_steps_per_env)
    if args.dry_run:
        library = MotionLibrary(command.motion_files, device="cpu", dt=cfg.decimation * cfg.sim.mujoco.timestep,
                                chunk_length=command.chunk_length, contact_labels=args.stage == 2)
        if args.stage == 2:
            import mujoco
            from pmt_tasks.backends.mjlab_pgmt.terrain import load_manifest
            from pmt_tasks.pgmt_terrain_layout import compatibility_families
            from pmt_tasks.pgmt_sampling import load_terrain_compatibility
            manifest = load_manifest(command.terrain_asset)
            compatible = load_terrain_compatibility(command.compatibility_file, library.files,
                                                   families=compatibility_families(manifest))
            if not compatible.any(-1).all():
                raise ValueError("At least one terrain family has no compatible motion")
            scene = mujoco.MjSpec()
            cfg.scene.spec_fn(scene)
            scene.compile()
        print(json.dumps(dict(backend="mjlab", stage=args.stage, motions=library.num_motions,
                              observations=list(cfg.observations), reward_terms=list(cfg.rewards),
                              policy_order="pmt_bfs", device=args.device), indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. MuJoCo Warp training needs a working NVIDIA driver; --device cpu is for small smoke checks.")
    for path in (args.resume, args.initialize_from):
        if path and not Path(path).is_file():
            raise FileNotFoundError(path)
    output = Path(args.log_dir or ROOT / "logs" / f"pgmt_stage{args.stage}_mjlab" /
                  datetime.now().strftime("%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "training_status.json"
    if status_path.exists() or any(output.glob("model_*.pt")):
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}; resume into a new log directory")
    status = dict(backend="mjlab", stage=args.stage, arguments=vars(args), completed_iterations=0)
    def save_status(**values):
        status.update(values, updated_at=datetime.now().isoformat())
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)
    (output / "config.json").write_text(json.dumps(dict(environment=serialize(cfg), agent=serialize(agent)), indent=2) + "\n")
    stop = {"reason": None}
    def request_stop(signum, frame):
        stop["reason"] = signal.Signals(signum).name
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    runner = env = raw = None
    started = time.monotonic()
    save_status(status="building_environment")
    try:
        raw = ManagerBasedRlEnv(cfg, device=args.device)
        env = PGMTVectorEnv(raw)
        runner = MJLabPGMTRunner(env, agent, log_dir=str(output), device=args.device)
        if args.resume:
            runner.load(args.resume)
        elif args.initialize_from:
            runner.initialize(args.initialize_from, instanttrack_weights=args.instanttrack_weights)
        def iteration_callback(info):
            save_status(status="training", completed_iterations=runner.completed_iterations)
            if args.stop_file and Path(args.stop_file).exists():
                stop["reason"] = "stop_file"
            if args.max_runtime_s and time.monotonic() - started >= args.max_runtime_s:
                stop["reason"] = "max_runtime_s"
            if stop["reason"]:
                raise GracefulStop(stop["reason"])
        runner.iteration_callback = iteration_callback
        save_status(status="training", completed_iterations=runner.completed_iterations)
        try:
            if args.max_iterations:
                runner.learn(args.max_iterations, init_at_random_ep_len=False)
        except GracefulStop:
            pass
        checkpoint = output / f"model_{runner.completed_iterations}.pt"
        runner.save(checkpoint)
        save_status(status="stopped" if stop["reason"] else "completed", reason=stop["reason"],
                    completed_iterations=runner.completed_iterations, checkpoint=str(checkpoint),
                    elapsed_s=time.monotonic() - started)
        print(f"Saved {checkpoint}", flush=True)
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise
    finally:
        if runner is not None:
            runner.close()
        if raw is not None:
            raw.close()


if __name__ == "__main__":
    main()
