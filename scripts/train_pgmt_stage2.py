"""Finite or production PGMT P2/P3 training with explicit migration / strict resume."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import subprocess
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('PMT_PGMT_PHYSICS_WORKER') != '1':
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', choices=('pgmt_stage2', 'pgmt_stage2_terrain'), default='pgmt_stage2')
source = parser.add_mutually_exclusive_group()
source.add_argument('--initialize-from', default=None)
source.add_argument('--resume', default=None)
parser.add_argument('--num_envs', type=int, default=160)
parser.add_argument('--gpus', help='Launch on comma-separated GPUs, e.g. 0,1; --num_envs is the global total')
parser.add_argument('--distributed', action='store_true', help='Use torchrun ranks (automatically enabled by --gpus)')
parser.add_argument('--max_iterations', type=int, default=25, help='Additional updates, or zero for a restore-only audit')
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--max_runtime_s', type=float, default=0., help='Stop after a completed PPO update; 0 means no time limit')
parser.add_argument('--stop_file', help='Graceful stop request, checked after each completed PPO update')
parser.add_argument('--env_id_bits', type=int, choices=range(5), default=0,
                    help='Shared-terrain broadphase AABB bits; environment collision filtering remains enabled')
parser.add_argument('--flat_only', action='store_true', help='Separate flat migration regression, not the formal terrain run')
parser.add_argument('--terrain-asset', help='Use a versioned terrain asset with its own family layout and proportions')
parser.add_argument('--terrain-reward-mode', choices=('geometry_v3', 'legacy'),
                    help='Four-head task: v3 geometry costs (default), or original rewards and 10 N thresholds')
parser.add_argument('--contact-switching-mode', choices=('geometry_v3', 'legacy'),
                    help='Override only T5; geometry_v3 uses substep mean terrain force and a -0.05 event weight')
parser.add_argument('--touchdown-event', action='store_true',
                    help='Override only T1: original per-touchdown height variance reward, weight 1.5')
parser.add_argument('--t1-t2-contact-threshold-n', type=float,
                    help='Set T1 terrain Fz and T2 contact-norm thresholds in N; preserve T3 and T5 thresholds')
parser.add_argument('--reference-run', help='Restore archived four-head Isaac environment/PPO settings (e.g. train_03)')
parser.add_argument('--force-feedback-v1', action='store_true', help='New InstantTrack runs use 99-D body plus 12-D current foot feedback, with separate body and force histories')
parser.add_argument('--force-feedback-audit', action='store_true', help='Short-test only: independently check PhysX manifolds, history and T1/T5 overrides on CPU')
parser.add_argument('--instanttrack-weights', choices=('ema', 'model'), default='ema')
parser.add_argument('--instanttrack-critic-init', choices=('source', 'fresh'), default='source',
                    help='Warm-start matching critic inputs, normalizer statistics, trunk and three value heads')
parser.add_argument('--learning-rate', type=float, help='Explicit override; otherwise retain task/reference-run PPO rate')
parser.add_argument('--log_dir')
parser.add_argument('--seed', type=int, default=722)
parser.add_argument('--acceptance', action='store_true', help='Controlled geometry, crossing and truncation audits before training')
parser.add_argument('--profile', choices=('perception', 'scan_only', 'blind'), default='perception')
parser.add_argument('--robot_usd')
parser.add_argument('--isaacsim6_compat', action='store_true')
parser.add_argument('--performance', default='none',
                    help='Comma-separated reset_stats,placement,contact_sensor,audit,reward_compile; none or all')
parser.add_argument('--tracking-relaxation', action='store_true', help='Enable the P3 lower tracking deadzones')
parser.add_argument('--tracking-relaxation-config', help='YAML mapping overriding tracking_relaxation settings')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from scripts.pgmt_distributed import RankLayout
layout = RankLayout.from_environment(os.environ)
try:
    local_num_envs = layout.local_envs(args.num_envs)
except ValueError as error:
    parser.error(str(error))
if args.distributed and not layout.distributed:
    parser.error('--distributed requires torchrun; use --gpus 0,1 for a local multi-GPU launch')
args.distributed = layout.distributed
if layout.distributed:
    if not args.headless or args.profile != 'perception':
        parser.error('Distributed PGMT training requires --headless and --profile perception')
    args.device = f'cuda:{layout.local_rank}'
run_seed = args.seed + layout.rank
from pmt_tasks.pgmt_performance import parse_features
performance_features = parse_features(args.performance)
if performance_features and args.task != 'pgmt_stage2_terrain':
    parser.error('Performance options require pgmt_stage2_terrain')
if args.num_envs < 5 or args.max_iterations < 0 or args.max_runtime_s < 0 or args.save_interval < 1:
    parser.error('At least five environments and nonnegative additional updates are required')
if args.profile != 'perception' and (args.resume or args.acceptance or not args.headless):
    parser.error('Baseline cost controls are headless, fresh and have no acceptance probes')
if args.task == 'pgmt_stage2_terrain' and args.profile != 'perception':
    parser.error('The formal four-head terrain task requires the perception profile')
if args.reference_run and (args.task != 'pgmt_stage2_terrain'
        or args.terrain_reward_mode or args.flat_only or args.tracking_relaxation or args.tracking_relaxation_config):
    parser.error('--reference-run requires the four-head terrain task; only explicit T1/T2/T5 overrides are supported')
if (args.contact_switching_mode or args.touchdown_event or args.t1_t2_contact_threshold_n is not None) and args.task != 'pgmt_stage2_terrain':
    parser.error('T1/T2/T5 overrides require the four-head terrain task')
if args.t1_t2_contact_threshold_n is not None and not 0 < args.t1_t2_contact_threshold_n < float('inf'):
    parser.error('--t1-t2-contact-threshold-n must be finite and positive')
if args.learning_rate is not None and not 0 < args.learning_rate < float('inf'):
    parser.error('--learning-rate must be finite and positive')
checkpoint = Path(args.resume or args.initialize_from or ROOT / 'logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt').resolve()
run_output = Path(args.log_dir or ROOT / ('logs/pgmt_p3_four_head' if args.task == 'pgmt_stage2_terrain' else 'logs/pgmt_p245')
              / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
output = layout.output_dir(run_output)
output.mkdir(parents=True, exist_ok=True)
status_path = output / 'training_status.json'
if status_path.exists():
    parser.error(f'Refusing to overwrite an existing run: {output}')
status = dict(status='launching', pid=os.getpid(), started_at=datetime.now().isoformat(),
              run_id=os.environ.get('PMT_PGMT_RUN_ID'), arguments=vars(args), checkpoint=str(checkpoint),
              output=str(output), completed_updates=0, rank=layout.rank, world_size=layout.world_size,
              local_num_envs=local_num_envs, total_num_envs=args.num_envs, seed=run_seed)
def save_status(**values):
    status.update(values, updated_at=datetime.now().isoformat())
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(status_path)
save_status()
stop_request = {'reason': None}
def request_stop(signum, frame):
    stop_request['reason'] = signal.Signals(signum).name
signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

class GracefulTrainingStop(Exception):
    pass
if args.isaacsim6_compat:
    from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
    install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    if args.isaacsim6_compat:
        from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
        install_rigid_scene_schema_bridge()
    import math
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from motion_tracking_rl import compat  # noqa: F401
    from motion_tracking_rl.runners import OnPolicyRunner, PGMTStage2Runner
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic
    from motion_tracking_rl.utils.pgmt_checkpoint import canonical
    from motion_tracking_rl.utils.pgmt_distributed import coordinated_stop, select_rank_state
    from pmt_tasks.builder import build_env_cfg, build_agent_cfg
    from pmt_tasks.agent_cfgs.pgmt import PGMTStage1PPORunnerCfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.pgmt_physx import shared_terrain_scene
    from pmt_tasks.pgmt_training_metrics import PGMTTrainingMetrics
    from pgmt_stage2_checks import equal_state, audit_geometry, RolloutAudit, save_debug, center_metrics

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
    if layout.distributed:
        # torch.cuda.nccl.version reports PyTorch's build-time NCCL version,
        # which can differ from a preloaded runtime library.
        import ctypes
        nccl_version = ctypes.c_int()
        if ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(nccl_version)) != 0:
            raise RuntimeError('Cannot query loaded NCCL runtime version')
        save_status(distributed_runtime=dict(backend='nccl', nccl_version=nccl_version.value,
            torch_nccl_build_version=list(torch.cuda.nccl.version()),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), nccl_library=os.environ.get('PGMT_NCCL_LIBRARY')))
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
    instanttrack = saved.get('format') == 'instanttrack.pgmt_force_cop.v1'
    force_source = saved.get('policy_metadata', {}).get('policy_family') == 'pgmt_stage2_force_v1'
    force_feedback = args.force_feedback_v1 or instanttrack or force_source
    if instanttrack and args.resume:
        raise ValueError('InstantTrack is weights-only initialization; use --initialize-from, not --resume')
    if force_feedback and (args.profile != 'perception' or not (instanttrack or force_source)):
        raise ValueError('Isaac force Stage2 requires an InstantTrack initialization or a Force Stage2 checkpoint')
    if args.force_feedback_audit and not force_feedback:
        raise ValueError('--force-feedback-audit requires a force-feedback policy')
    task_overrides = []
    if args.terrain_asset:
        terrain_asset = Path(args.terrain_asset).resolve()
        manifest = json.loads((terrain_asset / 'manifest.json').read_text())
        task_overrides = [f'terrain_asset={json.dumps(str(terrain_asset))}',
                          'terrain.family_proportions=' + json.dumps(manifest['spec'].get('family_proportions'))]
    if args.terrain_reward_mode:
        task_overrides.append(f'terrain_reward_mode={args.terrain_reward_mode}')
    historical_report = None
    if args.reference_run:
        from pmt_tasks.pgmt_historical_config import restore_historical_stage2, audit_historical_runtime
        cfg, agent, historical_report = restore_historical_stage2(args.reference_run)
        if args.terrain_asset:
            from pmt_tasks.pgmt_historical_config import override_historical_terrain
            override_historical_terrain(cfg, historical_report, args.terrain_asset)
    else:
        cfg = build_env_cfg(args.task, overrides=task_overrides)
        agent = build_agent_cfg(args.task, overrides=task_overrides) if args.profile == 'perception' else PGMTStage1PPORunnerCfg()
    if args.contact_switching_mode:
        from pmt_tasks.pgmt_contact_geometry import configure_contact_switching
        t5_override = configure_contact_switching(cfg, args.contact_switching_mode)
        if historical_report is not None:
            historical_report.setdefault('reward_overrides', []).append(t5_override)
    if args.touchdown_event:
        from pmt_tasks.pgmt_touchdown import configure_touchdown_event
        t1_override = configure_touchdown_event(cfg)
        if historical_report is not None:
            historical_report.setdefault('reward_overrides', []).append(t1_override)
    if args.t1_t2_contact_threshold_n is not None:
        from pmt_tasks.pgmt_contact_match import configure_t1_t2_contact_threshold
        threshold_override = configure_t1_t2_contact_threshold(cfg, args.t1_t2_contact_threshold_n)
        if historical_report is not None:
            historical_report.setdefault('reward_overrides', []).append(threshold_override)
    if args.tracking_relaxation or args.tracking_relaxation_config:
        import yaml
        from pmt_tasks.pgmt_tracking_relaxation import tracking_relaxation_config
        overrides = {}
        if args.tracking_relaxation_config:
            overrides = yaml.safe_load(Path(args.tracking_relaxation_config).read_text())
            if not isinstance(overrides, dict):
                raise ValueError('Tracking relaxation config must contain a YAML mapping')
        if args.tracking_relaxation:
            overrides['enabled'] = True
        cfg.pgmt_tracking_relaxation = tracking_relaxation_config(dict(cfg.pgmt_tracking_relaxation, **overrides))
    cfg.scene.num_envs, cfg.seed = local_num_envs, run_seed
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.enable_scene_query_support = True
    cfg.sim.use_fabric = True
    if args.robot_usd:
        from dataclasses import fields
        import isaaclab.sim as sim_utils
        usd = Path(args.robot_usd).resolve()
        if not usd.is_file():
            raise FileNotFoundError(usd)
        old = cfg.scene.robot.spawn
        shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
                  if f.name not in ('func', 'usd_path') and hasattr(old, f.name)}
        cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=str(usd), **shared)
    if args.profile == 'blind':
        from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTObservationsCfg
        cfg.observations = PGMTObservationsCfg()
        cfg.scene.height_scanner = None
    cfg.viewer.origin_type, cfg.viewer.asset_name, cfg.viewer.env_index = 'asset_root', 'robot', 0
    cfg.viewer.eye, cfg.viewer.lookat = (3., 3., 2.), (0., 0., 0.)
    agent.seed, agent.device = run_seed, args.device
    agent.max_iterations, agent.save_interval, agent.logger = args.max_iterations, args.save_interval, 'tensorboard'
    if args.learning_rate is not None:
        agent.algorithm.learning_rate = args.learning_rate
    if force_feedback:
        from pmt_tasks.pgmt_isaac_force_feedback import enable_isaac_force_feedback
        force_config = (saved['policy_metadata']['policy_kwargs']['force_feedback_cfg'] if force_source else None)
        layout_name = ("instanttrack99" if instanttrack else
                       saved["policy_metadata"]["policy_kwargs"].get("proprio_layout", "pgmt96"))
        current_force = instanttrack or saved["policy_metadata"]["policy_kwargs"].get("current_force_feedback", False)
        terrain_critic = instanttrack or saved["policy_metadata"]["policy_kwargs"].get("terrain_privileged_critic", False)
        enable_isaac_force_feedback(cfg, agent, force_config, proprio_layout=layout_name,
                                    current_force_feedback=current_force, terrain_privileged_critic=terrain_critic)
    if args.profile != 'perception':
        agent.algorithm.learning_rate = 1e-4  # Match the P2 default in the cost controls.
    if args.profile == 'perception':
        agent.policy.terrain_observation_cfg = deepcopy(cfg.pgmt_terrain_observation)
        semantic = cfg.to_dict()
        semantic.pop('viewer', None)  # GUI camera state does not affect training semantics.
        agent.pgmt_training_contract = dict(environment=canonical(semantic), profile=args.profile,
                                            physx_env_id_bits=args.env_id_bits, flat_only=args.flat_only)
        if layout.distributed:
            agent.pgmt_training_contract['distributed'] = dict(world_size=layout.world_size,
                total_num_envs=args.num_envs, rank=layout.rank, normalization='global_moments')
    dump_yaml(str(output / 'params/env.yaml'), cfg)
    dump_yaml(str(output / 'params/agent.yaml'), agent)
    (output / 'launch.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    save_status(status='building_environment', checkpoint_sha256=source_hash)
    def gpu_snapshot():
        return subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                               '--format=csv'], capture_output=True, text=True, check=True).stdout
    save_status(gpu_before=gpu_snapshot())
    with shared_terrain_scene(args.env_id_bits) as physics_setup:
        raw = ManagerBasedRLEnv(cfg=cfg)
    save_status(physics_setup=physics_setup, gpu_after_environment=gpu_snapshot())
    runner = None
    try:
        from pmt_tasks.pgmt_performance import install_performance, DeferredRewardAudit
        performance_report = install_performance(raw, performance_features)
        (output / 'performance.json').write_text(json.dumps(performance_report, indent=2) + '\n')
        save_status(performance=performance_report)
        if args.flat_only:
            terrain = raw.scene.terrain
            terrain.terrain_types.zero_()
            terrain.terrain_levels.zero_()
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env = RslRlVecEnvWrapper(raw)
        if historical_report is not None:
            audit_historical_runtime(raw, historical_report)
            if args.resume:
                historical_report['state'] = 'strict resume from the supplied checkpoint; reference run supplies configuration only'
            historical_report['runtime_overrides'] = dict(num_envs=local_num_envs, seed=run_seed,
                device=args.device, max_iterations=args.max_iterations, save_interval=args.save_interval,
                sampling_scope=getattr(raw.command_manager.get_term("motion").sampler, "kind", "global"),
                force_feedback=force_feedback, learning_rate=agent.algorithm.learning_rate)
            (output / 'historical_config.json').write_text(json.dumps(historical_report, indent=2) + '\n')
            save_status(historical_config=dict(archive_fields_match=True, runtime_curriculum_identity_match=True))
        if args.isaacsim6_compat and not args.headless:
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.enable_fabric(True)
        runner_cls = PGMTStage2Runner if args.profile == 'perception' else OnPolicyRunner
        runner = runner_cls(env, agent.to_dict(), log_dir=str(run_output), device=args.device)
        policy = runner.alg.policy
        four_head = policy.num_value_heads == 4
        if four_head:
            assert list(policy.value_groups) == ['upper', 'lower', 'terrain', 'auxiliary']
            assert len(raw.reward_manager.active_terms) == 27
            assert list(cfg.pgmt_reward_groups) == list(policy.value_groups)
            assert raw._pgmt_state.tracking_relaxation is not None or not cfg.pgmt_tracking_relaxation['enabled']
        if args.resume:
            saved = select_rank_state(saved, layout.world_size, layout.rank)
            runner.load(str(checkpoint))
            equal_state(policy.state_dict(), saved['model_state_dict'])
            equal_state(runner.alg.optimizer.state_dict(), saved['optimizer_state_dict'])
            expected_env = saved['environment_state_dict']
            old_curriculum = expected_env['curriculum']
            expected_env = dict(expected_env, curriculum=raw._pgmt_terrain_curriculum.upgrade_checkpoint(old_curriculum))
            sampler = raw.command_manager.get_term('motion').sampler
            pooled_prior = getattr(sampler, 'kind', None) == 'terrain_adaptive' and 'kind' not in expected_env['sampler']
            if pooled_prior:
                from pmt_tasks.pgmt_curriculum import terrain_sampler_checkpoint
                expected_env = dict(expected_env, sampler=terrain_sampler_checkpoint(sampler, expected_env['sampler']))
            equal_state(env.get_checkpoint_state(), expected_env)
            assert runner.completed_iterations == saved['next_iteration']
            assert runner.alg.learning_rate == saved['learning_rate']
            counts = expected_env['sampler']
            if counts.get('kind') == 'pbfm_emd':
                weights = counts['weights'].to(args.device)
                probabilities = weights / weights.sum()
            elif counts.get('kind') == 'terrain_adaptive':
                from pmt_tasks.pgmt_sampling import masked_motion_probabilities
                failure = counts['failed_motion_count'].to(args.device)
                success = counts['success_motion_count'].to(args.device)
                rates = failure / (failure + success + 1e-8)
                active = sampler.eligible.any(-1)
                probabilities = torch.zeros_like(rates)
                probabilities[active] = masked_motion_probabilities(rates[active], sampler.eligible[active],
                    beta=sampler.beta, uniform_ratio=sampler.uniform_ratio)
            else:
                failure = counts['failed_motion_count'].to(args.device)
                success = counts['success_motion_count'].to(args.device)
                probabilities = (failure / (failure + success + 1e-8)).clamp_min(1e-6).pow(sampler.beta)
                probabilities = probabilities / (probabilities.sum() + 1e-8)
                probabilities = probabilities * (1-sampler.uniform_ratio) + sampler.uniform_ratio/sampler.num_motions
                probabilities /= probabilities.sum()
            equal_state(sampler.get_sampling_probabilities(), probabilities)
            save_status(strict_restore=dict(model=True, normalizers=True, optimizer=True, curriculum=True,
                survival_schedule='fresh, no history in source' if old_curriculum['version'] == 1 else 'restored',
                sampler='legacy pooled counts split across compatible terrains' if pooled_prior else True,
                sampling_probabilities=True, learning_rate=True, next_iteration=runner.completed_iterations,
                physical_episode='fresh; no phantom sampler outcome'))
        elif instanttrack:
            from motion_tracking_rl.networks.pgmt_force_actor_critic import PGMTForceActorCritic
            from motion_tracking_rl.utils.pgmt_force_checkpoint import InstantTrack99Transfer
            robot = raw.scene["robot"]
            nominal = getattr(robot.data, "default_joint_pos_nominal", robot.data.default_joint_pos[0])
            scale = torch.as_tensor(raw.action_manager.get_term("joint_pos")._scale)
            if scale.ndim == 2:
                scale = scale[0]
            transfer = InstantTrack99Transfer(raw._pgmt_state.sdk_joint_order, nominal, scale)
            migration = runner.initialize_instanttrack_force(str(checkpoint), weights=args.instanttrack_weights,
                critic_init=args.instanttrack_critic_init, proprio_transfer=transfer)
            (output / 'migration.json').write_text(json.dumps(migration, indent=2) + '\n')
            # Reordered GEMM columns take different TF32 rounding paths on GPU.
            # Compare the migration itself in double precision on a small probe.
            obs = env.get_observations()[:64].to(device="cpu", dtype=torch.float64)
            old_kwargs = deepcopy(policy.get_checkpoint_metadata()['policy_kwargs'])
            for name in ('terrain_glimpse_cfg', 'terrain_mode', 'terrain_observation_cfg', 'value_groups',
                         'proprio_layout', 'current_force_feedback', 'terrain_privileged_critic'):
                old_kwargs.pop(name, None)
            old_groups = {name: policy.obs_groups[name] for name in ('policy', 'policy_history', 'command_window', 'critic')}
            source = saved[args.instanttrack_weights]
            old_obs = transfer.source_observations(obs, source['actor_obs_normalizer._mean'],
                                                   source['critic_obs_normalizer._mean'])
            old = PGMTForceActorCritic(old_obs, old_groups, num_actions=29, **old_kwargs).double().eval()
            old.load_state_dict(source, strict=True)
            probe_policy = deepcopy(policy).cpu().double().eval()
            probe_policy.terrain_mode = 'disabled'
            with torch.inference_mode():
                old_action, new_action = old.act_inference(old_obs), probe_policy.act_inference(obs)
                torch.testing.assert_close(new_action, old_action, rtol=1e-5, atol=1e-6)
                errors = dict(actor=float((old_action-new_action).abs().max()))
                if args.instanttrack_critic_init == 'source':
                    values = probe_policy.evaluate(obs)
                    mapped = values[:, [0, 1, 3]] if four_head else values
                    old_values = old.evaluate(old_obs)
                    torch.testing.assert_close(mapped, old_values, rtol=2e-6, atol=2e-5)
                    errors['critic_copied_heads'] = float((mapped-old_values).abs().max())
            policy.eval()
            del old, old_obs, source, probe_policy
            assert not runner.alg.optimizer.state and runner.completed_iterations == 0
            assert not raw._pgmt_terrain_curriculum.counts.any()
            save_status(migration_max_error=errors, migration_comparison_mode='matched_nominal_inputs_terrain_disabled',
                migrated_tensor_count=len(migration['copied']), initialized_tensor_count=len(migration['initialized']),
                critic_initialization=migration['critic'])
        elif args.profile == 'perception':
            same_policy = saved['policy_metadata']['policy_family'] == policy.get_checkpoint_metadata()['policy_family']
            migration = (runner.initialize_policy(str(checkpoint)) if same_policy else
                         runner.initialize_three_head(str(checkpoint)) if four_head else
                         runner.initialize_stage1(str(checkpoint)))
            (output / 'migration.json').write_text(json.dumps(migration, indent=2) + '\n')
            old_meta = saved['policy_metadata']
            obs = env.get_observations()
            if force_source:
                from motion_tracking_rl.networks.pgmt_force_actor_critic import PGMTForceActorCritic
            old_cls = PGMTForceActorCritic if force_source else PGMTActorCritic
            old = old_cls(obs, old_meta['obs_groups'], num_actions=29, **old_meta['policy_kwargs']).to(args.device)
            old.load_state_dict(saved['model_state_dict'], strict=True)
            old.eval()
            policy.eval()
            policy.terrain_mode = 'disabled' if old_meta['policy_family'] == 'pgmt_stage1' else old.terrain_mode
            with torch.inference_mode():
                values = policy.evaluate(obs)
                mapped_values = values[:, [0, 1, 3]] if four_head and not same_policy else values
                errors = dict(actor=float((old.act_inference(obs) - policy.act_inference(obs)).abs().max()),
                              critic=float((old.evaluate(obs) - mapped_values).abs().max()))
                if four_head and not same_policy:
                    assert values[:, 2].eq(0).all()
            assert errors['actor'] < 1e-6, errors
            if four_head:
                # Row expansion may change float32 GEMM reduction, while copied tensors are exact.
                torch.testing.assert_close(mapped_values, old.evaluate(obs), rtol=2e-6, atol=2e-5)
            else:
                assert errors['critic'] < 1e-6, errors
            policy.terrain_mode = 'enabled'
            del old
            assert not runner.alg.optimizer.state and runner.completed_iterations == 0
            assert not raw._pgmt_terrain_curriculum.counts.any()
            save_status(migration_max_error=errors, migration_disabled_max_error=errors,
                        migration_comparison_mode='blind' if old_meta['policy_family'] == 'pgmt_stage1' else 'perception',
                        migrated_tensor_count=len(migration['copied']),
                        initialized_tensor_count=len(migration['initialized']))
        else:
            assert policy.get_checkpoint_metadata()['signature'] == saved['policy_metadata']['signature']
            policy.load_state_dict(saved['model_state_dict'], strict=True)
        del saved
        if layout.distributed:
            # Fresh perception parameters must match before probes/checkpoints,
            # including zero-update migration runs.
            runner.alg.broadcast_parameters()
        if args.acceptance:
            audit_geometry(env, policy, output, save_status)
        obs = env.get_observations()
        probe_ids = torch.linspace(0,raw.num_envs-1,min(16,raw.num_envs),device=raw.device).long()
        probe = obs[probe_ids].clone()
        np.save(output / 'probe_families.npy', raw.scene.terrain.terrain_types[probe_ids].cpu().numpy())
        before = {name: p.detach().clone() for name, p in policy.named_parameters()}
        if four_head:
            final_name, final_layer = [(n, m) for n, m in policy.critic.named_modules() if isinstance(m, torch.nn.Linear)][-1]
            accumulated_head_gradient = torch.zeros(4, device=args.device)
            def collect_head_gradient(gradient):
                accumulated_head_gradient.add_(gradient.detach().abs().sum(-1))
            final_layer.weight.register_hook(collect_head_gradient)
        if args.profile == 'perception':
            save_debug(policy, probe, output / 'debug_before.npz')
            runner.save(str(run_output / 'initial.pt'))
        viewer = None
        if not args.headless:
            from pmt_tasks.pgmt_perception_viewer import PGMTPerceptionViewer
            viewer = PGMTPerceptionViewer(raw, policy, output)
            viewer.update(obs)
            for _ in range(5):
                raw.sim.render()
            viewer.capture('initial_viewport.png')
            viewer.global_view()
            for _ in range(5):
                raw.sim.render()
            viewer.capture('global_viewport.png')
            viewer.family.model.get_item_value_model().set_value(1)
            viewer.level.model.get_item_value_model().set_value(5)
            viewer.tile_view()
            for _ in range(5):
                raw.sim.render()
            viewer.capture('tile_stairs_L5.png')
            viewer.toggle()
            viewer.toggle()
            viewer.follow()
        audit = RolloutAudit(env, viewer) if args.acceptance or not args.headless else None
        original_step = env.step
        latest = [obs]
        reward_audit = dict(steps=0, done_events=0, timeout_events=0, max_group_sum_error=0.,
                            terrain_nonzero_samples=0)
        force_audits = []
        t5_audit = None
        if args.force_feedback_audit and raw._pgmt_state.support_switching:
            from pgmt_t5_checks import TerrainSwitchingAudit
            t5_audit = TerrainSwitchingAudit(raw)
        t1_audit = None
        if args.force_feedback_audit and raw._pgmt_state.touchdown_event_rewards:
            from pgmt_touchdown_checks import TouchdownEventAudit
            t1_audit = TouchdownEventAudit(raw)
        force_audit_step = 0
        telemetry = PGMTTrainingMetrics(raw) if args.profile == 'perception' and layout.rank == 0 else None
        deferred_audit = (DeferredRewardAudit(raw.reward_manager, policy.value_groups, raw.step_dt,
                                            raw.num_envs, args.device)
                          if four_head and 'audit' in performance_features else None)
        if deferred_audit is not None:
            original_returns = runner.alg.compute_returns
            def checked_returns(*a, **kw):
                # Stop on an invalid rollout before returns/GAE or PPO consumes it.
                reward_audit.update(deferred_audit.read())
                return original_returns(*a, **kw)
            runner.alg.compute_returns = checked_returns
        def step(actions):
            nonlocal force_audit_step
            if telemetry:
                telemetry.before_step()
            result = original_step(actions)
            if telemetry:
                telemetry.after_step()
            latest[0] = result[0]
            if args.force_feedback_audit:
                force_audit_step += 1
                if force_audit_step <= 8 or force_audit_step % 24 == 0 or result[2].any():
                    from pgmt_isaac_force_checks import audit_force_observation
                    force_audits.append(audit_force_observation(raw, result[0]))
                    (output / 'force_feedback_audit.json').write_text(json.dumps(force_audits, indent=2) + '\n')
                if t5_audit is not None:
                    (output / 'contact_switching_audit.json').write_text(json.dumps(t5_audit.rows, indent=2) + '\n')
                if t1_audit is not None:
                    (output / 'touchdown_event_audit.json').write_text(json.dumps(t1_audit.report(), indent=2) + '\n')
            if deferred_audit is not None:
                deferred_audit.add(result)
            elif four_head:
                grouped = result[3]['reward_groups']
                assert grouped.shape == (raw.num_envs, 4) and torch.isfinite(grouped).all()
                torch.testing.assert_close(grouped.sum(-1), result[1], rtol=2e-5, atol=2e-6)
                for i, name in enumerate(policy.value_groups):
                    indices = [j for j, term in enumerate(raw.reward_manager.active_terms) if term.startswith(name + '_')]
                    expected = raw.reward_manager._step_reward[:, indices].sum(-1) * raw.step_dt
                    torch.testing.assert_close(grouped[:, i], expected)
                reward_audit['steps'] += 1
                reward_audit['done_events'] += int(result[2].sum())
                reward_audit['timeout_events'] += int(result[3]['time_outs'].sum())
                reward_audit['terrain_nonzero_samples'] += int(grouped[:, 2].ne(0).sum())
                reward_audit['max_group_sum_error'] = max(reward_audit['max_group_sum_error'],
                                                         float((grouped.sum(-1)-result[1]).abs().max()))
            return result
        env.step = step
        updates = []
        old_log = runner.log
        torch.cuda.reset_peak_memory_stats(args.device)
        start_iteration = runner.current_learning_iteration
        learning_started = time.monotonic()
        stop_file = Path(args.stop_file).resolve() if args.stop_file else run_output / 'STOP_REQUESTED'
        def log(locs, *a, **kw):
            old_log(locs, *a, **kw)
            if not updates:
                runner.writer.add_text('Run/launch', '```json\n'+json.dumps(vars(args), indent=2)+'\n```', int(locs['it']))
                runner.writer.add_text('Run/reward_weights', '```json\n'+json.dumps({
                    name: raw.reward_manager.get_term_cfg(name).weight for name in raw.reward_manager.active_terms
                }, indent=2)+'\n```', int(locs['it']))
            losses = {key: float(value) for key, value in locs['loss_dict'].items()}
            assert all(math.isfinite(v) for v in losses.values())
            assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in policy.parameters())
            changes, gradients = {}, {}
            for group, prefix in [('selector', 'terrain_glimpse.selector'), ('patch_encoder', 'terrain_glimpse.patch_encoder'),
                                  ('ifm', 'intent_attention'), ('actor', 'actor'), ('critic', 'critic'),
                                  ('critic_terrain_encoder', 'critic_terrain_encoder.'),
                                  ('force_feedback', 'force_history_projection')]:
                params = [(n, p) for n, p in policy.named_parameters() if n.startswith(prefix)]
                if params:
                    changes[group] = max(float((p.detach()-before[n]).abs().max()) for n,p in params)
                    gradients[group] = sum(float(p.grad.abs().sum()) for _,p in params if p.grad is not None)
            record = dict(iteration=int(locs['it']), completed_iteration=int(locs['it'])+1,
                collection_s=float(locs['collection_time']), update_s=float(locs['learn_time']), losses=losses,
                learning_rate=runner.alg.learning_rate, parameter_change=changes, last_gradient_l1=gradients,
                peak_allocated_mib=torch.cuda.max_memory_allocated(args.device)/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(args.device)/2**20)
            if force_feedback:
                feedback = (latest[0]['force_history'][:, -1] if 'force_history' in latest[0]
                            else latest[0]['proprio_history'][:, -1, 96:]).reshape(raw.num_envs, 2, 6)
                record['force_feedback'] = {}
                for i, side in enumerate(('left', 'right')):
                    for label, value in (('force_norm_N', feedback[:, i, :3].norm(dim=-1).mean()),
                                         ('cop_valid_fraction', feedback[:, i, 5].mean())):
                        key = f'{side}_{label}'
                        record['force_feedback'][key] = float(value)
                        runner.writer.add_scalar('ForceFeedback/' + key, float(value), int(locs['it']))
            if four_head:
                head_change = (final_layer.weight.detach()-before[f'critic.{final_name}.weight']).abs().amax(-1)
                record.update(value_head_change=dict(zip(policy.value_groups, head_change.tolist())),
                              value_head_gradient_l1=dict(zip(policy.value_groups, accumulated_head_gradient.tolist())),
                              reward_group_audit=deepcopy(reward_audit))
                accumulated_head_gradient.zero_()
            if args.profile == 'perception':
                with torch.no_grad():
                    debug = policy.get_perception_debug(latest[0][::max(1, raw.num_envs//256)])
                record.update(valid_patch_fraction=float(debug.glimpse.token_valid.float().mean()),
                    coverage_mean=float(debug.glimpse.coverage.mean()),
                    terrain_attention_per_head=debug.terrain_attention_mass.mean((0,2)).tolist(),
                    **center_metrics(debug.glimpse.center_xy_m))
                # Match native Loss/* and Perf/* zero-based iteration axes.
                record['tensorboard_iteration'] = record['iteration']
                record['telemetry'] = telemetry.log(runner.writer, record['tensorboard_iteration'])
                for key in ('valid_patch_fraction', 'coverage_mean', 'center_overlap_fraction', 'center_boundary_fraction'):
                    runner.writer.add_scalar('Perception/' + key, record[key], record['tensorboard_iteration'])
                for head, mass in enumerate(record['terrain_attention_per_head']):
                    runner.writer.add_scalar(f'Perception/terrain_attention_head_{head}', mass, record['tensorboard_iteration'])
                for key in ('peak_allocated_mib', 'peak_reserved_mib'):
                    runner.writer.add_scalar('Memory/' + key, record[key], record['tensorboard_iteration'])
                runner.writer.flush()
            updates.append(record)
            with (output / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            save_status(status='training', completed_updates=len(updates), next_iteration=record['completed_iteration'],
                latest=record, checks=audit.checks if audit else {},
                curriculum_outcomes=raw._pgmt_terrain_curriculum.counts.sum((0,1)).tolist(),
                terrain_level_histogram=torch.bincount(raw.scene.terrain.terrain_levels, minlength=10).tolist())
            if viewer and len(updates) in (1,5,args.max_iterations):
                viewer.update(latest[0])
                raw.sim.render()
                viewer.capture(f'training_viewport_{len(updates)}.png')
            if len(updates) == 1 or len(updates) % 25 == 0:
                save_status(gpu_latest=gpu_snapshot())
        runner.log = log
        def iteration_completed(locs):
            # The logging hook runs only on rank zero. Stop decisions must be
            # collective or workers can deadlock in the next PPO all-reduce.
            if layout.rank != 0:
                save_status(status='training', completed_updates=int(locs['it']) + 1 - start_iteration,
                            next_iteration=int(locs['it']) + 1)
            if stop_file.exists():
                stop_request['reason'] = 'stop_file'
            if args.max_runtime_s and time.monotonic() - learning_started >= args.max_runtime_s:
                stop_request['reason'] = 'walltime_limit'
            stop_request['reason'] = coordinated_stop(stop_request['reason'], args.device, layout.distributed)
            if stop_request['reason']:
                raise GracefulTrainingStop(stop_request['reason'])
        runner.iteration_callback = iteration_completed
        save_status(status='training_ready', start_iteration=start_iteration,
                    policy_value_groups=list(policy.value_groups), observation_shapes={k:list(v.shape) for k,v in obs.items()})
        print(f'[{args.task}] {args.profile}: start={start_iteration}, updates={args.max_iterations}, output={output}', flush=True)
        if args.max_iterations:
            try:
                runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            except GracefulTrainingStop:
                runner.current_learning_iteration = runner.completed_iterations
        runner.save(str(run_output / 'final.pt'))
        command = raw.command_manager.get_term('motion')
        sampler = command.sampler
        probabilities = sampler.get_sampling_probabilities()
        if getattr(sampler, 'kind', None) == 'terrain_adaptive':
            from pmt_tasks.pgmt_sampling import marginal_motion_probabilities
            np.savez(output / 'motion_sampling_by_terrain.npz', families=np.asarray(sampler.families),
                     motions=np.asarray([Path(m.source_file).name for m in command.data_store.motions]),
                     eligible=sampler.eligible.cpu().numpy(), failure_rate=sampler.get_failure_probabilities().cpu().numpy(),
                     probability=probabilities.cpu().numpy())
            probabilities = marginal_motion_probabilities(probabilities, raw.scene.terrain.terrain_types)
        np.save(output / 'motion_sampling_probabilities.npy', probabilities.cpu().numpy())
        if args.profile == 'perception':
            save_debug(policy, probe, output / 'debug_after.npz')
            if args.max_iterations:
                required = ('terrain_glimpse.selector', 'terrain_glimpse.patch_encoder', 'intent_attention')
                for prefix in required:
                    assert any(not torch.equal(p, before[n]) for n,p in policy.named_parameters() if n.startswith(prefix)), prefix
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash
        if viewer:
            for _ in range(8):
                raw.sim.render()
        steady = updates[1:] or updates
        if layout.distributed:
            torch.distributed.barrier()
        save_status(status='completed', stop_reason=stop_request['reason'] or 'iteration_limit',
            training_walltime_s=time.monotonic()-learning_started,
            source_checkpoint_unchanged=True, final_checkpoint=str(run_output / 'final.pt'),
            next_iteration=getattr(runner, 'completed_iterations', start_iteration+len(updates)),
            mean_collection_s=float(np.mean([v['collection_s'] for v in steady])) if steady else None,
            mean_update_s=float(np.mean([v['update_s'] for v in steady])) if steady else None,
            simulator_version=version('isaacsim'),
            gpu_name=torch.cuda.get_device_name(args.device), gpu_after=gpu_snapshot(),
            active_rewards=raw.reward_manager.active_terms, reward_group_audit=reward_audit,
            tracking_relaxation=cfg.pgmt_tracking_relaxation)
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        raw.close()


def checked_main():
    try:
        main()
    except BaseException as error:
        save_status(status='failed', error=repr(error))
        raise
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

from pmt_tasks.runtime_options import run_and_close
run_and_close(checked_main, app)
