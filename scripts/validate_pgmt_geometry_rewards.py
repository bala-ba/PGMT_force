"""Native PhysX/T1/T5 settlement, reset, and one-update PPO smoke validation."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--mode', choices=('geometry_v3', 'legacy'), default='geometry_v3')
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--num_envs', type=int, default=14)
parser.add_argument('--checkpoint', default=str(ROOT/'logs/pgmt_stage2_v8_emd/train_cuda0_20261112_104329/model_19999.pt'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    from pmt_tasks.builder import build_env_cfg, build_agent_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from motion_tracking_rl.runners import PGMTStage2Runner
    from motion_tracking_rl.utils.pgmt_checkpoint import canonical

    install_rigid_scene_schema_bridge()
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    overrides = ['terrain_reward_mode='+args.mode]
    cfg = build_env_cfg('pgmt_stage2_terrain', overrides=overrides)
    cfg.pmt_motion_paths = [str(ROOT/'assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz')]
    cfg.__post_init__()
    cfg.scene.num_envs, cfg.seed = args.num_envs, 1731
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = args.device != 'cpu'
    for name in ('physics_material', 'add_joint_default_pos', 'base_com', 'push_robot'):
        setattr(cfg.events, name, None)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ('func', 'usd_path') and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT/'logs/pgmt_p1_acceptance/native_robot/main.usd'), **shared)
    contract = canonical(cfg.to_dict())
    raw = ManagerBasedRLEnv(cfg)
    report = dict(mode=args.mode, steps=args.steps, environments=args.num_envs,
                  plane_sources=[0, 0, 0], contact_points_max=0, overflow_samples=0,
                  t1_range=[float('inf'), -float('inf')], t5_events=0,
                  support_samples=0, force_mean_max_error=0., reset_events=0)
    try:
        env = RslRlVecEnvWrapper(raw)
        state = get_pgmt_state(raw)
        terrain = raw.scene.terrain
        families = terrain.manifest['spec']['families']
        terrain.terrain_types[:] = torch.arange(raw.num_envs, device=raw.device) % len(families)
        terrain.terrain_levels[:] = (torch.arange(raw.num_envs, device=raw.device)//len(families)).remainder(2)*9
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        report['families'] = list(families)
        report['levels'] = terrain.terrain_levels.tolist()
        family_costs = [[] for _ in families]
        env.reset()
        agent = build_agent_cfg('pgmt_stage2_terrain', overrides=overrides)
        agent.device, agent.num_steps_per_env = args.device, 8
        agent.algorithm.num_learning_epochs = agent.algorithm.num_mini_batches = 1
        agent.logger, agent.save_interval = 'tensorboard', 1
        agent.pgmt_training_contract = dict(environment=contract)
        runner = PGMTStage2Runner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        runner.initialize_policy(args.checkpoint)
        actor = runner.get_inference_policy(device=args.device)
        physics = []
        original_update = raw.scene.update
        def update(dt):
            original_update(dt)
            physics.append(state.robot_foot_terrain_forces()[..., 2].clone())
        raw.scene.update = update
        start = time.monotonic()
        with torch.inference_mode():
            obs = env.get_observations()
            for k in range(args.steps):
                physics.clear()
                obs, reward, done, extras = env.step(actor(obs))
                assert torch.isfinite(reward).all()
                report['reset_events'] += int(done.sum())
                t1, t5 = state.touchdown_state(for_reward=True), state.contact_switching_state(for_reward=True)
                assert len(physics) == raw.cfg.decimation
                assert torch.isfinite(t1['rate']).all() and torch.isfinite(t5['rate']).all()
                report['t1_range'][0] = min(report['t1_range'][0], float(t1['rate'].min()))
                report['t1_range'][1] = max(report['t1_range'][1], float(t1['rate'].max()))
                for j in range(len(families)):
                    family_costs[j].append(t1['rate'][j::len(families)].clone())
                for name, rate in (('terrain_touchdown_quality', t1['rate']), ('terrain_contact_switching', t5['rate'])):
                    idx = raw.reward_manager.active_terms.index(name)
                    torch.testing.assert_close(raw.reward_manager._step_reward[:, idx],
                                               rate*raw.reward_manager.get_term_cfg(name).weight)
                if args.mode == 'geometry_v3':
                    assert t1['physics_samples'].eq(raw.cfg.decimation).all()
                    torch.testing.assert_close(t5['mean_force_z'], torch.stack(physics).mean(0))
                    error = (t5['mean_force_z']-torch.stack(physics).mean(0)).abs().max()
                    report['force_mean_max_error'] = max(report['force_mean_max_error'], float(error))
                    assert (t1['rate'] >= 0).all() and (t1['rate'] <= 1).all()
                    support = t1['support'] & t1['support_valid']
                    report['support_samples'] += int(support.sum())
                    report['contact_points_max'] = max(report['contact_points_max'], int(t1['contact_count'].max()))
                    report['overflow_samples'] += int(t1['contact_overflow'].sum())
                    for j in range(3):
                        report['plane_sources'][j] += int(((t1['plane_source'] == j) & support).sum())
                    report['t5_events'] += int(t5['event'].sum())
                    if k == args.steps//2:
                        terminal = t1['rate'].clone()
                        keep = state.support_tracker.state[1:].clone()
                        raw._reset_idx(torch.tensor([0], device=raw.device))
                        assert not state.support_tracker.valid[0].any()
                        torch.testing.assert_close(state.support_tracker.state[1:], keep)
                        torch.testing.assert_close(state.touchdown_state(for_reward=True)['rate'], terminal)
                else:
                    from pmt_tasks.pgmt_contact_switching import evaluate_contact_switching
                    s = state.contact_state(for_reward=True)
                    # The frozen foot/context rows remain correct even after reset.
                    # T5 is exactly the untouched legacy shared-event kernel.
                    expected = evaluate_contact_switching(s['fast_contact_switch'], s['contact_history_valid'])
                    torch.testing.assert_close(t5['rate'], expected['rate'], rtol=0, atol=0)
                    assert cfg.pgmt_contact_events['contact_threshold_n'] == 10.
        report['rollout_seconds'] = time.monotonic()-start
        report['t1_mean_by_family'] = {name: float(torch.cat(rows).mean())
                                      for name, rows in zip(families, family_costs)}
        raw.scene.update = original_update
        runner.learn(1, init_at_random_ep_len=False)
        runner.save(output/'final.pt')
        runner.load(output/'final.pt')
        report.update(passed=True, ppo_updates=1, strict_new_checkpoint_resume=True,
                      force_threshold_n=cfg.pgmt_contact_events['contact_threshold_n'])
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        raw.close()


try:
    main()
finally:
    app.close()
