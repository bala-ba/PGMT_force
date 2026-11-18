"""Bounded v6/Force/T1-event/T5-v3/relaxation acceptance: rewards, PPO and restore."""
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
    parser.add_argument("--seed", type=int, default=722)
    parser.add_argument("--rollout-steps", type=int, default=24)
    args = parser.parse_args()
    if args.rollout_steps < 8:
        parser.error("At least eight rollout steps are required for the reset audit")
    import mujoco
    import numpy as np
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_agent
    from pmt_tasks.backends.mjlab_pgmt.force_feedback import enable_force_feedback_v1
    from pmt_tasks.backends.mjlab_pgmt.runner import MJLabPGMTRunner, serialize
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    from pmt_tasks.pgmt_tracking_relaxation import RELAXATION_TERMS

    torch.set_num_threads(4)
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    cfg.scene.num_envs, cfg.seed = 10, args.seed
    cfg.commands["motion"].motion_files = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.commands["motion"].fixed_start_frame = 0
    agent = build_agent(build_task_config("pgmt_stage2"), stage=2)
    enable_force_feedback_v1(cfg, agent)
    agent.update(num_steps_per_env=24, save_interval=1, seed=args.seed)
    agent["algorithm"]["learning_rate"] = 1e-6
    (args.output / "config.json").write_text(json.dumps(dict(environment=serialize(cfg), agent=serialize(agent)), indent=2))
    print("Building ten v6 environments, all five terrain families", flush=True)
    started = time.monotonic()
    raw = ManagerBasedRlEnv(cfg, device=args.device)
    runner = None
    try:
        vector = PGMTVectorEnv(raw)
        runner = MJLabPGMTRunner(vector, agent, log_dir=str(args.output), device=args.device)
        policy = runner.alg.policy
        before = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        runner.initialize(args.checkpoint)
        migration = runner.initialization
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["ema"]
        actual = policy.state_dict()
        for key in migration["copied"]:
            torch.testing.assert_close(actual[key].cpu(), source[key], rtol=0, atol=0)
        for key in migration["partially_copied"]:
            torch.testing.assert_close(actual[key][[0, 1, 3]].cpu(), source[key], rtol=0, atol=0)
            torch.testing.assert_close(actual[key][2].cpu(), before[key][2], rtol=0, atol=0)
        assert actual["critic.6.weight"][2].count_nonzero() > 0
        report = dict(initialization=migration, build_seconds=time.monotonic()-started,
            terrain=cfg.commands["motion"].terrain_asset, families=vector.command.terrain.compatibility_families,
            motions=vector.command.data_store.num_motions, groups=list(cfg.pgmt_reward_groups),
            reward_weights={k: v.weight for k, v in cfg.rewards.items()}, rollout=[], seed=args.seed,
            reward_contract=vector.command.terrain_rewards.contract,
            tracking_relaxation=vector.command.identity["tracking_relaxation"],
            t1_geometry_rays=0, t1_max_height_error_m=0.)
        assert report["tracking_relaxation"]["config"]["enabled"]
        native = mujoco.MjModel.from_binary_path(str(Path(cfg.commands["motion"].terrain_asset) / "source.mjb"))
        native_data = mujoco.MjData(native)
        mujoco.mj_forward(native, native_data)
        # Alternate L0/L9 per family, retaining the existing center spawn grid.
        vector.command.terrain.terrain_levels[1::2] = 9
        vector.command.started.zero_()
        vector.reset()
        initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        previous_valid = np.zeros((10, 2), dtype=bool)
        previous_support = previous_valid.copy()
        off_steps = np.zeros((10, 2), dtype=np.int64)
        episodes = np.full(10, -1)
        for i in range(args.rollout_steps):
            if i == 6:
                raw.episode_length_buf[0] = raw.max_episode_length-1
            old_frames = vector.command.frame_ids.clone()
            tick = time.monotonic()
            with torch.inference_mode():
                actions = policy.act_inference(vector.get_observations())
                obs, reward, done, info = vector.step(actions)
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(reward).all()
            assert info["reward_groups"].shape == (10, 4)
            torch.testing.assert_close(info["reward_groups"].sum(-1), reward, rtol=1e-5, atol=1e-6)
            relaxation = vector.command.tracking_relaxation_state()
            torch.testing.assert_close(relaxation["reference_frame_ids"], old_frames, rtol=0, atol=0)
            assert relaxation["sample_step"].eq(raw.common_step_counter).all()
            # Independent CPU metadata lookup at the actual pre-reset torso XY.
            xy = relaxation["anchor_xy_w"].cpu().numpy()
            levels, families = np.full(10, -1), np.full(10, -1)
            for tile in vector.command.terrain.manifest["tiles"]:
                bounds = np.asarray(tile["bounds"])
                inside = ((xy >= bounds[0]) & (xy < bounds[1])).all(-1)
                levels[inside], families[inside] = tile["level"], tile["family_group_index"]
            np.testing.assert_array_equal(relaxation["tile_level"].cpu().numpy(), levels)
            np.testing.assert_array_equal(relaxation["family_group_index"].cpu().numpy(), families)
            scale = np.where(np.isin(families, [1, 2, 3]), np.clip(levels/9., 0., 1.), 0.)
            np.testing.assert_allclose(relaxation["level_scale"].cpu().numpy(), scale, rtol=1e-6)
            relaxation_check = dict(active_envs=int((scale > 0).sum()), terms={})
            for name, maximum in RELAXATION_TERMS.items():
                params = cfg.pgmt_tracking_relaxation
                alpha = [params["alpha"][name].get(element, 1.)
                         for element in report["tracking_relaxation"]["names"][name]]
                margin = scale[:, None] * params[maximum] * np.asarray(alpha)[None]
                error = relaxation[name+"_raw_error"].cpu().numpy()
                residual = np.maximum(error-margin, 0.)
                sigma = vector.command.cfg.tracking[name.removeprefix("lower_")+"_sigma"]
                expected = np.exp(-np.square(residual).mean(-1)/sigma**2)
                actual_rate = vector.command._rewards[name].cpu().numpy()
                np.testing.assert_allclose(relaxation[name+"_margin"].cpu().numpy(), margin, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(relaxation[name+"_relaxed_error"].cpu().numpy(), residual, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(actual_rate, expected, rtol=1e-5, atol=1e-6)
                strict = np.exp(-np.square(error).mean(-1)/sigma**2)
                np.testing.assert_allclose(relaxation[name+"_strict_rate"].cpu().numpy(), strict, rtol=1e-5, atol=1e-6)
                assert np.all(actual_rate >= strict-1e-6)
                np.testing.assert_array_equal(actual_rate[scale == 0], relaxation[name+"_strict_rate"].cpu().numpy()[scale == 0])
                gain = float((actual_rate-strict).mean()) * cfg.rewards[name].weight * raw.step_dt
                np.testing.assert_allclose(float(info["log"][f"TrackingRelaxation/{name}_reward_gain"]), gain, rtol=1e-5, atol=1e-7)
                relaxation_check["terms"][name] = dict(max_margin=float(margin.max()), mean_reward_gain=gain)
            snapshot = vector.command.terrain_rewards.snapshot
            assert (snapshot["physics_samples"] == 4).all()
            torch.testing.assert_close(snapshot["frame_ids"], old_frames, rtol=0, atol=0)
            assert snapshot["contact"]["contact_valid"].all()
            assert not (vector.command.terrain_rewards.reader.counts_tensor > 64).any()
            t1, contact = snapshot["touchdown_quality"], snapshot["contact"]
            current_episodes = contact["episode_id"].cpu().numpy()
            reset = current_episodes != episodes
            previous_valid[reset] = False
            off_steps[reset] = 0
            forces = contact["terrain_forces_w"].cpu().numpy()
            valid = np.isfinite(forces).all(-1)
            support = valid & (forces[..., 2] > cfg.pgmt_contact_events["support_threshold_n"])
            history = previous_valid & valid
            event = history & support & ~previous_support & (off_steps >= cfg.pgmt_contact_events["touchdown_min_non_support_steps"])
            np.testing.assert_array_equal(t1["event"].cpu().numpy(), event)
            off_steps = np.where(valid & ~support, np.where(history, off_steps, 0)+1, 0)
            previous_valid, previous_support, episodes = valid, support, current_episodes.copy()
            heights = t1["height_w"].cpu().numpy()
            footprint_valid = t1["point_valid"].cpu().numpy().all(-1) & np.isfinite(heights).all(-1)
            variance = np.var(np.where(footprint_valid[..., None], heights, 0.), axis=-1)
            expected_quality = np.where(event & footprint_valid, np.exp(-variance/cfg.pgmt_touchdown_quality["sigma_m"]**2), 0.)
            np.testing.assert_allclose(t1["rate"].cpu().numpy(), expected_quality.sum(-1), rtol=1e-5, atol=1e-6)
            xy = t1["query_xy_w"].cpu().numpy()
            for world, foot in np.argwhere(footprint_valid):
                native_heights = []
                for point in xy[world, foot]:
                    origin = np.r_[point.astype(np.float64), vector.command.terrain.ray_top]
                    distance = mujoco.mj_ray(native, native_data, origin, np.array([0., 0., -1.]),
                                             None, True, -1, np.zeros(1, dtype=np.int32))
                    assert distance >= 0
                    native_heights.append(origin[2]-distance)
                error = float(np.max(np.abs(np.array(native_heights)-heights[world, foot])))
                assert error < 1e-4
                report["t1_geometry_rays"] += 9
                report["t1_max_height_error_m"] = max(report["t1_max_height_error_m"], error)
            if i == 6:
                assert done[0] and vector.command.frame_ids[0] == 0
            rates = {k: v.cpu().tolist() for k, v in vector.command._rewards.items() if k.startswith("terrain_")}
            assert all(torch.isfinite(vector.command._rewards[k]).all() for k in rates)
            report["rollout"].append(dict(seconds=time.monotonic()-tick, rates=rates,
                resets=int(done.sum()), physics_samples=snapshot["physics_samples"].cpu().tolist(),
                touchdown_events=int(event.sum()), rewarded_feet=int((expected_quality > 0).sum()),
                tracking_relaxation=relaxation_check,
                t1_integrated_sum=float(t1["rate"].sum()) * cfg.rewards["terrain_touchdown_quality"].weight * raw.step_dt))
            print(f"Reward step {i+1}: {report['rollout'][-1]['seconds']:.3f}s, resets={int(done.sum())}", flush=True)
        report["touchdown_events"] = sum(row["touchdown_events"] for row in report["rollout"])
        assert report["touchdown_events"] > 0 and report["t1_geometry_rays"] > 0
        assert any(row["tracking_relaxation"]["active_envs"] > 0 for row in report["rollout"])
        for name, maximum in RELAXATION_TERMS.items():
            assert max(row["tracking_relaxation"]["terms"][name]["max_margin"] for row in report["rollout"]) == cfg.pgmt_tracking_relaxation[maximum]
        (args.output / "acceptance.json").write_text(json.dumps(report, indent=2))
        print("Running one PPO update, including one-clip EMD evaluation", flush=True)
        runner.learn(1, init_at_random_ep_len=False)
        assert runner.completed_iterations == 1
        assert all(torch.isfinite(v).all() for v in policy.state_dict().values())
        final = policy.state_dict()
        report["critic_row_change_l1"] = (final["critic.6.weight"]-initial["critic.6.weight"]).abs().sum(-1).cpu().tolist()
        assert all(change > 0 for change in report["critic_row_change_l1"])
        report["adam"] = {}
        for prefix in ("actor.", "critic.", "force_history_projection.", "terrain_glimpse."):
            moments = [runner.alg.optimizer.state[p]["exp_avg"] for name, p in policy.named_parameters()
                       if name.startswith(prefix) and p in runner.alg.optimizer.state]
            assert moments and all(torch.isfinite(v).all() for v in moments)
            norm = sum(float(v.abs().sum()) for v in moments)
            assert norm > 0
            report["adam"][prefix] = norm
        checkpoint = args.output / "model_1.pt"
        runner.save(checkpoint)
        mismatched = vector.get_checkpoint_state()
        mismatched["identity"]["tracking_relaxation"]["config"]["enabled"] = False
        try:
            vector.validate_checkpoint_state(mismatched)
        except ValueError:
            report["rejects_changed_tracking_relaxation"] = True
        else:
            raise AssertionError("Checkpoint contract accepted a changed relaxation rule")
        runner.load(checkpoint)
        assert runner.completed_iterations == 1
        report.update(completed_iterations=1, transitions=240, checkpoint=str(checkpoint), strict_restore=True,
                      elapsed_seconds=time.monotonic()-started, passed=True)
        (args.output / "acceptance.json").write_text(json.dumps(report, indent=2))
        print(f"PASS: {args.output / 'acceptance.json'}", flush=True)
    finally:
        if runner is not None:
            runner.close()
        raw.close()


if __name__ == "__main__":
    main()
