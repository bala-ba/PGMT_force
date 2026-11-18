"""Force feedback V1: legacy migration, gradients, batched wrenches and reset."""
from pathlib import Path
from copy import deepcopy
import os

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from motion_tracking_rl.networks import PGMTActorCritic, PGMTForceActorCritic
from motion_tracking_rl.pgmt_force_feedback import force_feedback_contract
from motion_tracking_rl.utils.pgmt_force_checkpoint import migrate_stage1_force

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt"
GROUPS = dict(policy=["proprio"], policy_history=["proprio_history"],
              command_window=["command_window"], critic=["critic"])


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations():
    old = TensorDict(dict(proprio=torch.randn(3, 96), proprio_history=torch.randn(3, 10, 96),
                         command_window=torch.randn(3, 8, 61), critic=torch.randn(3, 251)), [3])
    new = old.clone()
    force = torch.randn(3, 10, 2, 6)
    force[..., :3] *= 100
    force[..., 3:5] *= .03
    force[..., 5] = torch.randint(0, 2, (3, 10, 2))
    new["proprio_history"] = torch.cat((old["proprio_history"], force.flatten(-2)), -1)
    return old, new


def test_legacy_checkpoint_and_force_migration_are_action_exact():
    if not SOURCE.exists():
        pytest.skip("Requires local pretrained Stage1 checkpoint")
    old_obs, new_obs = observations()
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    kwargs = saved["policy_metadata"]["policy_kwargs"]
    old = PGMTActorCritic(old_obs, GROUPS, 29, **kwargs).eval()
    old.load_state_dict(saved["model_state_dict"], strict=True)
    assert old.get_checkpoint_metadata() == saved["policy_metadata"]
    new = PGMTForceActorCritic(new_obs, GROUPS, 29, **kwargs).eval()
    migrated = migrate_stage1_force(new, SOURCE)
    assert migrated["initialized"] == ["force_history_projection.weight"]
    torch.testing.assert_close(old.act_inference(old_obs), new.act_inference(new_obs), atol=0, rtol=0)
    torch.testing.assert_close(old.evaluate(old_obs), new.evaluate(new_obs), atol=0, rtol=0)
    for key, value in saved["model_state_dict"].items():
        torch.testing.assert_close(new.state_dict()[key], value, atol=0, rtol=0)
    assert new.get_checkpoint_metadata()["obs_schema"]["available_obs"]["proprio_history"]["shape"] == [10, 108]


def test_force_projection_gets_gradients_and_changes_intent():
    _, obs = observations()
    policy = PGMTForceActorCritic(obs, GROUPS, 29)
    before = policy.act_inference(obs).detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    action = policy.act_inference(obs)
    loss = (action-torch.randn_like(action)).square().mean() + policy.evaluate(obs).square().mean()
    loss.backward()
    grad = policy.force_history_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0
    optimizer.step()
    assert not torch.equal(policy.act_inference(obs), before)
    no_force = obs.clone()
    no_force["proprio_history"][..., 96:] = 0
    assert not torch.equal(policy._encode(obs)[1], policy._encode(no_force)[1])


def test_invalid_cop_mask_keeps_force_but_ignores_cop():
    _, obs = observations()
    history = obs["proprio_history"]
    history[..., 101] = history[..., 107] = 0
    policy = PGMTForceActorCritic(obs, GROUPS, 29)
    torch.nn.init.normal_(policy.force_history_projection.weight, std=.1)
    corrupted = obs.clone()
    corrupted["proprio_history"][..., [99, 100, 105, 106]] = float("nan")
    torch.testing.assert_close(policy.act_inference(obs), policy.act_inference(corrupted), atol=0, rtol=0)
    force_changed = obs.clone()
    force_changed["proprio_history"][..., 96] += 100
    assert not torch.equal(policy.act_inference(obs), policy.act_inference(force_changed))
    with pytest.raises(ValueError, match="96"):
        PGMTActorCritic(obs, GROUPS, 29)


def test_configuration_is_explicit_and_old_contract_is_unchanged():
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_agent
    from pmt_tasks.backends.mjlab_pgmt.force_feedback import enable_force_feedback_v1
    from pmt_tasks.backends.mjlab_pgmt.runner import serialize
    from scripts.train_pgmt_mjlab import parse_args
    cfg = build_env_cfg("pgmt_stage1", backend="mjlab")
    old = serialize(cfg)
    agent = build_agent(build_task_config("pgmt_stage1"), stage=1)
    new = deepcopy(cfg)
    enable_force_feedback_v1(new, agent)
    assert serialize(cfg) == old
    assert agent["policy"]["class_name"] == "PGMTForceActorCritic"
    assert new.observations["proprio_history"].history_length == 10
    assert not parse_args([]).force_feedback_v1
    assert parse_args([]).learning_rate is None
    assert parse_args(["--learning-rate", "1e-5"]).learning_rate == 1e-5
    with pytest.raises(SystemExit):
        parse_args(["--learning-rate", "nan"])
    assert parse_args(["--stage", "2", "--force-feedback-v1", "--dry-run"]).force_feedback_v1


def test_batched_wrench_pair_sign_rotation_self_collision_and_reset():
    import warp as wp
    from pmt_tasks.backends.mjlab_pgmt.force_feedback import _accumulate, _features
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    wp.init()
    device = "cpu"
    def arr(x, dtype):
        return wp.array(np.array(x), dtype=dtype, device=device)
    # Slots 0/1 are opposite geom order; slot 4 is self contact; 5 is inactive garbage.
    pairs = arr([[0, 1], [1, 0], [0, 2], [0, 1], [1, 2], [0, 1]], wp.vec2i)
    worlds = arr([0, 0, 1, 1, 0, 0], int)
    positions = arr([[.1, .02, -.035], [0, -.02, -.035], [0, 0, -.035],
                     [10, .1, -.035], [0, 0, 0], [999, 999, 999]], wp.vec3)
    fs = np.array([[0, 0, 100, 0, 0, 0], [0, 0, -50, 0, 0, 0], [0, 0, 70, 0, 0, 0],
                   [20, 0, 200, 0, 0, 0], [0, 0, 999, 0, 0, 0], [0, 0, 999, 0, 0, 0]], np.float32)
    frames = np.tile([[0, 0, 1], [1, 0, 0], [0, 1, 0]], (6, 1, 1)).astype(np.float32)
    frames[1, 0] *= -1
    rotations = np.tile(np.eye(3, dtype=np.float32), (2, 2, 1, 1))
    rotations[1, 0] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    origins = np.zeros((2, 2, 3), np.float32); origins[1, 0, 0] = 10
    force = wp.zeros((2, 2), dtype=wp.vec3, device=device)
    moment = wp.zeros_like(force)
    normal = wp.zeros((2, 2), dtype=float, device=device)
    nacon = arr([5], int)
    wp.launch(_accumulate, dim=6, inputs=[nacon, pairs, worlds, positions, arr(frames, wp.mat33),
        arr(fs, wp.spatial_vector), arr([-1, 0, 1], int), arr([1, 0, 0], int), arr([0, 1], int),
        arr(origins, wp.vec3), arr(rotations, wp.mat33), force, moment, normal], device=device)
    np.testing.assert_allclose(force.numpy(), [[[0, 0, 150], [0, 0, 0]], [[0, -20, 200], [0, 0, 70]]], atol=1e-5)
    output = wp.zeros((2, 12), dtype=float, device=device)
    def features(reset, count=5):
        wp.launch(_features, dim=(2, 2), inputs=[force, moment, normal, arr(reset, wp.int64),
            4, -.035, 10., 10., arr([count], int), arr([0, 0], int), 6, 20, output], device=device)
        return output.numpy().copy()
    values = features([-1, -1])
    np.testing.assert_allclose(values[0, 3:6], [2/30, 1/150, 1], atol=1e-6)
    np.testing.assert_allclose(values[1, 3:6], [.1, 0, 1], atol=1e-6)
    assert not values[0, 6:].any()
    reset = features([-1, 4])
    np.testing.assert_array_equal(reset[0], values[0])
    assert not reset[1].any()
    assert np.isnan(features([-1, -1], count=6)[:, 0]).all()


@pytest.mark.skipif(os.environ.get("PMT_FORCE_INTEGRATION") != "1", reason="opt-in real MJWarp rollout")
def test_real_feedback_history_reset_and_raw_wrench_agree():
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_agent
    from pmt_tasks.backends.mjlab_pgmt.force_feedback import enable_force_feedback_v1
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    device = os.environ.get("PMT_FORCE_TEST_DEVICE", "cpu")
    cfg = build_env_cfg("pgmt_stage1", backend="mjlab")
    cfg.scene.num_envs, cfg.events = 2, {}
    cfg.commands["motion"].motion_files = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz")]
    cfg.commands["motion"].evaluation, cfg.commands["motion"].noise = True, False
    cfg.terminations = {"short": TerminationTermCfg(func=lambda env: env.episode_length_buf >= 20, time_out=True)}
    agent = build_agent(build_task_config("pgmt_stage1"), stage=1)
    enable_force_feedback_v1(cfg, agent)
    env = PGMTVectorEnv(ManagerBasedRlEnv(cfg, device=device))
    try:
        policy = PGMTForceActorCritic(env.get_observations(), GROUPS, 29,
            **{k:v for k,v in agent["policy"].items() if k != "class_name"}).to(device)
        migrate_stage1_force(policy, SOURCE)
        obs = env.get_observations()
        assert obs["proprio_history"].shape == (2, 10, 108)
        assert not obs["proprio_history"][...,96:].any()
        for _ in range(15):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy.act_inference(obs))
            torch.testing.assert_close(obs["proprio_history"][:, -1, :96], obs["proprio"])
            assert torch.isfinite(obs["proprio_history"]).all()
        fb, d = env.env._pgmt_force_feedback, env.env.sim.wp_data
        expected = np.zeros((2,2,6))
        n = int(d.nacon.numpy()[0]); pairs = d.contact.geom.numpy()[:n]
        forces = fb.contact_forces.numpy()[:n,:3]; pos = d.contact.pos.numpy()[:n]
        worlds = d.contact.worldid.numpy()[:n]; body = fb.body_ids.numpy()
        rotations = d.xmat.numpy(); origins = d.xpos.numpy()
        gf, terrain = fb.geom_foot.numpy(), fb.terrain.numpy()
        for i,(a,b) in enumerate(pairs):
            if gf[a]>=0 and terrain[b]: foot,sign=int(gf[a]),-1
            elif gf[b]>=0 and terrain[a]: foot,sign=int(gf[b]),1
            else: continue
            w=int(worlds[i]); r=rotations[w,body[foot]].T
            f=r@(sign*forces[i]); p=r@(pos[i]-origins[w,body[foot]])
            expected[w,foot,:3]+=f;expected[w,foot,3:]+=np.cross(p,f)
        np.testing.assert_allclose(fb.forces.numpy(),expected[:,:,:3],atol=5e-4,rtol=1e-5)
        np.testing.assert_allclose(fb.moments.numpy(),expected[:,:,3:],atol=1e-4,rtol=1e-5)
        current = obs["proprio_history"][:,-1,96:].reshape(2,2,6).cpu().numpy()
        assert current[...,5].sum()>0
        for w,foot in np.argwhere(current[...,5]>.5):
            f=expected[w,foot,:3];m=expected[w,foot,3:]
            np.testing.assert_allclose(current[w,foot,3:5],[(-.035*f[0]-m[1])/f[2],(m[0]-.035*f[1])/f[2]],atol=1e-5)
        saved = env.get_observations().clone()
        torch.testing.assert_close(saved["proprio_history"],env.get_observations()["proprio_history"],atol=0,rtol=0)
        env.env.reset(env_ids=torch.tensor([0],device=device))
        assert not env.get_observations()["proprio_history"][0,:,96:].any()
        assert env.get_observations()["proprio_history"][1,:,96:].abs().sum()>0
        # Env 1 reaches its timeout first, while env 0 is in the new episode.
        obs = env.get_observations()
        automatic_resets = 0
        for _ in range(8):
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy.act_inference(obs))
            if done.any():
                automatic_resets += int(done.sum())
                assert not obs["proprio_history"][done.bool(), :, 96:].any()
        assert automatic_resets > 0
    finally:
        env.close()
