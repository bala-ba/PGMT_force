"""PGMT MJWarp contracts; simulator tests are enabled with PMT_MJLAB_INTEGRATION=1."""
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch

from pmt_tasks.backends.mjlab_pgmt.motion import BFS_BODIES, BFS_JOINTS, MotionLibrary, BinSampler


def clip(path, frames=7, **changes):
    q = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 100
    quat = np.zeros((frames, 30, 4), dtype=np.float32)
    quat[..., 0] = 1
    data = dict(fps=np.array(50.), joint_pos=q, joint_vel=q * 0,
                body_pos_w=np.zeros((frames, 30, 3), dtype=np.float32), body_quat_w=quat,
                body_lin_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((frames, 30, 3), dtype=np.float32))
    data.update(changes)
    np.savez(path, **data)
    return data


def test_library_keeps_all_files_and_clamps_to_segment(tmp_path):
    a = clip(tmp_path / "a.npz")
    clip(tmp_path / "b.npz", frames=3)
    lib = MotionLibrary(tmp_path, device="cpu", dt=.02, chunk_length=4)
    assert lib.motion_lengths.tolist() == [4, 3, 3]
    ref = lib.gather(torch.tensor([[0, 1, 2]]), torch.tensor([[100, 100, -5]]))
    np.testing.assert_array_equal(ref["joint_pos"][0, 0], a["joint_pos"][3])
    np.testing.assert_array_equal(ref["joint_pos"][0, 1], a["joint_pos"][6])
    assert len(lib.identity["sha256"]) == 2


def test_named_motion_is_reordered_into_bfs(tmp_path):
    data = clip(tmp_path / "original.npz")
    data["joint_names"] = np.array(BFS_JOINTS[::-1])
    data["body_names"] = np.array(BFS_BODIES[::-1])
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        data[key] = data[key][:, ::-1]
    np.savez(tmp_path / "reordered.npz", **data)
    lib = MotionLibrary(tmp_path, device="cpu", dt=.02)
    ref = lib.gather(torch.tensor([0, 1]), torch.tensor([3, 3]))
    torch.testing.assert_close(ref["joint_pos"][0], ref["joint_pos"][1], rtol=0, atol=0)


@pytest.mark.parametrize("change", [dict(fps=60.), dict(joint_pos=np.full((7, 29), np.nan)),
                                    dict(body_quat_w=np.zeros((7, 30, 4)))])
def test_library_rejects_bad_data(tmp_path, change):
    clip(tmp_path / "bad.npz", **change)
    with pytest.raises(ValueError):
        MotionLibrary(tmp_path, device="cpu", dt=.02)


def test_bin_sampler_updates_start_bins_and_roundtrips():
    cfg = dict(alpha=1., update_interval=1, kernel_size=1, uniform_ratio=.1)
    sampler = BinSampler(torch.tensor([120, 80]), fps=50, config=cfg)
    sampler.update(torch.tensor([0, 1]), torch.tensor([60, 0]), torch.tensor([True, False]))
    sampler.step()
    assert sampler.failed[0, 1] == 1 and sampler.failed[0, 0] == 0
    assert sampler.probabilities()[0][0] > sampler.probabilities()[0][1]
    restored = BinSampler(torch.tensor([120, 80]), fps=50, config=cfg)
    restored.load_state_dict(sampler.state_dict())
    torch.manual_seed(9)
    expected = sampler.sample(100)
    torch.manual_seed(9)
    actual = restored.sample(100)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)
    assert torch.all(actual[1] < restored.lengths[actual[0]] - 1)
    bad = sampler.state_dict()
    bad["failed"][0, 0] = -1
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)


@pytest.mark.parametrize("stage", [1, 2])
def test_configs_emit_without_isaac(stage):
    pytest.importorskip("mjlab")
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_agent
    from pmt_tasks.backends.mjlab_pgmt.runner import serialize
    cfg = build_env_cfg(f"pgmt_stage{stage}", backend="mjlab")
    agent = build_agent(build_task_config(f"pgmt_stage{stage}"), stage=stage)
    assert tuple(cfg.pgmt_reward_groups) == (("upper", "lower", "auxiliary") if stage == 1 else
                                             ("upper", "lower", "terrain", "auxiliary"))
    assert len(cfg.rewards) == (21 if stage == 1 else 27)
    assert cfg.observations["proprio_history"].history_length == 10
    assert ("pgmt_terrain" in cfg.observations) == (stage == 2)
    assert agent["policy"]["class_name"] == "PGMTActorCritic"
    assert serialize(cfg)["pgmt_stage"] == stage


def test_wrapper_uses_terminal_rates_and_failure_wins_timeout():
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    terms = dict(upper_a=torch.tensor([1., 2.]), lower_b=torch.tensor([3., 4.]), auxiliary_c=torch.tensor([5., 6.]))
    cfg = SimpleNamespace(pgmt_reward_groups=("upper", "lower", "auxiliary"),
                          rewards={key: SimpleNamespace(weight=2.) for key in terms})
    scalar = sum(terms.values()) * .04
    def step(actions):
        return dict(proprio=torch.zeros(2, 96)), scalar, torch.tensor([True, False]), torch.ones(2, dtype=torch.bool), {}
    env = PGMTVectorEnv.__new__(PGMTVectorEnv)
    env.env = SimpleNamespace(step=step, step_dt=.02)
    env.cfg, env.num_envs = cfg, 2
    env.command = SimpleNamespace(_rewards=terms)
    _, reward, done, extras = env.step(torch.zeros(2, 29))
    torch.testing.assert_close(extras["reward_groups"].sum(-1), reward)
    assert extras["time_outs"].tolist() == [False, True]
    assert done.tolist() == [1, 1]


def test_native_v8_geometry_is_not_resampled():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("warp")
    root = Path(__file__).resolve().parents[1] / "assets/terrain/pgmt_stage2_v8"
    if not (root / "source.mjb").exists() or mujoco.__version__ != "3.8.0":
        pytest.skip("Requires exported v8 asset and MuJoCo 3.8.0")
    from pmt_tasks.backends.mjlab_pgmt.terrain import add_native_terrain
    source = mujoco.MjModel.from_binary_path(str(root / "source.mjb"))
    spec = mujoco.MjSpec()
    add_native_terrain(spec, asset=root)
    model = spec.compile()
    np.testing.assert_array_equal(model.hfield_data, source.hfield_data)
    np.testing.assert_array_equal(model.hfield_size, source.hfield_size)
    assert model.ngeom == source.ngeom
    np.testing.assert_array_equal(model.geom_type, source.geom_type)
    np.testing.assert_allclose(model.geom_size, source.geom_size, rtol=0, atol=1e-12)
    data, original = mujoco.MjData(model), mujoco.MjData(source)
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(source, original)
    np.testing.assert_allclose(data.geom_xpos, original.geom_xpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.geom_xmat, original.geom_xmat, atol=1e-12, rtol=0)


def test_v8_warp_queries_match_native_collision_heights():
    mujoco = pytest.importorskip("mujoco")
    wp = pytest.importorskip("warp")
    root = Path(__file__).resolve().parents[1] / "assets/terrain/pgmt_stage2_v8"
    if not (root / "source.mjb").exists() or mujoco.__version__ != "3.8.0":
        pytest.skip("Requires exported v8 asset and MuJoCo 3.8.0")
    from pmt_tasks.backends.mjlab_pgmt.terrain import Terrain
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    terrain = Terrain(root, 2, "cpu", torch.zeros(2, 3))
    xy = terrain.terrain_origins[..., :2].reshape(-1, 2) + torch.tensor([.37, -.21])
    actual = terrain.height(xy).numpy()
    model = mujoco.MjModel.from_binary_path(str(root / "source.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    expected = []
    for point in xy.numpy():
        start = np.r_[point, terrain.ray_top]
        distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]), None, True, -1,
                                 np.zeros(1, dtype=np.int32))
        assert distance >= 0
        expected.append(terrain.ray_top - distance)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)


@pytest.mark.skipif(os.environ.get("PMT_MJLAB_INTEGRATION") != "1", reason="opt-in real MJWarp simulation")
def test_real_rollout_reset_order_rewards_and_history(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mjlab")
    import warp as wp
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
    from mjlab.envs import ManagerBasedRlEnv
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    model = get_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    b = [model.body(name).id for name in BFS_BODIES]
    j = [model.joint(name).qposadr[0] for name in BFS_JOINTS]
    path = tmp_path / "standing.npz"
    clip(path, frames=4, joint_pos=np.tile(data.qpos[j], (4, 1)),
         body_pos_w=np.tile(data.xpos[b], (4, 1, 1)), body_quat_w=np.tile(data.xquat[b], (4, 1, 1)))
    cfg = build_env_cfg("pgmt_stage1", backend="mjlab")
    cfg.scene.num_envs, cfg.events = 2, {}
    cfg.commands["motion"].motion_files = [str(path)]
    cfg.commands["motion"].noise = False
    env = PGMTVectorEnv(ManagerBasedRlEnv(cfg, device="cpu"))
    try:
        c = env.command
        action = env.env.action_manager.get_term("joint_pos")
        assert action.target_names == BFS_JOINTS
        for _ in range(5):
            obs, reward, done, info = env.step(torch.zeros(2, 29))
            torch.testing.assert_close(info["reward_groups"].sum(-1), reward, rtol=1e-5, atol=1e-6)
            assert torch.isfinite(reward).all()
            torch.testing.assert_close(obs["proprio_history"][:, -1], obs["proprio"])
            for key in obs.keys():
                assert torch.isfinite(obs[key]).all()
            if done.any():
                # No extra reference advance is allowed after an auto-reset.
                torch.testing.assert_close(c.frame_ids[done.bool()], c.start_frames[done.bool()])
        # Repeated observation reads must not append history or draw noise.
        first = env.get_observations().clone()
        for key in first.keys():
            torch.testing.assert_close(first[key], env.get_observations()[key], rtol=0, atol=0)
    finally:
        env.close()
