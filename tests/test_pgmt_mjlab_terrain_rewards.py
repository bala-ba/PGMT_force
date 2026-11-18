"""Backend contact-force conventions, labeled references and four-group sums."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_pgmt_mjlab import clip
from pmt_tasks.backends.mjlab_pgmt.motion import MotionLibrary


def test_reference_labels_keep_left_right_and_invalid_boundaries(tmp_path):
    clip(tmp_path / "clip.npz", contacts=np.tile([0, 1], (7, 1)),
         contact_valid=np.ones((7, 2)), foot_names=np.array(["right_ankle_roll_link", "left_ankle_roll_link"]))
    lib = MotionLibrary(tmp_path, device="cpu", dt=.02, chunk_length=4, contact_labels=True)
    data = lib.contact_labels.gather(torch.tensor([0, 0, 1, 1]), torch.tensor([-1, 3, 2, 3]))
    assert data["contacts"].tolist() == [[1., 0.]] * 4
    assert data["contact_valid"].all(-1).tolist() == [False, True, True, False]
    clip(tmp_path / "missing.npz")
    with pytest.raises(ValueError, match="contact metadata"):
        MotionLibrary(tmp_path, device="cpu", dt=.02, contact_labels=True)


def test_normal_force_pair_order_self_contact_cloud_and_capacity():
    import warp as wp
    from pmt_tasks.backends.mjlab_pgmt.terrain_rewards import _collect, _invalidate
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    wp.init()
    def array(value, dtype):
        return wp.array(np.asarray(value), dtype=dtype, device="cpu")
    # Geometry 0: terrain, 1: left foot, 2: right foot, 3: non-terrain object.
    pairs = array([[0, 1], [1, 0], [1, 2], [3, 2], [0, 1]], wp.vec2i)
    frame = np.tile(np.eye(3), (5, 1, 1)).astype(np.float32)
    frame[0] = frame[2] = frame[4] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    # Non-terrain object is a vertical wall; its normal force enters T4/T6.
    frame[3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    frame[1] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    force = array([[30, 0, 100, 0, 0, 0], [20, 0, -50, 0, 0, 0],
                   [70, 0, 10, 0, 0, 0], [80, 0, 20, 0, 0, 0], [0, 0, 999, 0, 0, 0]], wp.spatial_vector)
    net = wp.zeros((2, 2), dtype=wp.vec3, device="cpu")
    ground = wp.zeros_like(net)
    points = wp.zeros((2, 2, 3), dtype=wp.vec3, device="cpu")
    counts = wp.zeros((2, 2), dtype=int, device="cpu")
    wp.launch(_collect, dim=5, inputs=[array([4], int), pairs, array([0]*5, int),
        array([[i, 0, 0] for i in range(5)], wp.vec3), array(frame, wp.mat33), force,
        array([-1, 0, 1, -1], int), array([1, 0, 0, 0], int), 3, net, ground, points, counts], device="cpu")
    np.testing.assert_array_equal(net.numpy()[0], [[0, 0, 140], [80, 0, 10]])
    np.testing.assert_array_equal(ground.numpy()[0], [[0, 0, 150], [0, 0, 0]])
    assert counts.numpy().tolist() == [[2, 0], [0, 0]]
    np.testing.assert_array_equal(points.numpy()[0, 0, :2, 0], [0, 1])
    from pmt_tasks.pgmt_stumble import evaluate_stumble
    from pmt_tasks.pgmt_contact_force import evaluate_contact_force
    force_w = wp.to_torch(net)
    valid = torch.ones(2, 2, dtype=torch.bool)
    assert evaluate_stumble(force_w, valid)["rate"].tolist() == [1., 0.]
    assert evaluate_contact_force(force_w, valid)["rate"].tolist() == [26100., 0.]
    wp.launch(_invalidate, dim=(2, 2), inputs=[array([4], int), array([0, 512], int), 5, 512, 3,
        net, ground, counts], device="cpu")
    assert np.isfinite(net.numpy()[0]).all() and np.isnan(net.numpy()[1]).all()
    assert counts.numpy()[1].tolist() == [4, 4]


def test_stage2_weights_equal_isaac_event_override_and_four_group_dt_once():
    from omegaconf import OmegaConf
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import ROOT, build_agent
    from pmt_tasks.backends.mjlab_pgmt.vecenv import PGMTVectorEnv
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    isaac = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/reward/pgmt_stage2_terrain.yaml"))
    from pmt_tasks.pgmt_touchdown import TOUCHDOWN_QUALITY_CONFIG, configure_touchdown_event
    override = SimpleNamespace(pgmt_terrain_reward_mode="geometry_v3",
        pgmt_touchdown_quality=dict(TOUCHDOWN_QUALITY_CONFIG),
        rewards=SimpleNamespace(terrain_touchdown_quality=SimpleNamespace(weight=isaac["weights"]["terrain_touchdown_quality"])))
    configure_touchdown_event(override)
    isaac["weights"]["terrain_touchdown_quality"] = override.rewards.terrain_touchdown_quality.weight
    assert cfg.pgmt_touchdown_quality == override.pgmt_touchdown_quality
    assert cfg.pgmt_contact_events["support_threshold_n"] == 1.
    assert cfg.pgmt_contact_events["touchdown_min_non_support_steps"] == 2
    assert cfg.pgmt_contact_events["contact_threshold_n"] == 1.  # T2/T3 unchanged.
    assert {key: term.weight for key, term in cfg.rewards.items()} == isaac["weights"]
    assert list(cfg.pgmt_reward_groups) == isaac["groups"]
    agent = build_agent(build_task_config("pgmt_stage2"), stage=2)
    assert agent["policy"]["value_groups"] == isaac["groups"]
    assert cfg.metrics["terrain_support_force_z"].per_substep
    rates = {name: torch.tensor([i*.25, i*.5]) for i, name in enumerate(cfg.rewards, 1)}
    scalar = sum(rates[name]*term.weight*.02 for name, term in cfg.rewards.items())
    vec = PGMTVectorEnv.__new__(PGMTVectorEnv)
    vec.cfg, vec.num_envs = cfg, 2
    vec.command = SimpleNamespace(_rewards=rates)
    vec.env = SimpleNamespace(step_dt=.02, step=lambda action: (
        {"proprio": torch.zeros(2, 96)}, scalar, torch.tensor([True, False]), torch.tensor([True, True]), {}))
    _, reward, _, info = vec.step(torch.zeros(2, 29))
    assert info["reward_groups"].shape == (2, 4)
    torch.testing.assert_close(info["reward_groups"].sum(-1), reward)
    assert info["time_outs"].tolist() == [False, True]


def test_t1_matches_reference_run_geometry_with_requested_one_newton_threshold():
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.backends.mjlab_pgmt.config import ROOT
    from pmt_tasks.pgmt_historical_config import read_archive
    from pmt_tasks.pgmt_touchdown import configure_touchdown_event
    path = ROOT / "logs/pgmt_stage2_center2x2/train_03/params/env.yaml"
    if not path.is_file():
        pytest.skip("Requires the user's archived reference run")
    archive = read_archive(path)
    reference = SimpleNamespace(pgmt_terrain_reward_mode="legacy",
        pgmt_touchdown_quality=archive["pgmt_touchdown_quality"],
        rewards=SimpleNamespace(terrain_touchdown_quality=SimpleNamespace(
            weight=archive["rewards"]["terrain_touchdown_quality"]["weight"])))
    configure_touchdown_event(reference)
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    assert cfg.pgmt_touchdown_quality == reference.pgmt_touchdown_quality
    assert cfg.rewards["terrain_touchdown_quality"].weight == reference.rewards.terrain_touchdown_quality.weight
    assert cfg.pgmt_contact_events["touchdown_min_non_support_steps"] == archive["pgmt_contact_events"]["touchdown_min_non_support_steps"]
    assert cfg.pgmt_contact_events["support_threshold_n"] == cfg.pgmt_contact_events["contact_threshold_n"] == 1.


@pytest.mark.parametrize("reference_valid", [False, True])
def test_backend_t1_landings_only_independent_of_t5_and_labels(monkeypatch, reference_valid):
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.backends.mjlab_pgmt import terrain_rewards as backend
    from pmt_tasks.backends.mjlab_pgmt.motion import BFS_BODIES
    from pmt_tasks.pgmt_contact_reference import ContactLabelStore

    class Reader:
        def __init__(self, env, slots):
            self.net_tensor = torch.zeros(1, 2, 3)
            self.ground_tensor = torch.zeros_like(self.net_tensor)

        def read(self):
            pass

        def cloud(self):
            raise AssertionError("Event T1 must not evaluate continuous contact-plane risk")

    monkeypatch.setattr(backend, "TerrainContactReader", Reader)
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02, common_step_counter=0, cfg=cfg)
    # Unknown labels or labeled flight cannot veto physical touchdown rewards.
    labels = ContactLabelStore([SimpleNamespace(source_file="test", num_frames=32, metadata=dict(
        contacts=np.zeros((32, 2)), contact_valid=np.full((32, 2), reference_valid),
        foot_names=np.array(["left_ankle_roll_link", "right_ankle_roll_link"])))], device="cpu")
    queries = []

    def query(xy):
        queries.append(xy.clone())
        return dict(height=xy[:, 0]*.2, valid=torch.ones(len(xy), dtype=torch.bool))

    command = SimpleNamespace(_env=env, anchor=BFS_BODIES.index("torso_link"),
        motion_ids=torch.zeros(1, dtype=torch.long), frame_ids=torch.zeros(1, dtype=torch.long),
        data_store=SimpleNamespace(contact_labels=labels), terrain=SimpleNamespace(query=query))
    runtime = backend.TerrainContactRewards(command)
    state = dict(body_pos=torch.zeros(1, 30, 3), body_quat=torch.zeros(1, 30, 4),
                 body_lin_vel=torch.zeros(1, 30, 3))
    state["body_quat"][..., 0] = 1.
    # T1/T2 use strict >1 N. T5's 1/0.5 N hysteresis retains support at 0.75 N.
    forces = [(100,100), (100,100), (1.,100), (.75,100), (1.01,100), (100,100),
              (100,1.), (100,1.01), (1.,1.), (.75,.75), (1.01,1.01)]
    quality = np.exp(-np.var(np.repeat(np.array([-.1, 0., .1])+.039, 3)*.2)/.03**2)
    for step, force in enumerate(forces, 1):
        runtime.reader.ground_tensor[0, :, 2] = torch.tensor(force)
        runtime.reader.net_tensor.copy_(runtime.reader.ground_tensor)
        for _ in range(cfg.decimation):
            runtime.capture_substep()
        env.common_step_counter = step
        command.frame_ids.fill_(step)
        rates = runtime.compute(state)
        expected = quality * (1 if step == 5 else 2 if step == 11 else 0)
        assert rates["terrain_touchdown_quality"].item() == pytest.approx(expected, abs=1e-6)
        assert (rates["terrain_touchdown_quality"]*1.5*env.step_dt).item() == pytest.approx(expected*.03, abs=1e-6)
        assert rates["terrain_contact_switching"].item() == 0
        expected_contact = torch.tensor([force]) > 1.
        torch.testing.assert_close(runtime.snapshot["contact"]["contact"], expected_contact)
        expected_match = float((~expected_contact).float().mean()) if reference_valid else 0.
        assert rates["terrain_reference_contact_match"].item() == pytest.approx(expected_match)
        if step == 11:
            frozen = runtime.snapshot
    assert [len(points) for points in queries] == [9, 18]
    runtime.reset(torch.tensor([0]))
    assert runtime.snapshot is frozen
    assert frozen["touchdown_quality"]["rate"].item() == pytest.approx(2*quality, abs=1e-6)
    env.common_step_counter += 1
    for _ in range(cfg.decimation):
        runtime.capture_substep()
    rates = runtime.compute(state)
    assert rates["terrain_touchdown_quality"].eq(0).all()
    assert not runtime.snapshot["touchdown_quality"]["event"].any()
    assert len(queries) == 2  # Reset-on-ground earns no delayed payout or query.
