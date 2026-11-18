"""MJWarp relaxation wiring: real tile metadata, reward invariants and resets."""
import json
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.backends.mjlab_pgmt.motion import BFS_BODIES
from pmt_tasks.pgmt_math import TRACKING_CONFIG
from pmt_tasks.pgmt_terrain_layout import training_manifest
from pmt_tasks.pgmt_terrain_query import TerrainTileMap
from pmt_tasks.pgmt_tracking_relaxation import RELAXATION_TERMS
from test_pgmt_tracking_relaxation import sample


def runtime(n, *, enabled=True):
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.backends.mjlab_pgmt.config import ROOT
    from pmt_tasks.backends.mjlab_pgmt.mdp import Motion
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    cfg.pgmt_tracking_relaxation["enabled"] = enabled
    manifest = training_manifest(json.loads((ROOT / "assets/terrain/pgmt_stage2_v6_v8difficulty_mjwarp/manifest.json").read_text()))
    bounds = torch.empty(10, 5, 2, 2, dtype=torch.float64)
    for tile in manifest["tiles"]:
        bounds[tile["level"], tile["family_index"]] = torch.tensor(tile["bounds"])
    tiles = TerrainTileMap(manifest, bounds)
    state, ref, groups, target, limits, errors, strict = sample(n, "cpu")
    c = Motion.__new__(Motion)
    c._env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.02, common_step_counter=1, cfg=cfg,
        scene={"pgmt_contacts": SimpleNamespace(data=SimpleNamespace(force=torch.ones_like(state["body_pos"])*30))})
    c.cfg = SimpleNamespace(tracking=dict(TRACKING_CONFIG), stage=2, fixed_start_frame=0, evaluation=False)
    c.terrain = SimpleNamespace(tile_map=tiles, terrain_levels=torch.zeros(n, dtype=torch.long),
                                terrain_types=torch.zeros(n, dtype=torch.long))
    c.groups, c.anchor = groups, BFS_BODIES.index("torso_link")
    c.joints = torch.arange(29)
    c.robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=limits))
    c.robot_state, c.reference, c.qtarget = lambda: state, lambda: ref, lambda: target
    c.previous_target = target.clone()
    c.previous_velocity, c.previous_ref_velocity = state["body_lin_vel"].clone(), ref["body_lin_vel"].clone()
    c.pending_reset, c.started = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    c.motion_ids = torch.zeros(n, dtype=torch.long)
    c.frame_ids = torch.full((n,), 7, dtype=torch.long)
    c.start_frames, c.reset_step = c.motion_ids.clone(), torch.full_like(c.motion_ids, -1)
    c._reward_step, c._rewards = -1, None
    c.identity, c.timers, c.curriculum, c.terrain_rewards = {}, {}, None, None
    c.sampler = SimpleNamespace(weights=torch.ones(1))
    c.compatible = torch.ones(5, 1, dtype=torch.bool)
    c.data_store = SimpleNamespace(motion_lengths=torch.tensor([100]))
    c._settle = lambda ids: None
    c._configure_tracking_relaxation()

    def move(xy, ids=slice(None)):
        delta = xy - state["body_pos"][ids, c.anchor, :2]
        state["body_pos"][ids, :, :2] += delta[:, None]

    return c, state, ref, strict, errors, manifest, move


def test_config_enables_mjwarp_stage2_and_preserves_stage1_schema():
    from pmt_tasks.builder import build_env_cfg, build_task_config
    from pmt_tasks.backends.mjlab_pgmt.config import build_env
    from pmt_tasks.backends.mjlab_pgmt.runner import serialize
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    assert cfg.pgmt_tracking_relaxation["enabled"]
    assert len(cfg.rewards) == 27 and len(cfg.pgmt_reward_groups) == 4
    assert "pgmt_tracking_relaxation" not in serialize(build_env_cfg("pgmt_stage1", backend="mjlab"))
    task = build_task_config("pgmt_stage2")
    task.mjlab_tracking_relaxation.enabled = False
    assert not build_env(task, stage=2).pgmt_tracking_relaxation["enabled"]
    task.tracking_relaxation.position_max = -1
    with pytest.raises(ValueError):
        build_env(task, stage=2)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_v6_all_families_levels_change_only_three_lower_rates(enabled):
    c, state, ref, strict, errors, manifest, move = runtime(50, enabled=enabled)
    move(torch.tensor([tile["center"][:2] for tile in manifest["tiles"]]))
    before = {key: value.clone() for key, value in state.items()}
    reference = {key: value.clone() for key, value in ref.items()}
    expected_strict = strict()
    actual = c.rewards()
    assert len(actual) == 21
    scale = errors.new_tensor([tile["level"]/9 if enabled and tile["family_group_index"] in (1, 2, 3) else 0
                               for tile in manifest["tiles"]])
    for name, cap in zip(RELAXATION_TERMS, (.2, .7, .5)):
        margin = scale[:, None].expand(-1, 12)*cap
        sigma = TRACKING_CONFIG[name.removeprefix("lower_")+"_sigma"]
        expected = torch.exp(-(errors[None]-margin).clamp_min(0).square().mean(-1)/sigma**2)
        torch.testing.assert_close(actual[name], expected)
        torch.testing.assert_close(actual[name][scale == 0], expected_strict[name][scale == 0], rtol=0, atol=0)
        if enabled:
            torch.testing.assert_close(c.tracking_relaxation_state()[name+"_margin"], margin)
    for name in actual.keys()-RELAXATION_TERMS.keys():
        torch.testing.assert_close(actual[name], expected_strict[name], rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(state[key], before[key], rtol=0, atol=0)
        torch.testing.assert_close(ref[key], reference[key], rtol=0, atol=0)
    assert c.identity["tracking_relaxation"]["difficulty_source"] == "current_tile_level"


def test_cross_tile_cache_partial_reset_invalid_map_and_flat_evaluation():
    c, _, _, _, _, manifest, move = runtime(2)
    def center(level, group):
        return torch.tensor([next(t["center"][:2] for t in manifest["tiles"]
                                  if t["level"] == level and t["family_group_index"] == group)], dtype=torch.float64)
    calls = []
    locate = c.terrain.tile_map.locate
    def counted(xy):
        calls.append(xy.clone())
        return locate(xy)
    c.terrain.tile_map.locate = counted
    move(center(9, 2).expand(2, -1))  # Current stairs L9, birth flat L0.
    with pytest.raises(RuntimeError, match="No settled"):
        c.tracking_relaxation_state()
    first = c.rewards()
    frozen = c.tracking_relaxation_state()
    assert frozen["tile_level"].tolist() == [9, 9] and frozen["birth_level"].tolist() == [0, 0]
    frozen["lower_body_position_margin"].zero_()
    assert c.tracking_relaxation_state()["lower_body_position_margin"].eq(.2).all()
    c.place = lambda ids: move(center(0, 0), ids)
    c._resample_command(torch.tensor([0]))
    assert c.rewards() is first and len(calls) == 1
    assert c.frame_ids.tolist() == [0, 7]
    assert c.tracking_relaxation_state()["reference_frame_ids"].tolist() == [7, 7]
    assert c.tracking_relaxation_state()["tile_level"].tolist() == [9, 9]
    move(center(3, 1), torch.tensor([1]))
    c._env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="this control step"):
        c.tracking_relaxation_state()
    c.rewards()
    current = c.tracking_relaxation_state()
    torch.testing.assert_close(current["lower_body_position_margin"], torch.tensor([[0.]*12, [.2/3]*12], dtype=torch.float64))
    assert current["tile_level"].tolist() == [0, 3]
    move(torch.full((2, 2), -1e5))
    c._env.common_step_counter += 1
    c.rewards()
    invalid = c.tracking_relaxation_state()
    assert not invalid["tile_valid"].any() and invalid["level_scale"].eq(0).all()
    assert invalid["lower_body_position_margin"].eq(0).all()
    c.terrain = None
    c._configure_tracking_relaxation()
    assert c.tracking_relaxation is None  # Separate flat EMD worlds remain flat.
