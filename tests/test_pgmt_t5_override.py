"""Independent T5 selection on the historical reward and reset lifecycle."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_contact_geometry import (
    CONTACT_GEOMETRY_CONFIG, TerrainSupportState, configure_contact_switching,
)
from pmt_tasks.pgmt_contact_state import CONTACT_STATE_CONFIG, ContactEventState
from pmt_tasks.pgmt_runtime import PGMTState


def test_override_changes_only_t5_and_retains_legacy_contact_thresholds():
    cfg = SimpleNamespace(pgmt_terrain_reward_mode='legacy', pgmt_contact_switching=True,
        pgmt_contact_events=deepcopy(CONTACT_STATE_CONFIG),
        pgmt_geometry_contacts=deepcopy(CONTACT_GEOMETRY_CONFIG),
        rewards=SimpleNamespace(terrain_touchdown_quality=SimpleNamespace(weight=1.5),
                                terrain_contact_switching=SimpleNamespace(weight=-30.)))
    expected = deepcopy(cfg)
    report = configure_contact_switching(cfg, 'geometry_v3')
    expected.pgmt_contact_switching_mode = 'geometry_v3'
    expected.rewards.terrain_contact_switching.weight = -.05
    assert cfg == expected
    assert cfg.pgmt_contact_events['contact_threshold_n'] == 10.
    assert cfg.pgmt_contact_events['support_threshold_n'] == 10.
    assert report['previous_weight'] == -30. and report['weight'] == -.05
    configure_contact_switching(cfg, 'legacy')
    assert cfg.rewards.terrain_contact_switching.weight == -30.
    with pytest.raises(ValueError, match='Unknown'):
        configure_contact_switching(cfg, 'bad')


def make_state(monkeypatch):
    state = PGMTState.__new__(PGMTState)
    net = torch.zeros(2, 2, 3)
    state.env = SimpleNamespace(common_step_counter=0, step_dt=.02, cfg=SimpleNamespace(decimation=4),
        scene={'contact_forces': SimpleNamespace(data=SimpleNamespace(net_forces_w=net))})
    state.geometry_rewards, state.support_switching = False, True
    state.contact_tracker = ContactEventState(2, device='cpu', dt=.02)
    state.support_tracker = TerrainSupportState(2, device='cpu', dt=.02)
    state._support_force_sum = torch.zeros(2, 2)
    state._support_force_count = torch.zeros(2, dtype=torch.long)
    state.contact_switching_enabled = state.contact_match_enabled = state.slip_enabled = True
    state._contact_switching_live = {}
    state._contact_switching_reward = state._reward_contacts = state._contact_reward_step = None
    state._reward_step, state._obs_version = None, 0
    state.contact_order = state.foot_body_indices = torch.tensor([0, 1])
    state.anchor, state.config, state.groups = 0, {}, {}
    state.pending_reset = torch.ones(2, dtype=torch.bool)
    state.previous_target = torch.zeros(2, 29)
    state.previous_velocity, state.previous_ref_velocity = torch.zeros(2, 2, 3), torch.zeros(2, 2, 3)
    state.robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 29),
                                                       soft_joint_pos_limits=torch.zeros(2, 29, 2)))
    labels = dict(contacts=torch.zeros(2, 2), contact_valid=torch.ones(2, 2, dtype=torch.bool),
                  motion_ids=torch.tensor([0, 1]), frame_ids=torch.tensor([8, 12]))
    state.command = SimpleNamespace(motion_ids=labels['motion_ids'], frame_ids=labels['frame_ids'])
    state.contact_label_store = None
    monkeypatch.setattr('pmt_tasks.pgmt_runtime.reference_switch_candidates', lambda *a: {})
    state._contact_context = {f'reference_{key}': value.clone() for key, value in labels.items()}
    state._invalidate_contact_context(slice(None))
    physical = dict(body_pos=torch.zeros(2, 2, 3), body_quat=torch.zeros(2, 2, 4),
                    body_lin_vel=torch.ones(2, 2, 3), body_ang_vel=torch.zeros(2, 2, 3))
    state.reference_contacts = lambda: labels
    state.robot_foot_terrain_forces = lambda: net
    state.robot_state = state.reference = lambda: physical
    monkeypatch.setattr('pmt_tasks.pgmt_runtime.reward_terms', lambda *a, **kw: {'upper_fixture': torch.zeros(2)})
    return state, net


def test_substep_mean_t5_is_independent_of_legacy_t2_t3_and_reset(monkeypatch):
    state, net = make_state(monkeypatch)
    # Last substep Fz=0 throughout, but the four-substep mean is 2 N in stance.
    # Old shared 10 N contact remains false; new T5 sees both support transitions.
    for step, total in enumerate((0., 8., 8., 8., 8., 0.)):
        state.env.common_step_counter = step
        state._support_force_sum.fill_(total)
        state._support_force_count.fill_(4)
        result = state.rewards()
        assert not state.contact_state(for_reward=True)['contact'].any()
        assert result['terrain_reference_contact_match'].eq(1.).all()
        assert result['terrain_slip'].eq(0.).all()
        torch.testing.assert_close(state._contact_switching_reward['mean_force_z'], torch.full((2, 2), total/4))
        assert state._contact_switching_reward['physics_samples'].eq(4).all()
    torch.testing.assert_close(result['terrain_contact_switching'], torch.full((2,), 20.))
    torch.testing.assert_close(result['terrain_contact_switching'] * -.05 * .02, torch.full((2,), -.02))
    terminal = state.contact_switching_state(for_reward=True)
    state.reset(torch.tensor([0]))
    assert state.contact_switching_state()['rate'][0] == 0
    assert not state.support_tracker.valid[0].any() and state.support_tracker.valid[1].all()
    for _ in range(3):
        torch.testing.assert_close(state.rewards()['terrain_contact_switching'], terminal['rate'])
        torch.testing.assert_close(state.contact_switching_state(for_reward=True)['mean_force_z'], terminal['mean_force_z'])
    state.env.common_step_counter += 1
    state._support_force_sum.fill_(8.)
    state._support_force_count.fill_(4)
    result = state.rewards()['terrain_contact_switching']
    torch.testing.assert_close(result, torch.tensor([0., 80.]))


def test_incomplete_physics_window_invalidates_only_affected_environment(monkeypatch):
    state, _ = make_state(monkeypatch)
    state._support_force_sum.fill_(8.)
    state._support_force_count[:] = torch.tensor([3, 4])
    state.rewards()
    snapshot = state.contact_switching_state(for_reward=True)
    assert torch.isnan(snapshot['mean_force_z'][0]).all()
    assert not snapshot['valid'][0].any() and snapshot['valid'][1].all()
    assert snapshot['rate'].eq(0).all()
