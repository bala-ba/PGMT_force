"""One-newton T1/T2 detection without changing T3 or the new T5 hysteresis."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_contact_geometry import CONTACT_GEOMETRY_CONFIG
from pmt_tasks.pgmt_contact_match import configure_t1_t2_contact_threshold
from pmt_tasks.pgmt_contact_state import CONTACT_STATE_CONFIG, ContactEventState
from pmt_tasks.pgmt_runtime import PGMTState
from pmt_tasks.pgmt_touchdown import TOUCHDOWN_QUALITY_CONFIG


def config():
    return SimpleNamespace(pgmt_contact_events=CONTACT_STATE_CONFIG.copy(),
        pgmt_reference_contact_match=True,
        pgmt_touchdown_quality=dict(TOUCHDOWN_QUALITY_CONFIG, settlement='touchdown_event_v1'),
        pgmt_contact_switching_mode='geometry_v3', pgmt_geometry_contacts=CONTACT_GEOMETRY_CONFIG.copy())


def test_threshold_override_changes_only_t1_support_and_t2_private_threshold():
    cfg = config()
    expected = deepcopy(cfg)
    expected.pgmt_contact_events['support_threshold_n'] = 1.
    expected.pgmt_reference_contact_threshold_n = 1.
    report = configure_t1_t2_contact_threshold(cfg, 1.)
    assert cfg == expected
    assert report['previous_support_threshold_n'] == report['previous_reference_contact_threshold_n'] == 10.
    assert report['shared_contact_threshold_n'] == 10.


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf'), True])
def test_invalid_threshold_is_rejected_without_mutating_config(bad):
    cfg = config()
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match='finite and positive'):
        configure_t1_t2_contact_threshold(cfg, bad)
    assert cfg == before


def test_geometry_t1_cannot_silently_ignore_support_threshold_override():
    cfg = config()
    cfg.pgmt_terrain_reward_mode = 'geometry_v3'
    cfg.pgmt_touchdown_quality['settlement'] = 'support_pattern_mean_v2'
    with pytest.raises(ValueError, match='variance T1'):
        configure_t1_t2_contact_threshold(cfg, 1.)


def test_runtime_one_newton_boundaries_force_directions_and_unchanged_slip():
    cfg = config()
    configure_t1_t2_contact_threshold(cfg, 1.)
    values = torch.tensor([[0., 0., .9], [0., 0., 1.], [0., 0., 1.0001],
                           [2., 0., 0.], [0., 0., -2.], [0., 0., 11.], [float('nan'), 0., 0.]])
    n = len(values)
    force = torch.zeros(n, 2, 3)
    query_calls = []
    def query(xy):
        query_calls.append(xy.clone())
        return dict(height=torch.zeros(len(xy)), valid=torch.ones(len(xy), dtype=torch.bool))
    state = PGMTState.__new__(PGMTState)
    state.env = SimpleNamespace(common_step_counter=0, scene=SimpleNamespace(terrain=SimpleNamespace(query=query)))
    state.contact_tracker = ContactEventState(n, device='cpu', dt=.02, config=cfg.pgmt_contact_events)
    state.touchdown_config = cfg.pgmt_touchdown_quality
    state.reference_contact_threshold_n = cfg.pgmt_reference_contact_threshold_n
    state.contact_match_enabled = state.slip_enabled = True
    state.anchor, state.foot_body_indices = 0, torch.tensor([0, 1])
    labels = dict(contacts=torch.ones(n, 2), contact_valid=torch.ones(n, 2, dtype=torch.bool),
                  motion_ids=torch.arange(n), frame_ids=torch.zeros(n, dtype=torch.long))
    state.reference_contacts = lambda: labels
    state.robot_foot_terrain_forces = lambda: force
    poses = dict(body_pos=torch.zeros(n, 2, 3), body_quat=torch.zeros(n, 2, 4),
                 body_lin_vel=torch.ones(n, 2, 3), body_ang_vel=torch.zeros(n, 2, 3))
    poses['body_quat'][..., 0] = 1.
    for step in range(3):
        state.env.common_step_counter = step
        if step == 2:
            force[:] = values[:, None]
        state._update_contact_state(poses, force)
    t1 = state.touchdown_state(for_reward=True)
    t2 = state.reference_contact_match_state(for_reward=True)
    assert t1['rate'].tolist() == [0., 0., 2., 0., 0., 2., 0.]
    assert t2['rate'].tolist() == [0., 0., 1., 1., 1., 1., 0.]
    assert not t2['actual_valid'][-1].any()
    assert state._slip_reward['rate'].tolist() == [0., 0., 0., 0., 0., 4., 0.]
    assert len(query_calls) == 1 and len(query_calls[0]) == 36
    state.env.common_step_counter += 1
    state._update_contact_state(poses, force)
    assert state.touchdown_state(for_reward=True)['rate'].eq(0).all()
    assert len(query_calls) == 1  # Holding a 1 N support never repays touchdown.
