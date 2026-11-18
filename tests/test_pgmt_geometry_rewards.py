"""Analytical geometry and event accounting, independent of Isaac initialization."""
import pytest
import torch
from pmt_tasks.pgmt_support_risk import SupportRiskGeometry, SUPPORT_RISK_CONFIG, _fit
from pmt_tasks.pgmt_contact_geometry import TerrainSupportState, pack_contact_points


def query_plane(xy):
    return dict(height=torch.zeros_like(xy[:, 0]), valid=torch.ones_like(xy[:, 0], dtype=torch.bool))


def evaluate(query=query_plane, *, contacts=None, position=None, quaternion=None, overflow=False):
    geometry = SupportRiskGeometry(device='cpu')
    pos = torch.tensor([[[0., 0., .035]]*2]) if position is None else position
    quat = torch.tensor([[[1., 0., 0., 0.]]*2]) if quaternion is None else quaternion
    cloud = torch.zeros(1, 2, 8, 3)
    valid = torch.zeros(1, 2, 8, dtype=torch.bool)
    if contacts is not None:
        cloud[:, :, :len(contacts)] = torch.tensor(contacts)
        valid[:, :, :len(contacts)] = True
    return geometry.evaluate(torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool),
                             pos, quat, cloud, valid, torch.full((1, 2), overflow), query)


def test_flat_fallback_and_true_contact_plane():
    for contacts, source in ((None, 2), ([[-.03, -.02, 0.], [.1, -.02, 0.], [.02, .02, 0.]], 1)):
        r = evaluate(contacts=contacts)
        assert r['plane_source'].eq(source).all()
        assert r['valid'].all()
        torch.testing.assert_close(r['rate'], torch.zeros(1))
        torch.testing.assert_close(r['coverage'], torch.ones(1, 2))
        assert r['query_xy_w'].shape == (1, 2, 25, 2)


def test_deduplication_collinear_and_overflow_fallback():
    for contacts, count in (([[0., 0., 0.]]*4, 1),
                            ([[-.03, 0., 0.], [0., 0., 0.], [.1, 0., 0.]], 3)):
        r = evaluate(contacts=contacts)
        assert r['contact_count'].eq(count).all()
        assert r['plane_source'].eq(2).all()
    r = evaluate(contacts=[[-.03, -.02, 0.], [.1, -.02, 0.], [.02, .02, 0.]], overflow=True)
    assert r['plane_source'].eq(2).all()


def test_continuous_slope_is_safe():
    def slope(xy):
        return dict(height=.4*xy[:, 0], valid=torch.ones(len(xy), dtype=torch.bool))
    r = evaluate(slope)
    assert r['plane_source'].eq(2).all()
    torch.testing.assert_close(r['rate'], torch.zeros(1), atol=1e-6, rtol=0)


def test_half_foot_overhang_uses_all_25_points():
    def edge(xy):
        return dict(height=torch.where(xy[:, 0] < .04, 0., -.2), valid=torch.ones(len(xy), dtype=torch.bool))
    r = evaluate(edge, contacts=[[-.03, -.02, 0.], [.02, -.02, 0.], [.02, .02, 0.]])
    assert r['plane_source'].eq(1).all()
    assert (r['rms'] > .1).all() and (r['coverage'] < .7).all()
    assert r['edge_cost'].eq(1).all() and (r['rate'] > .5).all()


def test_margin_detects_edge_outside_foot_without_fake_window_edge():
    def edge(xy):
        return dict(height=torch.where(xy[:, 0] < .148, 0., -.2), valid=torch.ones(len(xy), dtype=torch.bool))
    r = evaluate(edge)
    torch.testing.assert_close(r['coverage'], torch.ones(1, 2))
    assert (r['edge_cost'] > 0).all()
    assert evaluate()['edge_cost'].eq(0).all()


def test_foot_sampling_includes_imported_capsule_end_caps():
    from pmt_tasks.pgmt_support_risk import FOOT_HULL, foot_sampling, FOOT_OUTLINE_ERROR_M
    hull = torch.tensor(FOOT_HULL)
    torch.testing.assert_close(hull.amin(0), torch.tensor([-.064, -.036]))
    torch.testing.assert_close(hull.amax(0), torch.tensor([.142, .036]))
    samples, weights = foot_sampling(device='cpu', dtype=torch.float32)
    assert samples[:, 0].min() < -.0639 and samples[:, 0].max() > .1419
    assert 0 < FOOT_OUTLINE_ERROR_M < .0002
    torch.testing.assert_close(weights.sum(), torch.tensor(1.))


def test_cross_tread_cloud_does_not_create_fictitious_ramp():
    def step(xy):
        return dict(height=torch.where(xy[:, 0] < .03, 0., .06), valid=torch.ones(len(xy), dtype=torch.bool))
    r = evaluate(step, contacts=[[-.03, -.02, 0.], [-.03, .02, 0.], [.12, 0., .06]])
    assert r['plane_source'].ne(1).all()
    assert (r['rate'] > 0).all()


def test_missing_terrain_is_risky_and_missing_anchor_is_invalid():
    def unknown(xy):
        return dict(height=torch.full((len(xy),), float('nan')), valid=torch.zeros(len(xy), dtype=torch.bool))
    r = evaluate(unknown)
    assert r['rate'].eq(1).all() and not r['valid'].any()
    r = evaluate(position=torch.tensor([[[0., 0., 1.]]*2]))
    assert r['rate'].eq(1).all() and r['plane_source'].eq(0).all()


@pytest.mark.parametrize('radians', [.2, 1.1])
def test_full_quaternion_projection_and_normal_heel_roll(radians):
    angle = torch.tensor(radians)
    q = torch.tensor([[[torch.cos(angle/2), 0., torch.sin(angle/2), 0.]]*2])
    r = evaluate(quaternion=q, contacts=[[-.03, -.02, 0.], [-.03, .02, 0.]])
    assert r['valid'].all() and r['rate'].eq(0).all()
    assert not torch.allclose(r['query_xy_w'], evaluate()['query_xy_w'])


def test_rms_preserves_nonzero_mean_height_error():
    def shifted(xy):
        return dict(height=torch.full((len(xy),), -.006), valid=torch.ones(len(xy), dtype=torch.bool))
    r = evaluate(shifted, contacts=[[-.03, -.02, 0.], [.1, -.02, 0.], [.02, .02, 0.]])
    assert r['plane_source'].eq(1).all()
    torch.testing.assert_close(r['rms'], torch.full((1, 2), .006))
    assert (r['rms_cost'] > 0).all()


def test_invalid_foot_sample_not_dropped_or_renormalized():
    target = SupportRiskGeometry(device='cpu').local[12, :2]
    def missing(xy):
        return dict(height=torch.zeros(len(xy)), valid=(xy-target).norm(dim=-1) > 1e-7)
    r = evaluate(missing)
    assert r['per_foot_cost'].eq(1).all() and not r['valid'].any()


def test_feet_are_independent_and_safe_landing_never_pays_positive_bonus():
    def edge(xy):
        return dict(height=torch.where(xy[:, 0] < .04, 0., -.2), valid=torch.ones(len(xy), dtype=torch.bool))
    pos = torch.tensor([[[0., 0., .035], [-1., 0., .035]]])
    r = evaluate(edge, position=pos)
    assert r['per_foot_cost'][0, 0] > 0 and r['per_foot_cost'][0, 1] == 0
    torch.testing.assert_close(r['rate'], r['per_foot_cost'].sum(-1)/2)
    geometry = SupportRiskGeometry(device='cpu')
    for support in (torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)):
        r = geometry.evaluate(support, torch.ones_like(support),
            torch.tensor([[[0., 0., .035]]*2]), torch.tensor([[[1., 0., 0., 0.]]*2]),
            torch.zeros(1, 2, 4, 3), torch.zeros(1, 2, 4, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool), query_plane)
        assert r['rate'].eq(0).all()


def test_support_hysteresis_one_newton_and_unknown_samples():
    tracker = TerrainSupportState(1, device='cpu', dt=.02)
    values = [.9, 1., .9, .6, .5, float('nan'), 1.]
    actual = []
    for step, force in enumerate(values):
        r = tracker.update(step, torch.full((1, 2), force))
        actual.append(bool(r['support'][0, 0]))
        if step in (0, 5, 6):
            assert not r['event'].any()
    assert actual == [False, True, True, True, False, False, True]


@pytest.mark.parametrize('steps,expected', [(4, .2), (5, 0.), (6, 0.)])
def test_switch_duration_has_no_off_by_one(steps, expected):
    tracker = TerrainSupportState(1, device='cpu', dt=.02)
    tracker.update(0, torch.zeros(1, 2))
    tracker.update(1, torch.full((1, 2), 2.))
    for k in range(2, 1+steps):
        tracker.update(k, torch.full((1, 2), 2.))
    r = tracker.update(1+steps, torch.zeros(1, 2))
    torch.testing.assert_close(r['soft_cost'], torch.full((1, 2), expected))
    torch.testing.assert_close(r['duration_s'], torch.full((1, 2), steps*.02))


def test_event_cost_dt_invariance_and_partial_reset():
    penalties = []
    for dt in (.01, .02):
        tracker = TerrainSupportState(2, device='cpu', dt=dt)
        tracker.update(0, torch.zeros(2, 2))
        tracker.update(1, torch.full((2, 2), 2.))
        steps = round(.08/dt)
        for k in range(2, steps+1):
            tracker.update(k, torch.full((2, 2), 2.))
        tracker.reset(torch.tensor([1]))
        r = tracker.update(steps+1, torch.zeros(2, 2))
        assert not r['event'][1].any() and r['rate'][1] == 0
        penalties.append(r['rate'][0]*dt*-.05)
    torch.testing.assert_close(*penalties)
    torch.testing.assert_close(penalties[0], torch.tensor(-.02))


def test_reference_transition_exempted_only_once_and_same_direction():
    tracker = TerrainSupportState(1, device='cpu', dt=.02)
    ref = dict(reference_ids=torch.full((1, 2, 1), 12), reference_states=torch.ones(1, 2, 1, dtype=torch.bool),
               reference_valid=torch.ones(1, 2, 1, dtype=torch.bool), reference_distance=torch.zeros(1, 1, 1))
    for k, force in enumerate([0., 2., 0., 2.]):
        r = tracker.update(k, torch.full((1, 2), force), **ref)
        if k == 1:
            assert r['reference_exempt'].all()
        if k in (2, 3):
            assert not r['reference_exempt'].any() and (r['rate'] > 0).all()


def test_physx_packing_keeps_points_across_filter_columns():
    points = torch.arange(30.).reshape(10, 3)
    counts = torch.tensor([[2, 1, 0], [0, 0, 2]])
    starts = torch.tensor([[1, 5, 0], [0, 0, 8]])
    cloud, mask, overflow = pack_contact_points((torch.ones(10, 1), points, None, None, counts, starts), 2, 4)
    torch.testing.assert_close(cloud[0, :3], points[[1, 2, 5]])
    torch.testing.assert_close(cloud[1, :2], points[[8, 9]])
    assert mask.sum(-1).tolist() == [3, 2] and not overflow.any()
    _, _, overflow = pack_contact_points((torch.ones(10, 1), points, None, None, counts, starts), 2, 2)
    assert overflow.tolist() == [True, False]


def test_two_column_jacobi_svd_matches_general_svd():
    generator = torch.Generator().manual_seed(732)
    points = torch.randn(128, 64, 3, generator=generator, dtype=torch.float64)*.04
    points[..., 2] = .3*points[..., 0]-.2*points[..., 1]+.001*points[..., 2]
    mask = torch.rand(128, 64, generator=generator) > .25
    center, slope, valid = _fit(points, mask, SUPPORT_RISK_CONFIG)
    assert valid.all()
    xy = torch.where(mask[..., None], points[..., :2]-center[:, None, :2], 0.)
    z = torch.where(mask, points[..., 2]-center[:, None, 2], 0.)
    expected = (torch.linalg.pinv(xy) @ z[..., None]).squeeze(-1)
    torch.testing.assert_close(slope, expected, rtol=1e-10, atol=1e-10)
    _, slope32, valid32 = _fit(points.float(), mask, SUPPORT_RISK_CONFIG)
    assert valid32.all()
    torch.testing.assert_close(slope32.double(), expected, rtol=1e-5, atol=1e-6)
