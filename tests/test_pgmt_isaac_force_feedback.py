"""Simulator-independent checks of the PhysX manifold math and archive parser."""
from pathlib import Path
import math

import pytest
import torch

from motion_tracking_rl.pgmt_force_feedback import force_feedback_contract
from pmt_tasks.pgmt_isaac_force_feedback import manifold_wrench, wrench_features
from pmt_tasks.pgmt_historical_config import read_archive, _restore, archived_differences


def test_disjoint_manifolds_sum_forces_and_moments_about_each_ankle():
    # Unused entries deliberately contain NaN. Pair order != buffer order.
    forces, points = torch.full((12, 3), float('nan')), torch.full((12, 3), float('nan'))
    origins = torch.tensor([[2., 1., .035], [-3., 2., .035]])
    forces[5], forces[6], forces[1] = torch.tensor([0., 0., 80.]), torch.tensor([0., 0., 20.]), torch.tensor([0., 0., 50.])
    points[5] = origins[0] + torch.tensor([.1, .02, -.035])
    points[6] = origins[0] + torch.tensor([-.1, -.02, -.035])
    points[1] = origins[1] + torch.tensor([.03, 0., -.035])
    counts, starts = torch.tensor([[0, 2], [1, 0]]), torch.tensor([[999, 5], [1, 999]])
    f, m, _, valid = manifold_wrench(forces, points, counts, starts, origins, slots=4)
    assert valid.all()
    torch.testing.assert_close(f, torch.tensor([[0., 0., 100.], [0., 0., 50.]]))
    torch.testing.assert_close(m, torch.tensor([[1.2, -6., 0.], [0., -1.5, 0.]]), atol=1e-5, rtol=1e-5)
    feedback = wrench_features(f, m, f[:, 2], torch.tensor([[1., 0., 0., 0.]]).expand(2, -1), force_feedback_contract())
    torch.testing.assert_close(feedback[:, 3:5], torch.tensor([[.06, .012], [.03, 0.]]), atol=1e-6, rtol=1e-5)
    assert feedback[:, 5].eq(1).all()


def test_friction_at_sole_plane_and_local_rotation():
    # A single support point: adding shear at the same sole point preserves CoP.
    f = torch.tensor([[20., -10., 100.]])
    r = torch.tensor([[.08, -.02, -.035]])
    m = torch.cross(r, f, dim=-1)
    # Rotate both wrench vectors 90 degrees about world Z, then invert at the foot.
    rotate = lambda v: torch.stack((-v[:, 1], v[:, 0], v[:, 2]), -1)
    q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
    value = wrench_features(rotate(f), rotate(m), torch.tensor([100.]), q, force_feedback_contract())
    torch.testing.assert_close(value, torch.tensor([[20., -10., 100., .08, -.02, 1.]]), atol=1e-5, rtol=1e-5)
    invalid = wrench_features(f, m, torch.tensor([9.]), torch.tensor([[1., 0., 0., 0.]]), force_feedback_contract())
    torch.testing.assert_close(invalid[:, :3], f)
    assert invalid[:, 3:].eq(0).all()


@pytest.mark.parametrize('case', ['overflow', 'saturated', 'out_of_bounds', 'nan'])
def test_bad_manifold_is_invalid(case):
    f, p = torch.ones(8, 3), torch.zeros(8, 3)
    counts, starts, slots = torch.tensor([[2]]), torch.tensor([[0]]), 4
    if case == 'overflow':
        counts[0, 0] = 9
    elif case == 'saturated':
        counts[0, 0], slots = 8, 8
    elif case == 'out_of_bounds':
        starts[0, 0] = 7
    else:
        f[0, 0] = float('nan')
    assert not manifold_wrench(f, p, counts, starts, torch.zeros(1, 3), slots=slots)[-1].item()


def test_pooled_physx_buffer_allows_more_contacts_on_one_foot_than_average_budget():
    f, p = torch.full((128, 3), float('nan')), torch.full((128, 3), float('nan'))
    f[:66] = torch.tensor([0., 0., 1.])
    p[:66] = torch.tensor([.1, .02, -.035])
    force, moment, _, valid = manifold_wrench(f, p, torch.tensor([[66], [0]]),
        torch.tensor([[0], [127]]), torch.zeros(2, 3), slots=64)
    assert valid.all()
    torch.testing.assert_close(force, torch.tensor([[0., 0., 66.], [0., 0., 0.]]))
    torch.testing.assert_close(moment, torch.tensor([[1.32, -6.6, 0.], [0., 0., 0.]]))


def test_archive_preserves_empty_mappings_and_safe_tags(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('x: !!python/tuple [1, 2]\ns: !!python/object/apply:builtins.slice [null, null, null]\np: {}\n')
    value = read_archive(path)
    assert value['x'] == (1, 2) and value['s'] == slice(None)
    assert _restore(dict(p=dict(unwanted=1)), dict(p={})) == dict(p={})
    assert archived_differences(dict(a=1), dict(a=2))[0]['path'] == '/a'
    path.write_text('x: !!python/object/apply:builtins.eval ["1+2"]')
    with pytest.raises(Exception):
        read_archive(path)


def test_actual_train03_archive():
    path = Path(__file__).resolve().parents[1] / 'logs/pgmt_stage2_center2x2/train_03/params/env.yaml'
    if not path.exists():
        pytest.skip('Local train_03 archive is not distributed with the repository')
    value = read_archive(path)
    motion = value['commands']['motion']
    assert motion['sampler_type'] == 'adaptive' and motion['fixed_start_frame'] == 0
    assert motion['placement_patch_grid'] == (2, 2)
    assert value['pgmt_reward_groups'] == ('upper', 'lower', 'terrain', 'auxiliary')


@pytest.mark.parametrize('change_layout', [False, True])
def test_historical_terrain_override_is_explicit_and_preserves_sampler(tmp_path, change_layout):
    import json
    from copy import deepcopy
    from types import SimpleNamespace as Obj
    from pmt_tasks.pgmt_historical_config import override_historical_terrain, audit_historical_runtime
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    base = dict(schema='test', asset_sha256='old-hash',
        spec=dict(num_rows=10, num_cols=5, families=['flat','stairs','slope','boxes','rough'],
                  region_size=[24.,24.], patch_grid=[4,4], patch_size=[6.,6.]),
        patches=[dict(region_name='L00_flat', patch_row=0, patch_col=0)])
    (old/'manifest.json').write_text(json.dumps(base))
    changed = deepcopy(base); changed['asset_sha256'] = 'new-hash'
    changed['spec']['spawn_patch_grid'] = [2,2]
    if change_layout:
        changed['spec']['spawn_patch_grid'] = [4,4]
    (new/'manifest.json').write_text(json.dumps(changed))
    cfg = Obj(pgmt_terrain_asset=str(old), commands=Obj(motion=Obj(placement_patch_grid=(2,2))),
        scene=Obj(terrain=Obj(prim_path='/World/ground', asset_directory=str(old)),
        pgmt_left_foot_terrain=Obj(), pgmt_right_foot_terrain=Obj()))
    identity = dict(terrain_sha256='old-hash', sampler=dict(beta=2), reset=dict(fixed_start_frame=0))
    report = dict(expected_curriculum_identity=deepcopy(identity))
    if change_layout:
        with pytest.raises(ValueError, match='spawn_patch_grid'):
            override_historical_terrain(cfg, report, new)
        assert cfg.pgmt_terrain_asset == str(old) and 'terrain_override' not in report
        return
    override_historical_terrain(cfg, report, new)
    assert cfg.scene.terrain.asset_directory == str(new)
    assert cfg.scene.pgmt_left_foot_terrain.filter_prim_paths_expr == ['/World/ground/terrain/Terrain/L00_flat/P00/Mesh']
    identity['terrain_sha256'] = 'new-hash'
    env = Obj(_pgmt_terrain_curriculum=Obj(identity=identity))
    audit_historical_runtime(env, report)
    assert report['expected_curriculum_identity']['terrain_sha256'] == 'old-hash'
    identity['sampler']['beta'] = 3
    with pytest.raises(ValueError, match='sampler/beta'):
        audit_historical_runtime(env, report)
