"""V6 geometry keeps its layout while matching the archived V8 level schedule."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pmt_tasks.pgmt_patch_terrain import family_level_difficulties, load_patch_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_patch_spec(ROOT / 'configs/terrain/pgmt_stage2_v6_v8difficulty.yaml')


def test_original_v6_schedule_and_configuration_remain_unchanged():
    original = json.loads((ROOT / SPEC['difficulty_alignment']['source_manifest']).read_text())
    spec = load_patch_spec(ROOT / 'configs/terrain/pgmt_stage2_v6.yaml')
    assert spec == original['spec']
    for values in family_level_difficulties(spec).values():
        np.testing.assert_array_equal(values, original['level_difficulty'])
    for key in ('families', 'patch_grid', 'patch_size', 'region_size', 'num_rows', 'num_cols'):
        assert SPEC[key] == spec[key]


@pytest.mark.parametrize('mutation', ['missing_family', 'short', 'nan', 'wrong_level'])
def test_invalid_level_schedule_fails_before_generating_geometry(mutation):
    spec = copy.deepcopy(SPEC)
    schedule = spec['family_level_difficulty']
    if mutation == 'missing_family':
        del schedule['stairs']
    elif mutation == 'short':
        schedule['stairs'].pop()
    elif mutation == 'nan':
        schedule['stairs'][3] = float('nan')
    else:
        schedule['stairs'][3] = .7
    with pytest.raises(ValueError, match='family_level_difficulty'):
        family_level_difficulties(spec)


def test_generated_geometry_matches_reference_levels_without_changing_family_types():
    path = ROOT / 'assets/terrain/pgmt_stage2_v6_v8difficulty/manifest.json'
    if not path.exists():
        pytest.skip('Build the matched terrain asset first')
    manifest = json.loads(path.read_text())
    alignment = SPEC['difficulty_alignment']
    reference_path = ROOT / alignment['reference_manifest']
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == alignment['reference_manifest_sha256']
    reference = json.loads(reference_path.read_text())
    lookup = {(p['family'], p['level']): p for p in reference['patches']}
    assert manifest['spec'] == SPEC
    assert manifest['region_count'] == 50 and manifest['patch_count'] == 800
    assert manifest['map_bounds'] == [[-120., -60.], [120., 60.]]
    assert SPEC['spawn_patch_grid'] == [2, 2]
    for patch in manifest['patches']:
        family = patch['family']
        ref = lookup[alignment['family_mapping'][family], patch['level']]
        assert patch['difficulty'] == ref['difficulty']
        assert patch['geometry']['native_generator'] == SPEC[family]['class_name']
        d, cfg = ref['difficulty'], ref['parameters']['source_config']
        def interpolate(key):
            lo, hi = cfg[key]
            return lo + d * (hi - lo)
        if family == 'stairs':
            assert patch['geometry']['measured_step_height_m'] == pytest.approx(interpolate('step_height_range'), abs=2e-6)
            assert patch['geometry']['verified_tread_depth_m'] == cfg['step_width']
            assert patch['parameters']['platform_width_m'] == cfg['platform_width']
        elif family == 'slope':
            assert patch['parameters']['slope_coefficient'] == pytest.approx(interpolate('slope_range'))
            measured = np.tan(np.radians(patch['geometry']['measured_slope_angle_deg']))
            assert abs(measured - interpolate('slope_range')) <= patch['geometry']['slope_coefficient_tolerance']
        elif family == 'boxes':
            assert patch['geometry']['grid_width_m'] == cfg['grid_width']
            assert patch['geometry']['input_height_amplitude_m'] == pytest.approx(interpolate('grid_height_range'))
        elif family == 'rough':
            assert patch['parameters']['rough_amplitude_m'] * 2 == pytest.approx(interpolate('height_range'))
    assert hashlib.sha256((path.parent / manifest['asset']).read_bytes()).hexdigest() == manifest['asset_sha256']
