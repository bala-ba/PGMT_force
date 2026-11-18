"""Guard geometry identity and v6 spawn semantics across the simulator boundary."""
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pmt_tasks.pgmt_isaac_native_terrain import patch_primitives, add_primitives

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/terrain/pgmt_stage2_v6_v8difficulty_mjwarp"


def test_boxes_are_decomposed_without_filling_gaps():
    a, b = trimesh.creation.box(extents=(1., 2., 3.)), trimesh.creation.box(extents=(2., 1., 1.))
    b.apply_translation((4, 0, 0))
    result = patch_primitives(trimesh.util.concatenate([a, b]))
    assert len(result) == 2
    np.testing.assert_allclose(result[0]["pos"], a.bounds.mean(0))
    np.testing.assert_allclose(result[1]["size"], b.extents / 2)


def test_non_box_mesh_is_rejected_instead_of_convexified():
    with pytest.raises(ValueError, match="not an axis-aligned box"):
        patch_primitives(trimesh.creation.icosphere(subdivisions=1))


def test_asymmetric_heightfield_preserves_diagonal_and_signed_height():
    mujoco = pytest.importorskip("mujoco")
    from pmt_tasks.pgmt_mujoco_mesh import heightfield_mesh
    heights = np.array([[0., .3, .1], [.4, .8, .2]])
    full = heightfield_mesh(heights, [1., 1., 1., 1.])
    mesh = trimesh.Trimesh(full.vertices[:6] - [0, 0, .2], full.faces[:4], process=False)
    primitives = patch_primitives(mesh)
    spec = mujoco.MjSpec()
    add_primitives(spec, primitives, name="signed")
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for triangle in mesh.triangles:
        point = triangle.mean(0)
        distance = mujoco.mj_ray(model, data, np.r_[point[:2], 5.], np.array([0., 0., -1.]),
                                None, True, -1, np.empty(1, dtype=np.int32))
        assert 5. - distance == pytest.approx(point[2], abs=1e-7)
    # Deliberately use the opposite diagonal in one cell.
    mesh.faces[[0, 2]] = [[0, 1, 3], [1, 4, 3]]
    with pytest.raises(ValueError, match="diagonal"):
        patch_primitives(mesh)


def test_default_mjlab_stage2_selects_corrected_isaac_v6():
    pytest.importorskip("mjlab")
    from pmt_tasks.builder import build_env_cfg
    cfg = build_env_cfg("pgmt_stage2", backend="mjlab")
    assert Path(cfg.commands["motion"].terrain_asset) == ASSET


@pytest.mark.skipif(not (ASSET / "manifest.json").exists(), reason="Export archived Isaac terrain first")
def test_converted_asset_keeps_isaac_difficulty_and_center_spawns():
    mujoco = pytest.importorskip("mujoco")
    if mujoco.__version__ != "3.8.0":
        pytest.skip("Native model was exported with MuJoCo 3.8.0")
    import torch
    import warp as wp
    from pmt_tasks.backends.mjlab_pgmt.terrain import Terrain
    wp.config.kernel_cache_dir = "/tmp/pgmt-mjlab-warp"
    source = json.loads((ROOT / "assets/terrain/pgmt_stage2_v6_v8difficulty/manifest.json").read_text())
    terrain = Terrain(ASSET, 50, "cpu", torch.zeros(50, 3))
    for field in ("families", "region_size", "patch_size", "patch_grid", "spawn_patch_grid", "family_level_difficulty"):
        assert terrain.manifest["spec"][field] == source["spec"][field]
    assert terrain.allowed_patch_ids.tolist() == [5, 6, 9, 10]
    for _ in range(4):
        _, _, ids = terrain.sample(torch.arange(50))
        assert set(ids.tolist()) <= {5, 6, 9, 10}
    # Training's query mesh must agree with the real compiled collider at all
    # 50 family/level regions, not just the conversion script's local models.
    xy = terrain.terrain_origins[..., :2].reshape(-1, 2) + torch.tensor([.371, -.213])
    queried = terrain.height(xy).numpy()
    model = mujoco.MjModel.from_binary_path(str(ASSET / "source.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    native = [5. - mujoco.mj_ray(model, data, np.r_[point, 5.], np.array([0., 0., -1.]),
              None, True, -1, np.empty(1, dtype=np.int32)) for point in xy.numpy()]
    np.testing.assert_allclose(queried, native, rtol=0, atol=2e-5)
