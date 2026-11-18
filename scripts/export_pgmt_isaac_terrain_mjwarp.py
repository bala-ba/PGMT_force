"""Convert archived Isaac v6 PLY collision surfaces; validate native MuJoCo rays.

This reads the completed Isaac asset instead of regenerating procedural terrain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "assets/terrain/pgmt_stage2_v6_v8difficulty")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/terrain/pgmt_stage2_v6_v8difficulty_mjwarp")
    args = parser.parse_args()
    import mujoco
    import numpy as np
    import trimesh
    from pmt_tasks.pgmt_isaac_native_terrain import patch_primitives, add_primitives
    from pmt_tasks.pgmt_terrain import ray_heights

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = json.loads((args.source / "manifest.json").read_text())
    if manifest["spec"]["type"] != "pgmt_patch_regions":
        raise ValueError("Only archived Isaac patch-region assets are supported")
    args.output.mkdir(parents=True)
    spec = mujoco.MjSpec()
    rng = np.random.default_rng(1731)
    counts, errors, names = {}, {}, {}
    grid_edge_retries = 0
    down, geom = np.array([0., 0., -1.]), np.empty(1, dtype=np.int32)

    def check(model, data, xy, expected, *, mesh=None, reference=None):
        nonlocal grid_edge_retries
        actual = np.array([5. - mujoco.mj_ray(model, data, np.r_[p, 5.], down, None, True, -1, geom) for p in xy])
        bad = np.abs(actual - expected) > 2e-6
        if bad.any() and (mesh is not None or reference is not None) and model.nhfield:
            # MuJoCo 3.8 CPU mj_ray can miss a top triangle at an exact grid
            # line for particular last-bit XY rounding. Independently query
            # BOTH surfaces 0.1 micrometre away; a geometry mismatch still fails.
            nearby = xy[bad] + [3e-8, 7e-8]
            nearby_reference = reference(nearby) if reference is not None else ray_heights(mesh, nearby)
            np.testing.assert_allclose(nearby_reference, expected[bad], atol=2e-6, rtol=0)
            actual[bad] = [5. - mujoco.mj_ray(model, data, np.r_[p, 5.], down, None, True, -1, geom) for p in nearby]
            grid_edge_retries += int(bad.sum())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
        return float(np.max(np.abs(actual - expected)))

    for index, patch in enumerate(manifest["patches"]):
        source = args.source / patch["mesh_file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != patch["mesh_sha256"]:
            raise ValueError(f"Archived Isaac mesh checksum mismatch: {source}")
        mesh = trimesh.load_mesh(source, process=False)
        primitives = patch_primitives(mesh)
        names[patch["name"]] = add_primitives(spec, primitives, name=patch["name"], center=patch["center"])
        # Query a small per-patch native model, avoiding a global all-geom scan.
        local = mujoco.MjSpec()
        add_primitives(local, primitives, name="patch")
        model = local.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        probes = np.asarray(patch["world_probes"]) - patch["center"]
        xy = rng.uniform(mesh.bounds[0, :2] + .0001, mesh.bounds[1, :2] - .0001, size=(128, 2))
        z = ray_heights(mesh, xy)
        # Both triangles of every heightfield cell are checked; this detects a
        # wrong transpose/triangle diagonal that grid-vertex tests would miss.
        if primitives[0]["kind"] == "hfield":
            centroids = mesh.triangles_center
            xy = np.vstack((xy, centroids[:, :2]))
            z = np.r_[z, centroids[:, 2]]
        xy, z = np.vstack((probes[:, :2], xy)), np.r_[probes[:, 2], z]
        error = check(model, data, xy, z, mesh=mesh)
        family = patch["family"]
        errors[family] = max(errors.get(family, 0.), error)
        counts[family] = counts.get(family, 0) + len(z)
        shutil.copy2(source, args.output / source.name)
        if (index + 1) % 80 == 0:
            print(f"Validated {index + 1}/{len(manifest['patches'])} original Isaac patches", flush=True)
    lower, upper = np.asarray(manifest["map_bounds"])
    border = manifest["spec"]["border_width"]
    outer_lo, outer_hi = lower - border, upper + border
    strips = [(outer_lo, [outer_hi[0], lower[1]]), ([outer_lo[0], upper[1]], outer_hi),
              ([outer_lo[0], lower[1]], [lower[0], upper[1]]), ([upper[0], lower[1]], [outer_hi[0], upper[1]])]
    auxiliary = []
    for i, (lo, hi) in enumerate(strips):
        lo, hi = np.asarray(lo), np.asarray(hi)
        center = np.r_[(lo + hi) / 2, 0.]
        size = np.r_[(hi - lo) / 2, .05]
        add_primitives(spec, [dict(kind="box", pos=[0, 0, -.05], size=size)], name=f"border_{i}", center=center)
        # Queries retain the original open border surface at Z=0.
        mesh = trimesh.Trimesh(vertices=[[-size[0], -size[1], 0], [size[0], -size[1], 0],
                                        [size[0], size[1], 0], [-size[0], size[1], 0]],
                               faces=[[0, 1, 2], [0, 2, 3]], process=False)
        path = args.output / f"border_{i}.ply"
        mesh.export(path)
        auxiliary.append(dict(mesh_file=path.name, mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), center=center.tolist()))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    seams = np.asarray(manifest["seam_probes"])
    def original_seam_heights(xy):
        result = []
        for point in xy:
            patch = next(p for p in manifest["patches"] if
                (point >= np.asarray(p["world_bounds"])[0, :2]).all() and
                (point <= np.asarray(p["world_bounds"])[1, :2]).all())
            mesh = trimesh.load_mesh(args.source / patch["mesh_file"], process=False)
            result.append(ray_heights(mesh, [point - np.asarray(patch["center"])[:2]])[0])
        return np.asarray(result)
    seam_error = check(model, data, seams[:, :2], seams[:, 2], reference=original_seam_heights)
    for patch in manifest["patches"]:
        patch["source_geom_ids"] = [model.geom(n).id for n in names[patch["name"]]]
    path = args.output / "source.mjb"
    mujoco.mj_saveModel(model, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["isaac_source"] = dict(asset=str(args.source.resolve()),
        manifest_sha256=hashlib.sha256((args.source / "manifest.json").read_bytes()).hexdigest(),
        usd_sha256=manifest["asset_sha256"], original_version=manifest["version"])
    manifest.update(asset="source.mjb", asset_sha256=digest, preview_only=False,
                    source=dict(model_sha256=digest, conversion="archived_isaac_native_primitives"), auxiliary_surfaces=auxiliary)
    manifest["spec"]["source_versions"] = dict(mujoco=mujoco.__version__)
    manifest["spec"]["family_proportions"] = {family: 1 / len(manifest["spec"]["families"]) for family in manifest["spec"]["families"]}
    report = dict(passed=True, source=manifest["isaac_source"], ngeom=model.ngeom, nhfield=model.nhfield,
        patch_count=len(manifest["patches"]), rays_by_family=counts, max_error_m_by_family=errors,
        seam_count=len(seams), seam_max_error_m=seam_error, tolerance_m=2e-6,
        cpu_ray_grid_edge_retries=grid_edge_retries, retry_offset_xy_m=[3e-8, 7e-8],
        checks="All archived probes + 128 random rays/patch + both triangles of every heightfield cell + all map seams",
        geometry="Original box components; original height samples and triangle diagonals; no procedural regeneration",
        added_solids="Open planes and heightfields have bases below their lowest surface; no plane crosses inverted terrain")
    manifest["validation"]["native_mujoco"] = report
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "conversion_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
