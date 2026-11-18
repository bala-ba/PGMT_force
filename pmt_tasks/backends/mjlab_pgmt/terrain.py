"""Audited native collision geometry and Warp queries for PGMT placement/scans."""
from pathlib import Path
import hashlib
import json

import mujoco
import numpy as np
import torch
import trimesh
import warp as wp

from pmt_tasks.pgmt_terrain_distribution import allocate_terrain_types
from pmt_tasks.pgmt_terrain_layout import training_manifest, compatibility_families, patch_bounds, sample_patch_positions, centered_patch_ids
from pmt_tasks.pgmt_terrain_query import TerrainTileMap

MISS_HEIGHT = wp.constant(float("inf"))


def load_manifest(root):
    root = Path(root)
    manifest = training_manifest(json.loads((root / "manifest.json").read_text()))
    source = root / "source.mjb"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source"]["model_sha256"]:
        raise ValueError("Native MuJoCo terrain checksum mismatch")
    return manifest


def add_native_terrain(spec, *, asset):
    """Copy static boxes and heightfields exactly; never convexify a tile mesh.

    MJB files are version-specific. Failure to load must be resolved by using
    the exporting MuJoCo version, not by falling back to flat ground.
    """
    manifest = load_manifest(asset)
    try:
        model = mujoco.MjModel.from_binary_path(str(Path(asset) / "source.mjb"))
    except ValueError as error:
        version = manifest["spec"].get("source_versions", {}).get("mujoco", "the export version")
        raise ValueError(f"Cannot load native terrain; use mujoco=={version} (installed {mujoco.__version__})") from error
    if model.nq or model.nv:
        raise ValueError("PGMT native terrain must be static")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for i in range(model.nhfield):
        rows, cols = int(model.hfield_nrow[i]), int(model.hfield_ncol[i])
        start = model.hfield_adr[i]
        heights = model.hfield_data[start:start + rows * cols].reshape(rows, cols)
        spec.add_hfield(name=f"pgmt_hfield_{i}", nrow=rows, ncol=cols,
                        size=model.hfield_size[i], userdata=heights.flatten())
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        kind = int(model.geom_type[i])
        if kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_PLANE):
            raise ValueError(f"Unsupported native terrain geom: {model.geom(i).name}")
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.geom_xmat[i])
        kwargs = dict(name=f"pgmt_terrain_{i}", type=kind, pos=data.geom_xpos[i], quat=quat,
                      size=model.geom_size[i], friction=model.geom_friction[i],
                      condim=int(model.geom_condim[i]), contype=int(model.geom_contype[i]),
                      conaffinity=int(model.geom_conaffinity[i]), solref=model.geom_solref[i],
                      solimp=model.geom_solimp[i], margin=float(model.geom_margin[i]),
                      gap=float(model.geom_gap[i]), rgba=model.geom_rgba[i], group=2)
        if kind == mujoco.mjtGeom.mjGEOM_HFIELD:
            kwargs["hfieldname"] = f"pgmt_hfield_{model.geom_dataid[i]}"
        spec.worldbody.add_geom(**kwargs)


@wp.kernel
def _height(mesh: wp.uint64, xy: wp.array(dtype=wp.vec3), top: float, out: wp.array(dtype=float)):
    i = wp.tid()
    result = wp.mesh_query_ray(mesh, wp.vec3(xy[i][0], xy[i][1], top), wp.vec3(0., 0., -1.), 1000.)
    out[i] = wp.where(result.result, top - result.t, MISS_HEIGHT)


@wp.kernel
def _distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    result = wp.mesh_query_point(mesh, points[i], 1000.)
    value = float(1.e10)
    if result.result:
        value = wp.length(points[i] - wp.mesh_eval_position(mesh, result.face, result.u, result.v))
    out[i] = value


class Terrain:
    """Birth tile curriculum plus current-position queries on audited triangles."""
    def __init__(self, asset, num_envs, device, env_origins):
        root = Path(asset)
        self.manifest = load_manifest(root)
        manifest = self.manifest
        self.compatibility_families = compatibility_families(manifest)
        rows, cols = manifest["spec"]["num_rows"], manifest["spec"]["num_cols"]
        origins = torch.zeros(rows, cols, 3, device=device)
        bounds = torch.zeros(rows, cols, 2, 2, device=device)
        for tile in manifest["tiles"]:
            r, c = tile["level"], tile["family_index"]
            origins[r, c] = torch.tensor(tile["origin"], device=device)
            bounds[r, c] = torch.tensor(tile["bounds"], device=device)
        self.terrain_origins, self.tile_bounds = origins, bounds
        self.tile_map = TerrainTileMap(manifest, bounds)
        self.walkable_bounds = bounds.new_tensor(manifest.get("walkable_bounds",
                                          [self.tile_map.lower.tolist(), self.tile_map.upper.tolist()]))
        self.terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.terrain_types = allocate_terrain_types(num_envs, manifest["spec"]["families"],
                                                    manifest["spec"]["family_proportions"], device=device)
        self.env_origins = env_origins
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        self.patch_bounds = patch_bounds(manifest, device=device)
        self.allowed_patch_ids = centered_patch_ids(manifest["spec"]["patch_grid"],
            manifest["spec"].get("spawn_patch_grid", manifest["spec"]["patch_grid"]), device=device)
        meshes = []
        for record in manifest["patches"] + manifest.get("auxiliary_surfaces", []):
            path = root / record["mesh_file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["mesh_sha256"]:
                raise ValueError(f"Terrain mesh checksum mismatch: {path}")
            mesh = trimesh.load_mesh(path, process=False)
            mesh.apply_translation(record["center"])
            meshes.append(mesh)
        combined = trimesh.util.concatenate(meshes)
        self.mesh = wp.Mesh(points=wp.array(combined.vertices.astype(np.float32), dtype=wp.vec3, device=device),
                            indices=wp.array(combined.faces.astype(np.int32).flatten(), dtype=wp.int32, device=device))
        self.ray_top = float(combined.bounds[1, 2]) + 5.

    def height(self, xy):
        points = torch.zeros((*xy.shape[:-1], 3), device=xy.device)
        points[..., :2] = xy
        out = torch.empty(xy.shape[:-1], device=xy.device)
        wp.launch(_height, dim=out.numel(), inputs=[self.mesh.id, wp.from_torch(points.reshape(-1, 3), dtype=wp.vec3),
                  self.ray_top], outputs=[wp.from_torch(out.flatten())], device=self.mesh.device)
        return out

    def distance(self, points):
        points = points.float().contiguous()
        out = torch.empty(points.shape[:-1], device=points.device)
        wp.launch(_distance, dim=out.numel(), inputs=[self.mesh.id, wp.from_torch(points.reshape(-1, 3), dtype=wp.vec3)],
                  outputs=[wp.from_torch(out.flatten())], device=self.mesh.device)
        return out

    def contains(self, xy):
        return torch.isfinite(xy).all(-1) & (xy >= self.walkable_bounds[0]).all(-1) & (xy < self.walkable_bounds[1]).all(-1)

    def query(self, xy):
        result = self.tile_map.locate(xy)
        inside = self.contains(xy)
        height = self.height(torch.where(inside[..., None], xy, self.tile_map.lower))
        result.update(valid=inside & torch.isfinite(height), height=height)
        return result

    def sample(self, ids, xy_mode="center"):
        return sample_patch_positions(self.patch_bounds, self.terrain_levels[ids], self.terrain_types[ids],
                                      margin=.25, xy_mode=xy_mode, allowed_patch_ids=self.allowed_patch_ids)
