"""Conservative sphere cover of the actual compiled mjlab robot colliders."""
import mujoco
import numpy as np

from .motion import BFS_BODIES


def collision_cover(model, *, entity="robot", spacing=.02):
    bodies, points, radii = [], [], []
    for i in range(model.ngeom):
        body = model.body(model.geom_bodyid[i]).name
        if not body.startswith(entity + "/") or not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        name = body[len(entity) + 1:]
        if name not in BFS_BODIES:
            raise ValueError(f"Unmapped robot collision body: {name}")
        kind, size = model.geom_type[i], model.geom_size[i]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            centres, radius = np.zeros((1, 3)), size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            count = max(2, int(np.ceil(2 * size[1] / spacing)) + 1)
            centres = np.zeros((count, 3))
            centres[:, 2] = np.linspace(-size[1], size[1], count)
            radius = np.hypot(size[0], size[1] / (count - 1))
        else:
            raise ValueError(f"Unsupported robot collision geometry for placement: {model.geom(i).name}")
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, model.geom_quat[i])
        points.extend(centres @ matrix.reshape(3, 3).T + model.geom_pos[i])
        radii.extend([radius] * len(centres))
        bodies.extend([BFS_BODIES.index(name)] * len(centres))
    if not bodies:
        raise ValueError("No robot collision geometry found")
    return np.array(bodies), np.array(points), np.array(radii)
