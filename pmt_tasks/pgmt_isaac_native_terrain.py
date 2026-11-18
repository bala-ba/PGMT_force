"""Lossless surface conversion of archived Isaac v6 patches to MuJoCo primitives.

Only verified axis-aligned boxes, horizontal planes and regular, consistently
triangulated heightfields are accepted. Unsupported meshes fail closed. Open
planes/heightfields acquire a solid base below their lowest vertex; their top
surface is unchanged (apart from archived float32 grid-coordinate rounding).
"""
import numpy as np


def patch_primitives(mesh):
    vertices = np.asarray(mesh.vertices)
    lower, upper = mesh.bounds
    if len(vertices) == 4 and len(mesh.faces) == 2 and upper[2] == lower[2]:
        corners = np.array([[x, y] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])])
        if set(map(tuple, vertices[:, :2])) != set(map(tuple, corners)):
            raise ValueError("Plane must cover its complete rectangular bounds")
        size = (upper - lower) / 2
        size[2] = .05
        return [dict(kind="box", pos=(lower + upper) / 2 - [0, 0, .05], size=size)]
    xs, ys = np.unique(vertices[:, 0]), np.unique(vertices[:, 1])
    if len(vertices) == len(xs) * len(ys) and len(np.unique(vertices[:, :2], axis=0)) == len(vertices):
        # Isaac stores X-major vertices; MuJoCo stores rows=Y, columns=X.
        ix, iy = np.searchsorted(xs, vertices[:, 0]), np.searchsorted(ys, vertices[:, 1])
        indices = iy * len(xs) + ix
        if len(np.unique(indices)) != len(vertices):
            raise ValueError("Heightfield has multiple elevations at a grid vertex")
        if not (np.allclose(xs, np.linspace(*xs[[0, -1]], len(xs)), atol=3e-7, rtol=0)
                and np.allclose(ys, np.linspace(*ys[[0, -1]], len(ys)), atol=3e-7, rtol=0)):
            raise ValueError("Heightfield grid is not uniformly spaced")
        a = (np.arange(len(ys) - 1)[:, None] * len(xs) + np.arange(len(xs) - 1)).ravel()
        expected = np.vstack((np.column_stack((a, a + 1, a + len(xs) + 1)),
                              np.column_stack((a, a + len(xs) + 1, a + len(xs)))))
        if (len(expected) != len(mesh.faces) or set(map(tuple, np.sort(expected, axis=1)))
                != set(map(tuple, np.sort(indices[mesh.faces], axis=1)))):
            raise ValueError("Heightfield triangles differ from the MuJoCo diagonal")
        heights = np.empty((len(ys), len(xs)))
        heights[iy, ix] = vertices[:, 2]
        zmin, zmax = heights.min(), heights.max()
        scale = max(float(zmax - zmin), .001)
        return [dict(kind="hfield", pos=[(xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2, zmin],
                     size=[(xs[-1] - xs[0]) / 2, (ys[-1] - ys[0]) / 2, scale, 1.],
                     heights=((heights - zmin) / scale).astype(np.float32))]
    result = []
    for component in mesh.split(only_watertight=False):
        lo, hi = component.bounds
        corners = {(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])}
        # Eight corners alone do not establish a box: verify full rectangular
        # faces, volume and watertightness, too.
        if (len(component.faces) != 12 or set(map(tuple, component.vertices)) != corners
                or not component.is_watertight or not component.is_winding_consistent
                or not np.isclose(abs(component.volume), np.prod(hi - lo), atol=1e-8, rtol=1e-6)
                or not np.all(np.max(np.abs(component.face_normals), axis=1) > 1 - 1e-10)):
            raise ValueError("Mesh component is not an axis-aligned box")
        result.append(dict(kind="box", pos=(lo + hi) / 2, size=(hi - lo) / 2))
    if not result:
        raise ValueError("Empty terrain mesh")
    return result


def add_primitives(spec, primitives, *, name, center=(0., 0., 0.)):
    import mujoco
    names = []
    for i, primitive in enumerate(primitives):
        geom_name = f"{name}_{i}"
        kwargs = dict(name=geom_name, pos=np.asarray(center) + primitive["pos"],
                      friction=[1., .005, .0001], condim=3, group=2, rgba=[.5, .6, .65, 1.])
        if primitive["kind"] == "box":
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=primitive["size"], **kwargs)
        else:
            heights = primitive["heights"]
            spec.add_hfield(name=geom_name, nrow=heights.shape[0], ncol=heights.shape[1],
                            size=primitive["size"], userdata=heights.ravel())
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=geom_name, **kwargs)
        names.append(geom_name)
    return names
