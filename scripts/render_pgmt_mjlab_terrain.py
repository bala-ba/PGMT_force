"""Render the native collision model used by PGMT MuJoCo Warp Stage2.

Example: MUJOCO_GL=egl python scripts/render_pgmt_mjlab_terrain.py --output logs/terrain
No Isaac, generated artwork, mesh substitution or physics stepping is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ROOT / "assets/terrain/pgmt_stage2_v6_v8difficulty_mjwarp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from pmt_tasks.backends.mjlab_pgmt.terrain import add_native_terrain, load_manifest

    manifest = load_manifest(args.asset)
    spec = mujoco.MjSpec()
    add_native_terrain(spec, asset=args.asset)
    spec.worldbody.add_light(name="inspection_key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-2, -4, 8], dir=[1, 1, -3], diffuse=[.65] * 3, specular=[0.] * 3, castshadow=False)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 1000
    model.vis.headlight.ambient[:] = [.22] * 3
    model.vis.headlight.diffuse[:] = [.25] * 3
    model.vis.headlight.specular[:] = 0
    model.vis.map.zfar = 10
    model.vis.map.znear = .0001
    families = manifest["spec"]["families"]
    palette = [(0.50, .68, .83, 1), (.70, .64, .49, 1), (.86, .58, .33, 1),
               (.76, .48, .28, 1), (.58, .70, .48, 1), (.37, .71, .61, 1), (.64, .53, .78, 1)]
    for patch in manifest["patches"]:
        model.geom_rgba[patch["source_geom_ids"]] = palette[families.index(patch["family"])]
    args.output.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    title_font = ImageFont.truetype(font_path, 30)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1

    terrain_spec = manifest["spec"]
    lower, upper = np.asarray(manifest["map_bounds"])
    extent = upper - lower
    with mujoco.Renderer(model, height=1000, width=1600, max_geom=model.ngeom + 100) as renderer:
        camera.lookat[:] = [0, 0, 0]
        camera.distance, camera.azimuth, camera.elevation = float(np.linalg.norm(extent) * 1.65), 135, -65
        renderer.update_scene(data, camera=camera, scene_option=option)
        overview = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(overview)
        draw.rectangle((0, 0, 1600, 88), fill=(18, 24, 34))
        draw.text((24, 10), "PGMT Stage 2 | actual MuJoCo Warp terrain collision model", font=title_font, fill="white")
        draw.text((24, 53), f"{manifest['version']} | {terrain_spec['num_rows']} levels x {len(families)} families | "
                  f"{extent[0]:g} x {extent[1]:g} m core", font=font, fill="#bdcbd8")
        overview.save(args.output / "terrain_overview.png")

    levels = [0, 5, 9]
    width, height, header = 400, 330, 100
    canvas = Image.new("RGB", (width * len(families), header + len(levels) * height), (18, 24, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 10), "MuJoCo Warp Stage 2 | native boxes and heightfields", font=title_font, fill="white")
    patch_size = terrain_spec["patch_size"]
    draw.text((24, 53), f"Rows: L0 / L5 / L9. Original {patch_size[0]:g} x {patch_size[1]:g} m patches; family colors are display only.",
              font=font, fill="#bdcbd8")
    with mujoco.Renderer(model, height=height - 40, width=width, max_geom=model.ngeom + 100) as renderer:
        for row, level in enumerate(levels):
            for column, family in enumerate(families):
                patch = next(p for p in manifest["patches"] if p["level"] == level and p["family"] == family)
                model.geom_group[:] = 5
                model.geom_group[patch["source_geom_ids"]] = 2
                option.geomgroup[5] = 0
                camera.lookat[:] = patch["center"]
                camera.lookat[2] = patch["world_bounds"][1][2] / 2
                camera.distance, camera.azimuth, camera.elevation = max(patch_size) * 1.9, 135, -40
                renderer.update_scene(data, camera=camera, scene_option=option)
                canvas.paste(Image.fromarray(renderer.render().copy()), (column * width, header + row * height + 40))
                draw.text((column * width + 12, header + row * height + 8), f"L{level}  {family}", font=font, fill="white")
    canvas.save(args.output / "terrain_families.png")
    source = mujoco.MjModel.from_binary_path(str(args.asset / "source.mjb"))
    np.testing.assert_array_equal(model.hfield_data, source.hfield_data)
    np.testing.assert_array_equal(model.geom_type, source.geom_type)
    np.testing.assert_array_equal(model.geom_size, source.geom_size)
    record = dict(asset=str(args.asset.resolve()), model_sha256=hashlib.sha256((args.asset / "source.mjb").read_bytes()).hexdigest(),
        mujoco=mujoco.__version__, ngeom=model.ngeom, nhfield=model.nhfield,
        geometry="compiled with the training add_native_terrain function", geometry_matches_source=True,
        visualization="family colors, directional lighting; family panels hide other tiles; collision geometry unchanged",
        images=["terrain_overview.png", "terrain_families.png"], spec=manifest["spec"])
    (args.output / "render_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
