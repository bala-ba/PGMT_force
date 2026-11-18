"""Render recorded Isaac root/joint states; never integrate MuJoCo physics."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["flat", "rough", "stairs_up", "stairs_down", "platforms", "slope", "boxes"]
LABELS = ["平地", "粗糙地", "凸楼梯 stairs_up", "凹楼梯 stairs_down", "平台", "坡地", "箱体障碍"]
COLORS = [".40 .50 .60 1", ".42 .52 .39 1", ".63 .45 .28 1", ".59 .38 .27 1",
          ".49 .42 .57 1", ".34 .53 .51 1", ".55 .46 .36 1"]
W, H = 384, 288


def build_model(asset_dir, output):
    source = ROOT/"TCRS/assets/g1/g1_29dof_rev_1_0.xml"
    xml = ET.parse(source).getroot()
    xml.find("compiler").set("meshdir", str(source.parent/"meshes"))
    for v in xml.findall("visual"):
        xml.remove(v)
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth=str(W), offheight=str(H))
    ET.SubElement(visual, "quality", shadowsize="1024", offsamples="2")
    ET.SubElement(visual, "headlight", ambient=".4 .4 .4", diffuse=".5 .5 .5", specular="0 0 0")
    assets, world = xml.find("asset"), xml.find("worldbody")
    for group in xml.findall("asset"):
        for tex in list(group.findall("texture")):
            if tex.get("type") == "skybox":
                group.remove(tex)
    for node in list(world):
        if node.tag == "light" or (node.tag == "geom" and node.get("type") == "plane"):
            world.remove(node)
    ET.SubElement(assets, "texture", name="xy_sky", type="skybox", builtin="gradient",
                  rgb1=".65 .74 .82", rgb2=".87 .91 .94", width="256", height="1536")
    ET.SubElement(world, "light", pos="0 -10 30", dir="0 1 -3", directional="true", diffuse=".6 .6 .6")
    manifest = json.loads((asset_dir/"manifest.json").read_text())
    parts = manifest["patches"]+manifest.get("auxiliary_surfaces", [])
    for i, part in enumerate(parts):
        source_mesh = asset_dir/part["mesh_file"]
        if hashlib.sha256(source_mesh.read_bytes()).hexdigest() != part["mesh_sha256"]:
            raise ValueError("Terrain mesh changed since export")
        dest = output/f"mesh_{i:03d}.obj"
        if not dest.exists():
            trimesh.load_mesh(source_mesh, process=False).export(dest)
        name = f"xy_terrain_{i}"
        ET.SubElement(assets, "mesh", name=name, file=str(dest))
        rgba = COLORS[FAMILIES.index(part["family"])] if part.get("family") in FAMILIES else ".55 .58 .6 1"
        ET.SubElement(world, "geom", type="mesh", mesh=name, pos=" ".join(map(str, part["center"])),
                      rgba=rgba, contype="0", conaffinity="0", group="2")
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 5, 9])
    ap.add_argument("--overwrite", action="store_true", help="Replace this evaluator's previously rendered videos")
    args = ap.parse_args()
    source = args.source.resolve()
    protocol = json.loads((source/"protocol.json").read_text())
    asset = Path(protocol["terrain_asset"])
    if hashlib.sha256((asset/"manifest.json").read_bytes()).hexdigest() != protocol["terrain_manifest_sha256"]:
        raise ValueError("Recorded terrain manifest changed")
    rows = {r["trial_id"]: r for r in map(json.loads, (source/"episodes.jsonl").read_text().splitlines())}
    recorded = {}
    for path in (source/"states").glob("*.npz"):
        with np.load(path, allow_pickle=False) as a:
            d = dict(a)
        trial = json.loads(str(d["trial_json"]))
        recorded[trial["level"], trial["family"]] = (trial, d)
    output = source/"videos"
    output.mkdir(exist_ok=True)
    assets = output/"render_assets"
    assets.mkdir(exist_ok=True)
    model = build_model(asset, assets)
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[1:3] = 1
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.6, 125., -27.
    renderer = mujoco.Renderer(model, width=W, height=H)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts = {n: ImageFont.truetype(font_path, n) for n in (16, 18, 22, 26)}
    proof_path = output/"render_audit.json"
    proofs = json.loads(proof_path.read_text()) if proof_path.exists() else []
    if args.overwrite:
        proofs = [p for p in proofs if p["level"] not in args.levels]
    try:
        for level in args.levels:
            clips = [recorded[level, family] for family in FAMILIES]
            video = output/f"level{level}_seven_terrains.mp4"
            if video.exists() and not args.overwrite:
                raise FileExistsError(video)
            dt, fps = protocol["control_dt"], 25
            count = int(np.ceil(max(len(d["root_pos"]) for _, d in clips)*dt*fps))
            joints = model.jnt_qposadr[[model.joint(str(n)).id for n in clips[0][1]["joint_names"]]]
            for t, d in clips:
                if not np.isfinite(d["root_pos"]).all() or not np.isfinite(d["joint_pos"]).all():
                    raise ValueError("Nonfinite pose")
                if len(d["root_pos"]) != rows[t["trial_id"]]["control_steps"]:
                    raise ValueError("Recorded episode duration mismatch")
                np.testing.assert_allclose(d["root_pos"][:, :2], d["xy"][:, :2], atol=1e-5, rtol=0)
            terminal_cache = {}
            with imageio.get_writer(video, fps=fps, codec="libx264", quality=None, macro_block_size=16,
                    ffmpeg_params=["-crf", "20", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"]) as writer:
                for fi in range(count):
                    time_s = (fi+1)/fps
                    canvas = Image.new("RGB", (7*W, 432), (19, 29, 41))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((18, 4), f"PGMT model_19999 · L{level} · {clips[0][0]['motion']} · t={time_s:.2f}s", font=fonts[26], fill="white")
                    draw.text((18, 40), "Isaac / PhysX 实测位姿回放 · 同一动作 / 七类地形 · 1× 速度 · 失败后保持终止姿态", font=fonts[18], fill=(172, 197, 221))
                    for col, (trial, d) in enumerate(clips):
                        x = col*W
                        k = min(max(0, int(round(time_s/dt))-1), len(d["root_pos"])-1)
                        row = rows[trial["trial_id"]]
                        done = k == len(d["root_pos"])-1
                        if col in terminal_cache:
                            frame = terminal_cache[col]
                        else:
                            data.qpos[:3] = d["root_pos"][k]
                            data.qpos[3:7] = d["root_quat"][k]
                            data.qpos[joints] = d["joint_pos"][k]
                            data.qvel[:] = 0
                            mujoco.mj_forward(model, data)
                            if data.time != 0.:
                                raise AssertionError("Replay must not integrate physics")
                            camera.lookat[:] = data.qpos[:3]+np.array([.15, 0., -.08])
                            renderer.update_scene(data, camera=camera, scene_option=options)
                            frame = Image.fromarray(renderer.render())
                            if done:
                                terminal_cache[col] = frame
                        canvas.paste(frame, (x, 99))
                        draw.text((x+12, 69), LABELS[col], font=fonts[22], fill="white")
                        err = np.linalg.norm(d["xy"][k, :2]-d["xy"][k, 2:])*100
                        label = ("完成" if row["success"] else "提前终止") if done else "执行中"
                        fill = (128, 225, 171) if done and row["success"] else (255, 145, 125) if done else (215, 226, 238)
                        draw.text((x+12, 390), f"{label} · XY {err:.1f} cm · {row['seconds']:.2f}s", font=fonts[18], fill=fill)
                    writer.append_data(np.asarray(canvas))
                    if fi == count//2:
                        canvas.save(output/f"level{level}_preview.png")
            proofs.append(dict(level=level, frames=count, fps=fps, file=video.name,
                               trial_ids=[t["trial_id"] for t, _ in clips], physics_integration_steps=0,
                               source="Isaac / PhysX recorded root poses and named joint angles",
                               renderer="MuJoCo FK and rasterization only; complete original terrain mesh"))
            print(f"Rendered {video}", flush=True)
    finally:
        renderer.close()
    proof_path.write_text(json.dumps(proofs, indent=2)+"\n")


if __name__ == "__main__":
    main()
