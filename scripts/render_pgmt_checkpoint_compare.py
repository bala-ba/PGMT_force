"""Render paired native Isaac trials with exact placed reference ghosts."""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

import render_pgmt_xy as terrain_render

LABELS = dict(zip(terrain_render.FAMILIES, ["平地", "粗糙地", "凸楼梯", "凹楼梯", "平台", "坡地", "箱体障碍"]))
MODELS = [("center2x2", "A · center2x2 / train_03", "训练地形 v6"),
          ("v8_emd", "B · v8_emd", "训练地形 v8")]


def quat_rotation(wxyz):
    return Rotation.from_quat(np.roll(wxyz, -1, axis=-1))


def writer(path):
    if path.exists():
        raise FileExistsError(path)
    return imageio.get_writer(path, fps=25, codec="libx264", quality=None, macro_block_size=16,
        ffmpeg_params=["-crf", "19", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    args = ap.parse_args()
    root = args.source.resolve()
    output = root/"videos"
    output.mkdir(exist_ok=True)
    protocols, records = [], {}
    for label, _, _ in MODELS:
        run = root/label
        status = json.loads((run/"training_status.json").read_text())
        assert status["status"] == "completed" and status["policy_state_unchanged"] and status["source_checkpoint_unchanged"]
        protocol = json.loads((run/"protocol.json").read_text())
        protocols.append(protocol)
        assert len(protocol["motion_files"]) == 1
        source = Path(protocol["motion_files"][0])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["motion_sha256"][str(source)]
        with np.load(source, allow_pickle=False) as d:
            motion = dict(d)
        rows = {r["trial_id"]: r for r in map(json.loads, (run/"episodes.jsonl").read_text().splitlines())}
        for file in (run/"states").glob("*.npz"):
            with np.load(file, allow_pickle=False) as d:
                data = dict(d)
            trial = json.loads(str(data["trial_json"]))
            row = rows[trial["trial_id"]]
            n = row["control_steps"]
            assert len(data["root_pos"]) == n
            rotation = quat_rotation(data["reference_rotation"])
            ref_positions = rotation.apply(motion["body_pos_w"].reshape(-1, 3)).reshape(motion["body_pos_w"].shape)+data["reference_translation"]
            ref_quats = np.roll((rotation*quat_rotation(motion["body_quat_w"][:, 0])).as_quat(), 1, axis=-1)
            np.testing.assert_allclose(ref_positions[:n, 0, :2], data["xy"][:, 2:], atol=1e-4, rtol=0)
            data.update(ref_positions=ref_positions, ref_quats=ref_quats, reference_joint_pos=motion["joint_pos"])
            records[label, trial["family"]] = (row, data)
    for key in ("terrain_asset", "terrain_manifest_sha256", "motion_sha256", "levels", "seed", "initial_world_yaw_rad", "spawn_offset_x_m", "original_terminations"):
        assert protocols[0][key] == protocols[1][key], key
    for family in terrain_render.FAMILIES:
        a, b = (records[label, family][1] for label, _, _ in MODELS)
        for key in ("reference_rotation", "reference_translation"):
            np.testing.assert_allclose(a[key], b[key], atol=1e-6, rtol=0)
    manifest_path = Path(protocols[0]["terrain_asset"])/"manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocols[0]["terrain_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    half_size = np.asarray(manifest["spec"]["region_size"])/2
    centers = {p["family"]: np.asarray(p["center"][:2]) for p in manifest["patches"] if p["level"] == 9}
    assets = output/"render_assets"
    assets.mkdir(exist_ok=True)
    terrain_render.W, terrain_render.H = 960, 640
    model = terrain_render.build_model(manifest_path.parent, assets)
    actual, reference = mujoco.MjData(model), mujoco.MjData(model)
    exemplar = next(iter(records.values()))[1]
    joints = model.jnt_qposadr[[model.joint(str(n)).id for n in exemplar["joint_names"]]]
    bodies = [model.body(str(n)).id for n in exemplar["body_names"]]
    options, ghost_options = mujoco.MjvOption(), mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[1:3] = 1
    ghost_options.geomgroup[:] = 0
    ghost_options.geomgroup[1] = 1
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 125., -24.
    renderer = mujoco.Renderer(model, width=960, height=640)
    perturb = mujoco.MjvPerturb()
    fonts = {n: ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n) for n in (16, 18, 20, 22, 26, 30)}
    motion_name = Path(protocols[0]["motion_files"][0]).stem
    frame_ids = list(range(0, len(motion["joint_pos"]), 2))
    proof = dict(source="Isaac Sim / PhysX native trajectories", render_only=True, physics_integration_steps=0,
                 common_terrain=protocols[0]["terrain_asset"], motion=motion_name,
                 shared_placement_verified=True, actual_fk_max_error_m=0., reference_fk_max_error_m=0., videos=[])

    def render(row, data, index):
        index = min(index, row["control_steps"]-1)
        actual.qpos[:3], actual.qpos[3:7] = data["root_pos"][index], data["root_quat"][index]
        actual.qpos[joints], actual.qvel[:] = data["joint_pos"][index], 0
        mujoco.mj_forward(model, actual)
        error = float(np.linalg.norm(actual.xpos[bodies]-data["body_pos"][index], axis=-1).max())
        proof["actual_fk_max_error_m"] = max(proof["actual_fk_max_error_m"], error)
        assert error < .001 and actual.time == 0
        reference.qpos[:3], reference.qpos[3:7] = data["ref_positions"][index, 0], data["ref_quats"][index]
        reference.qpos[joints], reference.qvel[:] = data["reference_joint_pos"][index], 0
        mujoco.mj_forward(model, reference)
        error = float(np.linalg.norm(reference.xpos[bodies]-data["ref_positions"][index], axis=-1).max())
        proof["reference_fk_max_error_m"] = max(proof["reference_fk_max_error_m"], error)
        assert error < .001 and reference.time == 0
        renderer.update_scene(actual, camera=camera, scene_option=options)
        first = renderer.scene.ngeom
        mujoco.mjv_addGeoms(model, reference, ghost_options, perturb, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), renderer.scene)
        for geom in renderer.scene.geoms[first:renderer.scene.ngeom]:
            geom.rgba[:], geom.matid, geom.emission = [.02, .8, .95, .32], -1, .1
        return Image.fromarray(renderer.render())

    comparison_rows = []
    overview_path = output/"L9_straight_all_terrains_compare.mp4"
    per_terrain = {f: output/f"L9_{f}_center2x2_vs_v8_emd.mp4" for f in terrain_render.FAMILIES}
    try:
        with ExitStack() as stack:
            overview_writer = stack.enter_context(writer(overview_path))
            individual = {f: stack.enter_context(writer(path)) for f, path in per_terrain.items()}
            for fi, step in enumerate(frame_ids):
                stamp = (step+1)*protocols[0]["control_dt"]
                overview = Image.new("RGB", (2688, 896), (15, 24, 35))
                od = ImageDraw.Draw(overview)
                od.text((18, 5), f"L9 七地形直走 · 两个 model_19999 对比 · {motion_name} · t={stamp:.2f}s", font=fonts[30], fill="white")
                od.text((18, 49), "统一 v8 L9 · 相同参考与起点 · 确定性策略 · 灰色=实际 / 青色=参考 · 失败后保持终止帧", font=fonts[22], fill=(158, 205, 229))
                for mi, (_, title, training_terrain) in enumerate(MODELS):
                    y = 96+mi*368
                    od.text((18, y), f"{title}  |  {training_terrain}", font=fonts[22], fill=(160, 203, 255) if mi == 0 else (141, 229, 186))
                for col, family in enumerate(terrain_render.FAMILIES):
                    pair = [records[label, family] for label, _, _ in MODELS]
                    points = []
                    for row, data in pair:
                        k = min(step, row["control_steps"]-1)
                        points.extend([data["root_pos"][k], data["ref_positions"][k, 0]])
                    points = np.asarray(points)
                    camera.lookat[:] = points.mean(0)+[.1, 0., -.08]
                    camera.distance = max(3.9, 3.+np.linalg.norm(points-points.mean(0), axis=-1).max())
                    canvas = Image.new("RGB", (1920, 800), (15, 24, 35))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((18, 4), f"v8 L9 · {LABELS[family]} · 直走 · {motion_name} · t={stamp:.2f}s", font=fonts[28] if 28 in fonts else fonts[26], fill="white")
                    for mi, ((label, title, training_terrain), (row, data)) in enumerate(zip(MODELS, pair)):
                        k = min(step, row["control_steps"]-1)
                        image = render(row, data, k)
                        canvas.paste(image, (mi*960, 80))
                        draw.text((mi*960+18, 44), f"{title}  |  {training_terrain}", font=fonts[22], fill=(143, 215, 237))
                        done = step >= row["control_steps"]-1
                        result = ("完成" if row["success"] else "提前终止") if done else "执行中"
                        local_xy = data["root_pos"][k, :2]-centers[family]
                        outside = bool((np.abs(local_xy) >= half_size).any())
                        color = (143, 229, 183) if done and row["success"] else (255, 143, 122) if done else (227, 236, 246)
                        err = float(np.linalg.norm(data["xy"][k, :2]-data["xy"][k, 2:])*100)
                        text = f"{result} · XY {err:.1f} cm · RMSE {row['xy_rmse_m']*100:.1f} cm · 终点 {row['seconds']:.2f}s"
                        draw.text((mi*960+18, 728), text, font=fonts[22], fill=color)
                        reasons = ", ".join(k for k, v in row["termination_flags"].items() if v and k not in ("time_out", "motion_finished"))
                        draw.text((mi*960+18, 765), (f"终止原因：{reasons}" if done and not row["success"] else "灰色：Isaac 实际机器人；青色：同一回合参考动作"), font=fonts[20], fill=(174, 198, 220))
                        if outside:
                            draw.rectangle((mi*960+18, 672, mi*960+490, 710), fill=(53, 28, 30))
                            draw.text((mi*960+28, 676), "实际骨盆已偏离原 L9 地块", font=fonts[22], fill=(255, 162, 140))
                        x, y = col*384, 96+mi*368
                        od.text((x+12, y+33), LABELS[family], font=fonts[20], fill="white")
                        overview.paste(image.resize((384, 256), Image.Resampling.LANCZOS), (x, y+60))
                        if outside:
                            od.rectangle((x+8, y+280, x+372, y+312), fill=(53, 28, 30))
                            od.text((x+14, y+281), "实际骨盆已偏离原 L9 地块", font=fonts[18], fill=(255, 162, 140))
                        od.text((x+10, y+322), f"{result} | XY {err:.1f}cm | {row['seconds']:.2f}s", font=fonts[18], fill=color)
                    individual[family].append_data(np.asarray(canvas))
                    if fi == len(frame_ids)//2:
                        canvas.save(output/f"L9_{family}_preview.jpg", quality=94)
                od.text((18, 843), "Isaac / PhysX 实测位姿 · 25 FPS / 1× · 本视频为单条直走动作对比，不代表全动作库成功率", font=fonts[22], fill=(172, 195, 214))
                overview_writer.append_data(np.asarray(overview))
                if fi in (0, len(frame_ids)//2, len(frame_ids)-1):
                    overview.save(output/f"overview_{fi:03d}.jpg", quality=94)
                if fi % 50 == 0:
                    print(f"[COMPARE VIDEO] {fi+1}/{len(frame_ids)}", flush=True)
    finally:
        renderer.close()
    for path in (overview_path, *per_terrain.values()):
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        assert count == len(frame_ids) and abs(seconds-count/25) < .03
        proof["videos"].append(dict(file=path.name, frames=count, fps=25, seconds=seconds,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for family in terrain_render.FAMILIES:
        for label, _, _ in MODELS:
            r = records[label, family][0]
            comparison_rows.append(dict(model=label, family=family, success=r["success"], seconds=r["seconds"],
                xy_rmse_cm=r["xy_rmse_m"]*100, xy_max_cm=r["xy_max_m"]*100,
                same_tile_fraction=r["same_tile_fraction"], termination_flags=json.dumps(r["termination_flags"])))
    with (root/"comparison.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]));w.writeheader();w.writerows(comparison_rows)
    (output/"render_audit.json").write_text(json.dumps(proof, indent=2)+"\n")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>L9 直走模型对比</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:30px auto;color:#172b3a}video{width:100%}table{border-collapse:collapse}th,td{padding:8px 16px;border-bottom:1px solid #ddd}code{background:#edf3f7}</style>'
    page += f'<h1>L9 七类地形：两个模型的直走对比</h1><p>参考动作 <code>{motion_name}</code>，9.98 秒，位移 {protocols[0]["straightness"]["endpoint_distance_m"]:.2f} m，最大横向偏移 {protocols[0]["straightness"]["max_lateral_deviation_m"]*100:.1f} cm。统一 v8 L9，相同朝向、碰撞有效起点 X 偏移 {protocols[0]["spawn_offset_x_m"]:.3f} m，无随机扰动。</p>'
    page += '<p>A 为 center2x2/train_03（v6 训练，跨地形测试）；B 为 v8_emd（v8 训练）。灰色为实际机器人，青色为原始参考。失败后保留终止帧。所有物理轨迹来自 Isaac / PhysX；MuJoCo 只作位姿校验和渲染。</p><p>每模型、每地形仅测试这条动作一次；不能据此推断全动作库的成功率。地形按出生地块归类，参考骨盆路线始终在该 L9 地块内，实际轨迹若偏离进入相邻地块，视频会标注；这不是无限单类地形测试。CSV 的 same_tile_fraction 给出停留在原 L9 地块内的帧比例。</p>'
    page += '<video controls preload="metadata" src="videos/L9_straight_all_terrains_compare.mp4"></video><table><tr><th>地形</th><th>A 结果 / XY RMSE</th><th>B 结果 / XY RMSE</th></tr>'
    for family in terrain_render.FAMILIES:
        page += f'<tr><td>{LABELS[family]}</td>'
        for label, _, _ in MODELS:
            r = records[label, family][0]
            page += f'<td>{"完成" if r["success"] else "提前终止"} {r["seconds"]:.2f}s / {r["xy_rmse_m"]*100:.1f}cm</td>'
        page += '</tr>'
    page += '</table><p><a href="comparison.csv">完整指标 CSV</a> · <a href="videos/render_audit.json">渲染校验</a></p>'
    for family, path in per_terrain.items():
        page += f'<h2>{LABELS[family]}</h2><video controls preload="metadata" src="videos/{html.escape(path.name)}"></video>'
    (root/"report.html").write_text(page+'</html>')
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
