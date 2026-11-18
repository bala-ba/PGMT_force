"""Render every frame of an original G1 reference clip, without policy or physics stepping."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_pgmt_stage1_recording import plane_model, FONT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    source, output = args.motion.resolve(), args.output.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.evaluation:
        protocol = json.loads((args.evaluation/"protocol.json").read_text())
        if protocol["motion_sha256"][str(source)] != digest:
            raise ValueError("Motion differs from the source used in evaluation")
    with np.load(source, allow_pickle=False) as archive:
        motion = dict(archive)
    fps = float(motion["fps"].reshape(-1)[0])
    n = len(motion["joint_pos"])
    names = motion["body_names"].tolist()
    pelvis = names.index("pelvis")
    # Only a constant XY translation keeps the scene near the rendering origin.
    origin = motion["body_pos_w"][0, pelvis].astype(float)
    origin[2] = 0.
    positions = motion["body_pos_w"].astype(float)-origin
    if not np.isfinite(positions).all() or not np.isfinite(motion["joint_pos"]).all():
        raise ValueError("Nonfinite reference pose")
    output.mkdir(parents=True, exist_ok=True)
    video = output/f"{source.stem}_original_reference.mp4"
    if video.exists():
        raise FileExistsError(video)
    model, data = plane_model(), None
    data = mujoco.MjData(model)
    body_ids = [model.body(name).id for name in names]
    joints = model.jnt_qposadr[[model.joint(name).id for name in motion["joint_names"]]]
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[1:3] = 1
    follow, fixed = mujoco.MjvCamera(), mujoco.MjvCamera()
    follow.azimuth, follow.elevation, follow.distance = 135., -22., 2.9
    fixed.azimuth, fixed.elevation = 45., -38.
    root_positions = positions[:, pelvis]
    fixed.lookat[:] = (root_positions.min(0)+root_positions.max(0))/2
    fixed.lookat[2] = .6
    fixed.distance = max(4.2, 3.+np.ptp(root_positions[:, :2], axis=0).max())
    renderer = mujoco.Renderer(model, width=960, height=640)
    fonts = {size: ImageFont.truetype(FONT, size) for size in (20, 22, 28)}
    max_position_error = 0.
    try:
        with imageio.get_writer(str(video), fps=fps, codec="libx264", quality=None, macro_block_size=16,
                ffmpeg_params=["-crf", "18", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"]) as writer:
            for i in range(n):
                data.qpos[:3] = root_positions[i]
                data.qpos[3:7] = motion["body_quat_w"][i, pelvis]
                data.qpos[joints] = motion["joint_pos"][i]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                error = float(np.linalg.norm(data.xpos[body_ids]-positions[i], axis=-1).max())
                max_position_error = max(max_position_error, error)
                if error > .001 or data.time != 0:
                    raise ValueError(f"Reference pose changed: frame={i}, error={error}, time={data.time}")
                follow.lookat[:] = data.qpos[:3]+[0., 0., -.04]
                canvas = Image.new("RGB", (1920, 800), (15, 24, 35))
                for col, camera in enumerate((follow, fixed)):
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    canvas.paste(Image.fromarray(renderer.render()), (col*960, 80))
                draw = ImageDraw.Draw(canvas)
                draw.text((20, 6), f"原始参考动作 · {source.stem}", font=fonts[28], fill=(237, 244, 252))
                draw.text((20, 44), f"G1 动作库原始数据  |  {fps:g} FPS  |  1× 速度  |  t={i/fps:.2f}s  |  帧 {i}/{n-1}",
                          font=fonts[22], fill=(133, 220, 237))
                for x, label in ((18, "跟随近景：观察步态"), (978, "固定视角：观察移动路线与转向")):
                    draw.rectangle((x-6, 91, x+410, 128), fill=(20, 31, 44))
                    draw.text((x+4, 94), label, font=fonts[22], fill=(231, 240, 249))
                draw.text((20, 732), "逐帧播放原始根部位姿与 29 个关节角；保持原始高度与朝向。", font=fonts[22], fill=(212, 228, 242))
                draw.text((20, 767), "来源：walk2_subject4，原始片段帧 1200–1500，经动作库重采样为 50 Hz。", font=fonts[20], fill=(162, 187, 210))
                writer.append_data(np.asarray(canvas))
                if i in (0, n//2, n-1):
                    canvas.save(output/f"preview_{i:03d}.jpg", quality=94)
    finally:
        renderer.close()
    encoded, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    if encoded != n or abs(seconds-n/fps) > .03:
        raise ValueError("Encoded timing differs from source")
    result = dict(video=str(video), source=str(source), source_sha256=digest,
                  video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(), frames=n, fps=fps,
                  duration_seconds=seconds, reference_time_seconds=[0., (n-1)/fps],
                  segment_id=str(motion["segment_id"]), original_frame_range=motion["source_frame"][[0,-1]].tolist(),
                  resolution=[1920,800], max_body_position_error_m=max_position_error,
                  xy_translation_subtracted_m=origin.tolist(), original_height_and_orientation_preserved=True,
                  joint_angles_unchanged=True, policy_inference=False, physics_integration_steps=0,
                  rendering="MuJoCo FK and rasterization of original reference frames")
    (output/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
