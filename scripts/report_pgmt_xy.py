"""Produce static plots and a local HTML report from a completed Isaac XY census."""
import argparse
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pgmt_xy_report import aggregate, write_csv, write_reports

LABELS = dict(flat="平地", rough="粗糙地", stairs_up="凸楼梯", stairs_down="凹楼梯",
              platforms="平台", slope="坡地", boxes="箱体障碍")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    status = json.loads((root/"training_status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Only completed evaluations may produce the final report")
    protocol = json.loads((root/"protocol.json").read_text())
    rows = [json.loads(s) for s in (root/"episodes.jsonl").read_text().splitlines()]
    expected = {t["trial_id"] for t in protocol["trials"]}
    if len(rows) != len(expected) or {r["trial_id"] for r in rows} != expected:
        raise ValueError("Missing or duplicate evaluation trials")
    summary = write_reports(root, rows, protocol)
    families, levels = list(LABELS), protocol["levels"]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    by_family = {r["family"]: r for r in summary["by_family"]}
    color = ["#3979b7", "#599b80", "#db9a44", "#cc6a4a", "#8a72ae", "#5c9daa", "#a98167"]
    x = np.arange(len(families))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for shift, key, label, shade in [(-.25, "success_rate", "完整动作成功", "#3979b7"),
                                    (0, "strict_050_success_rate", "成功且全程 XY ≤ 50 cm", "#599b80"),
                                    (.25, "strict_025_success_rate", "成功且全程 XY ≤ 25 cm", "#db9a44")]:
        values = [by_family[f][key]*100 for f in families]
        bars = axes[0].bar(x+shift, values, width=.25, label=label, color=shade)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].set(ylim=(0, 110), ylabel="占全部兼容测试的比例 / %", title="完成能力与精确位置跟踪")
    axes[0].legend(fontsize=9, loc="upper right")
    for shift, key, label, shade in [(-.18, "xy_rmse_m", "所有片段（含失败前缀）", "#3979b7"),
                                    (.18, "successful_xy_rmse_m", "仅成功片段", "#599b80")]:
        bars = axes[1].bar(x+shift, [by_family[f][key]*100 if by_family[f][key] is not None else np.nan
                                   for f in families], width=.36, label=label, color=shade)
        axes[1].bar_label(bars, fmt="%.1f", fontsize=9)
    axes[1].set(ylabel="XY RMSE / cm", title="骨盆 XY 轨迹误差（按控制帧汇总）")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [LABELS[f] for f in families])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("model_19999.pt · Isaac Sim 5.1 / PhysX · v8 L0–L9", fontsize=16)
    fig.savefig(root/"overview.png")
    plt.close(fig)
    by_level = {(r["family"], r["level"]): r for r in summary["by_family_level"]}
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    for ax, key, scale, title, cmap, vmax in [(axes[0], "success_rate", 100, "完整动作成功率 / %", "YlGnBu", 100),
                                           (axes[1], "xy_rmse_m", 100, "XY RMSE / cm（含失败前缀）", "YlOrRd", None)]:
        values = np.array([[by_level[f, level][key]*scale for level in levels] for f in families])
        im = ax.imshow(values, aspect="auto", vmin=0, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(levels)), [f"L{i}" for i in levels])
        ax.set_yticks(range(len(families)), [LABELS[f] for f in families])
        ax.set_title(title)
        for i in range(len(families)):
            for j in range(len(levels)):
                ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", fontsize=10,
                        color="white" if values[i,j] > .65*(vmax or values.max()) else "black")
        fig.colorbar(im, ax=ax, shrink=.9)
    fig.savefig(root/"terrain_levels.png")
    plt.close(fig)
    categories = sorted({re.sub(r"\d+$", "", r["motion"].split("_subject")[0]) for r in rows})
    category_rows = []
    for category in categories:
        for family in families:
            subset = [r for r in rows if r["family"] == family and re.sub(r"\d+$", "", r["motion"].split("_subject")[0]) == category]
            if subset:
                category_rows.append(dict(category=category, family=family, **aggregate(subset)))
    write_csv(root/"motion_category_summary.csv", category_rows)
    locomotion = [dict(family=f, **aggregate([r for r in rows if r["family"] == f and r["motion"].startswith(("walk", "run", "sprint"))])) for f in families]
    write_csv(root/"locomotion_summary.csv", locomotion)
    examples = []
    index = {r["trial_id"]: r for r in rows}
    for p in sorted((root/"states").glob("*.npz")):
        with np.load(p) as d:
            t = json.loads(str(d["trial_json"]))
            if t["level"] == 5:
                examples.append((t, d["xy"].copy()))
    if len(examples) == 7:
        examples.sort(key=lambda item: families.index(item[0]["family"]))
        relative_paths = np.concatenate([xy.reshape(-1, 2)-xy[0, 2:] for _, xy in examples])
        lower, upper = relative_paths.min(0)-.15, relative_paths.max(0)+.15
        error_top = max(np.linalg.norm(xy[:, :2]-xy[:, 2:], axis=-1).max()*100 for _, xy in examples)
        fig, axes = plt.subplots(2, 7, figsize=(19, 6), layout="constrained")
        for col, (t, xy) in enumerate(examples):
            row = index[t["trial_id"]]
            origin = xy[0, 2:]
            axes[0, col].plot(xy[:, 2]-origin[0], xy[:, 3]-origin[1], "--", color="#777", label="参考")
            axes[0, col].plot(xy[:, 0]-origin[0], xy[:, 1]-origin[1], color=color[col], label="实际")
            axes[0, col].scatter([0], [0], color="black", s=16)
            axes[0, col].set_aspect("equal", adjustable="box")
            axes[0, col].set(title=f"{LABELS[t['family']]} · {'成功' if row['success'] else '失败'}", xlabel="X / m", ylabel="Y / m",
                             xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            time = (np.arange(len(xy))+1)*protocol["control_dt"]
            error = np.linalg.norm(xy[:, :2]-xy[:, 2:], axis=-1)
            axes[1, col].plot(time, error*100, color=color[col])
            axes[1, col].axhline(25, ls=":", color="#db9a44")
            axes[1, col].axhline(50, ls=":", color="#999")
            axes[1, col].set(xlim=(0, 10), ylim=(0, max(55, error_top*1.06)), xlabel="时间 / s", ylabel="XY 误差 / cm", title=f"RMSE {row['xy_rmse_m']*100:.1f} cm")
            for ax in axes[:, col]:
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(f"同一动作跨地形对照 · {examples[0][0]['motion']} · L5 · 原生终止处停止", fontsize=16)
        fig.savefig(root/"xy_examples.png", bbox_inches="tight", pad_inches=.15)
        plt.close(fig)

    def table(items):
        headers = ["地形", "成功 / 总数", "成功率", "XY RMSE", "X / Y RMSE", "成功片段 XY RMSE", "全程≤25 cm且完成", "全程≤50 cm且完成", "同类地形帧占比"]
        out = "<table><thead><tr>"+"".join(f"<th>{h}</th>" for h in headers)+"</tr></thead><tbody>"
        for r in items:
            vals = [LABELS[r["family"]], f"{r['successes']:,} / {r['trials']:,}", f"{r['success_rate']:.2%}",
                    f"{r['xy_rmse_m']*100:.2f} cm", f"{r['x_rmse_m']*100:.2f} / {r['y_rmse_m']*100:.2f} cm",
                    f"{r['successful_xy_rmse_m']*100:.2f} cm" if r['successful_xy_rmse_m'] is not None else "—",
                    f"{r['strict_025_success_rate']:.2%}", f"{r['strict_050_success_rate']:.2%}", f"{r['same_family_fraction']:.2%}"]
            out += "<tr>"+"".join(f"<td>{v}</td>" for v in vals)+"</tr>"
        return out+"</tbody></table>"

    overall = summary["overall"]
    images = ["overview.png", "terrain_levels.png", "xy_examples.png"]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PGMT model_19999 Isaac 地形评测</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1450px;margin:36px auto;padding:0 24px;color:#172b3a}}h1{{font-size:30px}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dfe5e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#edf3f7}}img{{width:100%;margin-top:20px}}code{{background:#edf3f7;padding:2px 6px}}.small{{font-size:14px;color:#536674}}a{{color:#276a9e}}video{{width:100%}}</style>
<h1>PGMT model_19999：Isaac 地形运动与 XY 跟踪评测</h1>
<p>已完成 <b>{len(rows):,}</b> 次完整动作测试。总体原生成功率 <b>{overall['success_rate']:.2%}</b>，
XY RMSE <b>{overall['xy_rmse_m']*100:.2f} cm</b>；严格全程 ≤25 cm 且完成比例为 <b>{overall['strict_025_success_rate']:.2%}</b>。</p>
<p class="small">检查点：{html.escape(protocol['checkpoint'])}<br>SHA256：{protocol['checkpoint_sha256']}<br>
运行：{status['started_at']} — {status['updated_at']}，Isaac Sim 5.1.0 / Isaac Lab 2.3.2 / PhysX，CUDA GPU 0。</p>
<h2>评测口径</h2><ul>
<li>v8 的 7 类地形 × L0–L9；动作库共 862 个约 9.98 s 片段。楼梯按原兼容表使用 503 个片段，其余每类 862 个；每个组合执行一次，等权计数，不用 EMD 权重抽样。</li>
<li>从第 0 帧、地块中心、世界朝向 {protocol['initial_world_yaw_rad']} rad 开始；确定性 actor mean，关闭观测噪声、随机推力和随机动力学。物理 200 Hz，控制 50 Hz。</li>
<li>原生成功：到达参考末帧且无跟踪失败或地图越界。保留训练终止规则：XY 偏差 &gt;1 m 持续 0.2 s、躯干高度偏差 &gt;0.4 m 持续 0.2 s，以及原有朝向、手腕高度等规则；末帧与 time_out 同时出现仍可成功。</li>
<li>XY 误差为世界坐标中实际骨盆位置与本回合固定变换后的参考骨盆位置之差。在每个物理控制步后、参考推进及自动重置前记录，含终止帧。RMSE=√mean(Δx²+Δy²)，不是平均每条轨迹 RMSE。</li>
<li>失败片段保留至终止；后续自动重置的数据不计入。因失败前缀较短，误差不能单独替代成功率。25/50 cm 严格指标要求成功且所有记录帧均在阈值内。</li>
<li>地形归因按出生地块；原生地图允许跨相邻地形和等级，表中报告实际仍处同类地形的时间比例。此结果不等价于无限单类地形测试。</li>
<li>凸楼梯对应 stairs_up（金字塔），凹楼梯对应 stairs_down（倒金字塔）。名称描述地形形状；从中心起步后实际上下方向取决于动作轨迹，并非单独规定的上楼/下楼任务。</li>
<li>这是已训练动作库在固定初始朝向、无扰动条件下的确定性评测；没有测量随机朝向、随机起帧、推扰鲁棒性或未见动作的泛化成功率。没有更新策略或归一化统计。</li></ul>
<h2>全部兼容动作</h2>{table(summary['by_family'])}
<h2>公平对照：七类地形共有的 {summary['common_motion_count']} 条动作</h2>{table(summary['common_motion_by_family'])}
<h2>移动动作：walk / run / sprint</h2>{table(locomotion)}
<h2>图表</h2>"""
    page += "".join(f'<img src="{name}" alt="{name}">' for name in images if (root/name).exists())
    videos = sorted((root/"videos").glob("*.mp4")) if (root/"videos").exists() else []
    if videos:
        page += "<h2>同一动作的实际运动回放</h2><p>使用本次 Isaac / PhysX 采集的位姿，逐帧回放；渲染工具不运行物理仿真。失败时停在终止姿态。</p>"
        for p in videos:
            page += f'<p>{p.name}</p><video controls preload="metadata" src="videos/{p.name}"></video>'
    page += "<h2>终止原因</h2><p>同一片段可触发多个原因，以下计数不互斥。</p><table><tr><th>地形</th>"
    terms = list(overall["termination_counts"])
    page += "".join(f"<th>{k}</th>" for k in terms)+"</tr>"
    for row in summary["by_family"]:
        page += f"<tr><td>{LABELS[row['family']]}</td>"+"".join(f"<td>{row['termination_counts'][k]}</td>" for k in terms)+"</tr>"
    page += "</table><h2>可复核文件</h2><ul>"
    for name in ("terrain_summary.csv", "terrain_level_summary.csv", "common_motion_summary.csv", "locomotion_summary.csv", "motion_category_summary.csv", "episodes.csv", "summary.json", "protocol.json", "audit.json", "env.yaml", "training_status.json", "run.log"):
        page += f'<li><a href="{name}">{name}</a></li>'
    page += "</ul><p>trajectories/ 保存每个控制帧的实际/参考 XY；states/ 保存预先选定的代表性步行动作在所有地形、等级的根部位姿和关节角。</p></html>"
    (root/"report.html").write_text(page)
    print(root/"report.html")


if __name__ == "__main__":
    main()
