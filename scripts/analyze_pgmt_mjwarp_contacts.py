"""Offline resultant force, planar CoP and support coverage from raw MJWarp CSVs.

The convex hull coverage is a geometric support proxy, not physical contact area.
All moments include friction forces. Intended for the recorded flat, condim=3 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError


def hull_xy(points):
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points - points[0], tol=1e-10) < 2:
        return 0., points
    try:
        h = ConvexHull(points)
    except QhullError:
        return 0., points
    return float(h.volume), points[h.vertices]


def wrench_cop(positions, forces, plane_z, minimum_fz):
    force = np.asarray(forces, dtype=float).reshape(-1, 3).sum(axis=0)
    moment = np.cross(np.asarray(positions, dtype=float).reshape(-1, 3),
                      np.asarray(forces, dtype=float).reshape(-1, 3)).sum(axis=0)
    cop = np.full(2, np.nan)
    if force[2] >= minimum_fz:
        cop = np.array([(plane_z * force[0] - moment[1]) / force[2],
                        (moment[0] + plane_z * force[1]) / force[2]])
    return force, moment, cop


def geometry(xml_path):
    tree = ET.parse(xml_path)
    default_radius = float(tree.find(".//default[@class='foot_capsule']/geom").get("size").split()[0])
    result = {}
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    for side in ("left", "right"):
        body = tree.find(f".//body[@name='{side}_ankle_roll_link']")
        endpoints, boundary, bottoms, names = [], [], [], []
        for g in body.findall("geom"):
            if g.get("class") != "foot_capsule":
                continue
            p = np.fromstring(g.get("fromto"), sep=" ").reshape(2, 3)
            radius = float(g.get("size", str(default_radius)).split()[0])
            if not np.allclose(p[:, 2], p[0, 2]):
                raise ValueError("The analysis assumes coplanar capsule axes")
            endpoints.extend(p[:, :2])
            boundary.extend((p[:, None, :2] + radius * directions[None]).reshape(-1, 2))
            bottoms.extend(p[:, 2] - radius)
            names.append(g.get("name"))
        if len(names) != 7 or not np.allclose(bottoms, bottoms[0]):
            raise ValueError("Expected seven equal-bottom sole capsules")
        area, outline = hull_xy(boundary)
        endpoint_area, endpoint_outline = hull_xy(endpoints)
        result[side] = dict(plane_z_m=float(bottoms[0]), footprint_area_m2=area,
            axis_endpoint_hull_area_m2=endpoint_area, outline_xy_m=outline.tolist(),
            axis_endpoint_outline_xy_m=endpoint_outline.tolist(), collision_geoms=names)
    # No physical randomization was enabled during the recorded run.
    mass = sum(float(i.get("mass")) for i in tree.findall(".//inertial"))
    return result, mass


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
        min=float(values.min()), max=float(values.max()),
        p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)))


def self_checks():
    # A shear-loaded single contact on the reference plane must map to itself.
    p = np.array([[.1, .02, -.035]])
    f = np.array([[20., -5., 100.]])
    _, _, c = wrench_cop(p, f, -.035, 10.)
    np.testing.assert_allclose(c, p[0, :2], atol=1e-12)
    corners = np.array([[-.1, -.04, 0], [-.1, .04, 0], [.1, -.04, 0], [.1, .04, 0]])
    _, _, c = wrench_cop(corners, np.tile([0, 0, 100.], (4, 1)), 0, 10.)
    np.testing.assert_allclose(c, 0., atol=1e-12)
    assert abs(hull_xy(corners[:, :2])[0] - .016) < 1e-12
    assert hull_xy([[0, 0], [1, 0], [2, 0]])[0] == 0
    assert np.isnan(wrench_cop(np.empty((0, 3)), np.empty((0, 3)), 0, 10.)[2]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--support-threshold", type=float, default=10.)
    parser.add_argument("--point-threshold", type=float, default=.001)
    parser.add_argument("--exclude-reset-seconds", type=float, default=.1)
    args = parser.parse_args()
    if args.support_threshold <= 0 or args.point_threshold < 0 or args.exclude_reset_seconds < 0:
        parser.error("Support threshold must be positive; other thresholds must be nonnegative")
    self_checks()
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    geom, mass = geometry(G1_XML)
    run = args.run.resolve()
    output = args.output or run / "force_cop_area"
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "metadata.json").read_text())
    if meta["terrain_collision_geoms"] != ["terrain"]:
        raise ValueError("This analysis assumes the recorded horizontal plane terrain")
    frame = pd.read_csv(run / "contact_counts.csv").fillna({"termination": ""})
    points = pd.read_csv(run / "contact_points.csv")
    if not np.allclose(points.force_world_z, points.normal_force_N, atol=.001):
        raise ValueError("World Z must be the terrain normal in this flat-run analysis")
    for side in geom:
        recorded = {name.split("/")[-1] for name in meta["foot_collision_geoms"][side]}
        if recorded != set(geom[side]["collision_geoms"]):
            raise ValueError("Recorded collision geometry differs from reference model")
    completed = set(frame.loc[frame.termination.eq("motion_finished"), "episode"])
    groups = {(int(s), foot): p for (s, foot), p in points.groupby(["sample", "foot"])}
    empty = points.iloc[:0]
    records = []
    for item in frame.itertuples():
        for side in ("left", "right"):
            p = groups.get((int(item.sample), side), empty)
            world_p = p[["world_x", "world_y", "world_z"]].to_numpy(float)
            foot_p = p[["foot_x", "foot_y", "foot_z"]].to_numpy(float)
            world_f = p[["force_world_x", "force_world_y", "force_world_z"]].to_numpy(float)
            foot_f = p[["force_foot_x", "force_foot_y", "force_foot_z"]].to_numpy(float)
            fw, mw, cw = wrench_cop(world_p, world_f, 0., args.support_threshold)
            ff, mf, cf = wrench_cop(foot_p, foot_f, geom[side]["plane_z_m"], args.support_threshold)
            stance = bool(fw[2] >= args.support_threshold)
            if not stance:
                cf[:] = np.nan
            weights = p.normal_force_N.to_numpy(float)
            centroid = (weights @ foot_p / weights.sum()) if stance else np.full(3, np.nan)
            area = hull_xy(foot_p[weights > args.point_threshold, :2])[0]
            area1 = hull_xy(foot_p[weights > 1., :2])[0]
            area5 = hull_xy(foot_p[weights > 5., :2])[0]
            row = dict(sample=int(item.sample), time_s=item.time_s, episode=int(item.episode),
                episode_time_s=item.episode_step * meta["sample_dt"], foot=side,
                in_analysis=int(item.episode in completed and item.episode_step * meta["sample_dt"] > args.exclude_reset_seconds + 1e-9),
                stance=int(stance), cop_valid=int(np.isfinite(cf).all()), contacts=len(p),
                loaded_contacts=int((weights > args.point_threshold).sum()),
                force_norm_N=float(np.linalg.norm(fw)), horizontal_force_N=float(np.linalg.norm(fw[:2])),
                shear_to_normal=float(np.linalg.norm(fw[:2]) / fw[2]) if stance else np.nan,
                support_area_m2=area, support_coverage=area / geom[side]["footprint_area_m2"],
                support_coverage_1N=area1 / geom[side]["footprint_area_m2"],
                support_coverage_5N=area5 / geom[side]["footprint_area_m2"],
                endpoint_envelope_coverage=area / geom[side]["axis_endpoint_hull_area_m2"],
                cop_world_x_m=cw[0], cop_world_y_m=cw[1], cop_foot_x_m=cf[0], cop_foot_y_m=cf[1],
                normal_centroid_foot_x_m=centroid[0], normal_centroid_foot_y_m=centroid[1],
                cop_vs_normal_centroid_m=float(np.linalg.norm(cf-centroid[:2])))
            for name, values in (("force_world", fw), ("force_foot", ff), ("moment_world", mw), ("moment_foot", mf)):
                row.update({f"{name}_{a}": v for a, v in zip("xyz", values)})
            # A CoP removes both in-plane moments on the chosen reference plane.
            if row["cop_valid"]:
                residual = mf - np.cross(np.r_[cf, geom[side]["plane_z_m"]], ff)
                np.testing.assert_allclose(residual[:2], 0, atol=1e-9)
            if not np.isclose(fw[2], getattr(item, f"{side}_normal_force_N"), atol=.001):
                raise ValueError("Force sum does not match recorded counts table")
            records.append(row)
    data = pd.DataFrame(records)
    data.to_csv(output / "per_frame_metrics.csv", index=False)
    selected = data[data.in_analysis.eq(1)]
    report = dict(source=str(run), samples_all=len(frame), samples_analyzed=len(selected)//2,
        completed_episodes=sorted(int(v) for v in completed), nominal_robot_mass_kg=mass,
        nominal_weight_N=mass*9.81, geometry=geom, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        model_xml=str(G1_XML), model_xml_sha256=hashlib.sha256(G1_XML.read_bytes()).hexdigest(),
        definitions=dict(resultant="sum of all recorded contact force vectors, including friction",
            cop=f"wrench-equivalent CoP on ankle_roll_link z=-0.035 m plane, Fz_foot and Fz_world >= {args.support_threshold} N",
            world_cop="wrench-equivalent CoP on horizontal world z=0 plane",
            area=f"convex hull of Fn>{args.point_threshold} N contact points in foot XY divided by convex hull of projected sole capsule boundaries; not physical contact area"), feet={})
    columns = ["force_norm_N", "force_world_z", "horizontal_force_N", "shear_to_normal", "cop_foot_x_m",
        "cop_foot_y_m", "support_coverage", "support_area_m2", "support_coverage_1N", "support_coverage_5N",
        "endpoint_envelope_coverage", "cop_vs_normal_centroid_m"]
    for side in ("left", "right"):
        all_foot = data[data.foot.eq(side)]
        f = selected[selected.foot.eq(side)]
        stance = f[f.stance.eq(1)]
        cop_points = stance.loc[stance.cop_valid.eq(1), ["cop_foot_x_m", "cop_foot_y_m"]].to_numpy()
        outline_hull = ConvexHull(np.array(geom[side]["outline_xy_m"]))
        boundary_distances = cop_points @ outline_hull.equations[:, :2].T + outline_hull.equations[:, 2]
        report["feet"][side] = dict(stance_fraction=float(f.stance.mean()), samples=len(f),
            stance_samples=len(stance), cop_samples=int(stance.cop_valid.sum()),
            all_phase_mean_force_N=f[["force_world_x","force_world_y","force_world_z"]].mean().tolist(),
            vertical_impulse_Ns=float(f.force_world_z.sum()*meta["sample_dt"]),
            zero_hull_in_stance_fraction=float((stance.support_area_m2 < 1e-10).mean()),
            below_5percent_coverage_in_stance_fraction=float((stance.support_coverage < .05).mean()),
            broad_support_in_stance_fraction=float((stance.support_coverage>=.5).mean()),
            cop_outside_nominal_footprint_fraction=float((boundary_distances.max(axis=1) > 1e-5).mean()) if len(cop_points) else None,
            reference_endpoint_envelope_over_footprint=geom[side]["axis_endpoint_hull_area_m2"] / geom[side]["footprint_area_m2"],
            cop_forefoot_fraction=float((stance.loc[stance.cop_valid.eq(1),'cop_foot_x_m']>.08).mean()),
            cop_heel_fraction=float((stance.loc[stance.cop_valid.eq(1),'cop_foot_x_m']<0).mean()),
            stance_stats={c:stats(stance[c]) for c in columns},
            all_3000_frames_peak_force_N=float(all_foot.force_norm_N.max()),
            all_3000_frames_peak_vertical_force_N=float(all_foot.force_world_z.max()))
    left = selected[selected.foot.eq("left")].set_index("sample")
    right = selected[selected.foot.eq("right")].set_index("sample")
    report["combined"] = dict(
        double_support_fraction=float((left.stance.eq(1)&right.stance.eq(1)).mean()),
        flight_or_under_threshold_fraction=float((left.stance.eq(0)&right.stance.eq(0)).mean()),
        total_vertical_force_N=stats(left.force_world_z+right.force_world_z),
        left_vertical_impulse_share=float(left.force_world_z.sum()/(left.force_world_z.sum()+right.force_world_z.sum())),
        vertical_force_mean_over_nominal_weight=float((left.force_world_z+right.force_world_z).mean()/(mass*9.81)))
    report["validation"] = dict(analytic_shear_cop_and_hull_tests=True, csv_force_sums_match=True,
        cop_inplane_moment_residual_below_1e_9_Nm=True,
        finite_force_values=bool(np.isfinite(data.force_norm_N).all()),
        max_world_foot_force_norm_difference_N=float(np.max(np.abs(data.force_norm_N - np.linalg.norm(data[["force_foot_x","force_foot_y","force_foot_z"]],axis=1)))))
    (output/"metrics_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    plot(data, report, output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('geometry',)},indent=2,ensure_ascii=False))


def plot(data, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
    fig, axes = plt.subplots(4,1,figsize=(14,11),sharex=True,layout='constrained')
    for side,label,color in [('left','左脚','#16897E'),('right','右脚','#B2438E')]:
        f=data[data.foot.eq(side)&data.episode.eq(0)]
        t=f.episode_time_s
        axes[0].plot(t,f.force_world_z,color=color,lw=1,label=label+' Fz')
        axes[0].plot(t,f.horizontal_force_N,color=color,lw=.8,ls=':',alpha=.7,label=label+' 水平合力')
        axes[1].plot(t,1000*f.cop_foot_x_m,color=color,lw=1,label=label)
        axes[2].plot(t,1000*f.cop_foot_y_m,color=color,lw=1,label=label)
        v=f.support_coverage.where(f.stance.eq(1))
        axes[3].plot(t,100*v,color=color,lw=1,label=label)
        axes[3].plot(t,100*f.support_coverage_1N.where(f.stance.eq(1)),color=color,ls=':',lw=.8,alpha=.7)
    for ax in axes:
        ax.grid(alpha=.2); ax.set_xlim(0,9.98); ax.axvspan(0,.1,color='gray',alpha=.12)
    axes[0].legend(ncols=4,loc='upper right',fontsize=10)
    axes[0].set_ylabel('合力（N）')
    axes[1].set_ylabel('CoP x（mm）\n脚尖方向为正')
    axes[2].set_ylabel('CoP y（mm）\n左侧方向为正')
    axes[3].set_ylabel('支撑覆盖率（%）'); axes[3].set_xlabel('首回合仿真时间（s）')
    axes[3].set_ylim(-2,100)
    fig.suptitle('G1 足部合力、CoP 与支撑覆盖率｜首个完整回合\nCoP：足底参考平面 z = −35 mm；低载荷帧留空；面积实线阈值 0.001 N，虚线 1 N',fontsize=14)
    fig.savefig(output/'force_cop_area_timeseries.png',dpi=170)
    plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(9,9),layout='constrained')
    from matplotlib.colors import Normalize
    norm=Normalize(0,9.98)
    for ax,side,label in zip(axes,['left','right'],['左脚','右脚']):
        outline=np.array(report['geometry'][side]['outline_xy_m'])*1000
        ax.fill(outline[:,1],outline[:,0],facecolor='#F0F4F6',edgecolor='#8A9DAA',lw=1.5)
        f=data[data.foot.eq(side)&data.episode.eq(0)&data.cop_valid.eq(1)]
        sc=ax.scatter(f.cop_foot_y_m*1000,f.cop_foot_x_m*1000,c=f.episode_time_s,cmap='viridis',norm=norm,s=16,alpha=.8)
        ax.set(aspect='equal',xlim=(45,-45),ylim=(-75,155),xlabel='足部 y（mm）',ylabel='足部 x（mm）',title=label+' CoP')
        ax.grid(alpha=.2)
    fig.colorbar(sc,ax=axes,label='首回合时间（s）',shrink=.8)
    fig.suptitle('CoP 足底轨迹与碰撞轮廓\n足部坐标原点为 ankle_roll_link；轮廓为 capsule 投影凸包',fontsize=14)
    fig.savefig(output/'cop_footprint.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
