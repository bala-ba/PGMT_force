"""Offline summaries of first-episode, pre-reset native Isaac XY trajectories."""
import csv
import json
from pathlib import Path

import numpy as np


def trajectory_metrics(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 4 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Expected a finite, nonempty [samples,4] valid trajectory prefix")
    delta = xy[:, :2]-xy[:, 2:]
    squared = np.square(delta)
    error = np.linalg.norm(delta, axis=-1)
    return dict(xy_mean_m=float(error.mean()), xy_rmse_m=float(np.sqrt(squared.sum(-1).mean())),
        x_rmse_m=float(np.sqrt(squared[:, 0].mean())), y_rmse_m=float(np.sqrt(squared[:, 1].mean())),
        xy_p95_m=float(np.quantile(error, .95)), xy_max_m=float(error.max()), xy_final_m=float(error[-1]),
        xy_error_sum_m=float(error.sum()), x_error_sq_sum_m2=float(squared[:, 0].sum()),
        y_error_sq_sum_m2=float(squared[:, 1].sum()),
        within_010_count=int((error <= .1).sum()), within_025_count=int((error <= .25).sum()),
        within_050_count=int((error <= .5).sum()))


def aggregate(rows):
    n = len(rows)
    steps = sum(r["control_steps"] for r in rows)
    sx = sum(r["x_error_sq_sum_m2"] for r in rows)
    sy = sum(r["y_error_sq_sum_m2"] for r in rows)
    successes = [r for r in rows if r["success"]]
    successful_steps = sum(r["control_steps"] for r in successes)
    return dict(trials=n, successes=len(successes), success_rate=len(successes)/n,
        control_steps=steps, xy_mean_m=sum(r["xy_error_sum_m"] for r in rows)/steps,
        xy_rmse_m=float(np.sqrt((sx+sy)/steps)), x_rmse_m=float(np.sqrt(sx/steps)), y_rmse_m=float(np.sqrt(sy/steps)),
        successful_xy_rmse_m=float(np.sqrt(sum(r["x_error_sq_sum_m2"]+r["y_error_sq_sum_m2"] for r in successes)/successful_steps)) if successes else None,
        mean_episode_xy_rmse_m=float(np.mean([r["xy_rmse_m"] for r in rows])),
        p95_episode_xy_rmse_m=float(np.quantile([r["xy_rmse_m"] for r in rows], .95)),
        mean_seconds=float(np.mean([r["seconds"] for r in rows])),
        mean_reference_completion=float(np.mean([r["reference_completion"] for r in rows])),
        strict_025_success_rate=sum(r["success"] and r["xy_max_m"] <= .25 for r in rows)/n,
        strict_050_success_rate=sum(r["success"] and r["xy_max_m"] <= .5 for r in rows)/n,
        within_010_fraction=sum(r["within_010_count"] for r in rows)/steps,
        within_025_fraction=sum(r["within_025_count"] for r in rows)/steps,
        within_050_fraction=sum(r["within_050_count"] for r in rows)/steps,
        same_family_fraction=sum(r["same_family_fraction"]*r["control_steps"] for r in rows)/steps,
        same_tile_fraction=sum(r["same_tile_fraction"]*r["control_steps"] for r in rows)/steps,
        termination_counts={k: sum(r["termination_flags"][k] for r in rows) for k in rows[0]["termination_flags"]})


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()} for row in rows)


def write_reports(output, rows, protocol):
    output = Path(output)
    families = list(protocol["compatible_motion_counts"])
    by_family, by_level = [], []
    for family in families:
        selected = [r for r in rows if r["family"] == family]
        if not selected:
            continue
        by_family.append(dict(family=family, **aggregate(selected)))
        for level in protocol["levels"]:
            subset = [r for r in selected if r["level"] == level]
            if subset:
                by_level.append(dict(family=family, level=level, **aggregate(subset)))
    # The common compatible set enables comparisons without stair exclusion bias.
    motion_families = {}
    for row in rows:
        motion_families.setdefault(row["motion"], set()).add(row["family"])
    common = {m for m, f in motion_families.items() if f == set(families)}
    common_family = ([dict(family=f, **aggregate([r for r in rows if r["family"] == f and r["motion"] in common]))
                      for f in families] if common else [])
    summary = dict(overall=aggregate(rows), by_family=by_family, by_family_level=by_level,
                   common_motion_count=len(common), common_motion_by_family=common_family)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    write_csv(output/"episodes.csv", rows)
    write_csv(output/"terrain_summary.csv", by_family)
    write_csv(output/"terrain_level_summary.csv", by_level)
    if common_family:
        write_csv(output/"common_motion_summary.csv", common_family)
    return summary


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1])
    rows = [json.loads(s) for s in (root/"episodes.jsonl").read_text().splitlines()]
    write_reports(root, rows, json.loads((root/"protocol.json").read_text()))
