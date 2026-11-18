"""Independently recompute metrics from recorded XY prefixes and verify coverage."""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np

from pgmt_xy_report import trajectory_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--run_log", type=Path)
    args = ap.parse_args()
    root = args.source.resolve()
    status = json.loads((root/"training_status.json").read_text())
    assert status["status"] == "completed"
    assert status["source_checkpoint_unchanged"] and status["policy_state_unchanged"]
    protocol = json.loads((root/"protocol.json").read_text())
    rows = list(map(json.loads, (root/"episodes.jsonl").read_text().splitlines()))
    expected = {t["trial_id"]: t for t in protocol["trials"]}
    assert len(rows) == len(expected) == status["completed_trials"] == status["total_trials"]
    assert {r["trial_id"] for r in rows} == set(expected)
    grouped = defaultdict(list)
    for r in rows:
        for k, v in expected[r["trial_id"]].items():
            assert r[k] == v, (r["trial_id"], k)
        flags = r["termination_flags"]
        assert r["success"] == bool(flags["motion_finished"] and not r["failed"] and not flags["map_out_of_bounds"])
        assert r["failed"] == any(v for k, v in flags.items() if k not in ("time_out", "motion_finished", "map_out_of_bounds"))
        assert r["final_reference_frame"] == r["control_steps"]-1
        if r["success"]:
            assert r["final_reference_frame"] == r["motion_frames"]-1
        assert any(flags.values())
        grouped[r["trajectory_file"]].append(r)
    max_delta, checked_steps, errors = 0., 0, defaultdict(list)
    digests = {}
    for filename, episodes in sorted(grouped.items()):
        path = root/filename
        with np.load(path, allow_pickle=False) as d:
            xy, ids, counts = d["xy"], d["trial_ids"], d["control_steps"]
            assert len(episodes) == len(ids) == xy.shape[1]
            for row in episodes:
                j, n = row["trajectory_slot"], row["control_steps"]
                assert ids[j] == row["trial_id"] and counts[j] == n
                assert np.isfinite(xy[:n, j]).all() and np.isnan(xy[n:, j]).all()
                actual = trajectory_metrics(xy[:n, j])
                for k, value in actual.items():
                    delta = abs(value-row[k])
                    max_delta = max(max_delta, delta)
                    assert np.isclose(value, row[k], rtol=1e-12, atol=1e-12), (row["trial_id"], k)
                error = np.linalg.norm(xy[:n, j, :2].astype(float)-xy[:n, j, 2:].astype(float), axis=-1)
                errors[row["family"]].append(error)
                checked_steps += n
        digests[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    pooled = {}
    for family, arrays in errors.items():
        values = np.concatenate(arrays)
        pooled[family] = dict(samples=len(values), xy_p50_m=float(np.quantile(values, .5)),
                              xy_p95_m=float(np.quantile(values, .95)), xy_p99_m=float(np.quantile(values, .99)))
    assert hashlib.sha256(Path(protocol["checkpoint"]).read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    assert hashlib.sha256((Path(protocol["terrain_asset"])/"manifest.json").read_bytes()).hexdigest() == protocol["terrain_manifest_sha256"]
    for file, digest in protocol["motion_sha256"].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    log_details = {}
    if args.run_log:
        log = args.run_log.read_text()
        physics_errors = [s for s in log.splitlines() if "[omni.physx" in s and ("[Error]" in s or "overflow" in s.lower())]
        assert not physics_errors, physics_errors
        assert "Traceback (most recent call last)" not in log
        log_details = dict(native_physics_errors=0, tracebacks=0,
                           file_watcher_limit_messages=log.count("Failed to create change watch"))
        shutil.copyfile(args.run_log, root/"run.log")
    archive = root/"evaluation_source"
    archive.mkdir(exist_ok=True)
    for filename in ("eval_pgmt_xy.py", "eval_pgmt_xy.sh", "pgmt_xy_report.py", "report_pgmt_xy.py", "audit_pgmt_xy.py", "render_pgmt_xy.py"):
        shutil.copyfile(Path(__file__).parent/filename, archive/filename)
    report = dict(status="passed", trial_coverage_complete=True, audited_trials=len(rows),
        audited_control_steps=checked_steps, metrics_max_absolute_recomputation_error=max_delta,
        terminal_samples_included=True, reset_samples_excluded=True, policy_unchanged=True,
        checkpoint_unchanged=True, terrain_manifest_unchanged=True, motion_files_unchanged=True,
        pooled_xy_quantiles_by_family=pooled, trajectory_sha256=digests,
        versions={p: version(p) for p in ("isaacsim", "isaaclab", "torch", "numpy")}, **log_details)
    (root/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("trajectory_sha256", "pooled_xy_quantiles_by_family")}, indent=2))


if __name__ == "__main__":
    main()
