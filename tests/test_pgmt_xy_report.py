"""Analytical checks for recorded-trajectory evaluation, independent of Isaac."""
import numpy as np
import pytest

from scripts.pgmt_xy_report import aggregate, trajectory_metrics, write_reports


def episode(xy, success):
    n = len(xy)
    return dict(control_steps=n, seconds=n*.02, success=success,
                reference_completion=1. if success else .2,
                same_family_fraction=1., same_tile_fraction=1.,
                termination_flags=dict(root_xy=not success, motion_finished=success),
                **trajectory_metrics(xy))


def test_xy_vector_error_and_axis_rmse():
    result = trajectory_metrics([[13., 14., 10., 10.], [10., 10., 10., 10.]])
    assert result["xy_mean_m"] == 2.5
    assert result["xy_rmse_m"] == pytest.approx(np.sqrt(12.5))
    assert result["x_rmse_m"] == pytest.approx(np.sqrt(4.5))
    assert result["y_rmse_m"] == pytest.approx(np.sqrt(8.))
    assert result["xy_final_m"] == 0.


def test_failures_stay_in_denominator_and_metrics_are_sample_weighted():
    failed = episode([[3., 4., 0., 0.]], False)
    successful = episode(np.zeros((3, 4)), True)
    result = aggregate([failed, successful])
    assert result["success_rate"] == .5
    assert result["xy_rmse_m"] == 2.5
    assert result["xy_mean_m"] == 1.25
    assert result["successful_xy_rmse_m"] == 0.
    assert result["strict_025_success_rate"] == .5


def test_require_explicit_valid_prefix_and_reject_missing_samples():
    recorded = np.array([[0., 0., 0., 0.], [np.nan, np.nan, np.nan, np.nan]])
    assert trajectory_metrics(recorded[:1])["xy_rmse_m"] == 0.
    with pytest.raises(ValueError):
        trajectory_metrics(recorded)
    with pytest.raises(ValueError):
        trajectory_metrics(np.zeros((0, 4)))


def test_summary_accepts_disjoint_motion_subsets_for_smoke_runs(tmp_path):
    rows = [dict(family="flat", level=0, motion="dance", **episode([[0., 0., 0., 0.]], True)),
            dict(family="stairs_up", level=0, motion="walk", **episode([[1., 0., 0., 0.]], False))]
    report = write_reports(tmp_path, rows, {"compatible_motion_counts": {"flat": 1, "stairs_up": 1}, "levels": [0]})
    assert report["common_motion_count"] == 0
    assert report["common_motion_by_family"] == []
    assert (tmp_path/"terrain_summary.csv").exists()
