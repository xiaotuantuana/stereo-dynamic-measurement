from __future__ import annotations

import math

from stereo_research.metrics import evaluate_rows, frame_runtime_values, paired_bootstrap_ci


def _prediction(
    frame: int,
    status: str = "valid",
    left_x: float | str = 10.0,
    right_x: float | str = 8.0,
    x_m: float | str = 0.0,
    z_m: float | str = 2.0,
    total_ms: float = 5.0,
) -> dict[str, object]:
    return {
        "method": "full",
        "frame": frame,
        "point_id": "P1",
        "status": status,
        "left_x": left_x,
        "left_y": 20.0 if status == "valid" else "",
        "right_x": right_x,
        "right_y": 20.0 if status == "valid" else "",
        "X_m": x_m,
        "Y_m": 0.0 if status == "valid" else "",
        "Z_m": z_m,
        "distance_m": z_m,
        "total_ms": total_ms,
    }


def test_metrics_separate_false_matches_coverage_and_accurate_tracking() -> None:
    predictions = [
        _prediction(0),
        _prediction(1, right_x=4.0),
        _prediction(2, status="flow_failed", left_x=10.0, right_x="", x_m="", z_m=""),
        _prediction(3),
    ]
    ground_truth = [
        {
            "frame": frame,
            "point_id": "P1",
            "left_x": 10.0,
            "left_y": 20.0,
            "right_x": 8.0,
            "right_y": 20.0,
            "X_m": 0.0,
            "Y_m": 0.0,
            "Z_m": 2.0,
            "distance_m": 2.0,
        }
        for frame in range(4)
    ]

    summary = evaluate_rows(predictions, ground_truth)

    assert summary["coverage_pct"] == 75.0
    assert math.isclose(summary["false_match_rate_pct"], 100.0 / 3.0)
    assert summary["accurate_tracking_rate_pct"] == 50.0
    assert summary["gt_annotation_count"] == 4


def test_xyz_and_radial_errors_are_reported_independently() -> None:
    predictions = [
        _prediction(0, x_m=0.1, z_m=2.2),
        _prediction(1, x_m=-0.1, z_m=1.8),
    ]
    ground_truth = [
        {
            "frame": frame,
            "point_id": "P1",
            "X_m": 0.0,
            "Y_m": 0.0,
            "Z_m": 2.0,
            "distance_m": 2.0,
        }
        for frame in range(2)
    ]

    summary = evaluate_rows(predictions, ground_truth)

    assert math.isclose(summary["x_mae_m"], 0.1)
    assert math.isclose(summary["z_mae_m"], 0.2)
    assert math.isclose(summary["xyz_rmse_m"], math.sqrt(0.05))
    assert math.isclose(summary["distance_mae_m"], 0.2)
    assert summary["xyz_metric_available"] is True


def test_distance_only_truth_does_not_claim_xyz_error() -> None:
    predictions = [_prediction(0, z_m=2.1)]
    ground_truth = [{"frame": 0, "point_id": "P1", "distance_m": 2.0}]

    summary = evaluate_rows(predictions, ground_truth)

    assert summary["distance_metric_available"] is True
    assert summary["xyz_metric_available"] is False
    assert summary["x_mae_m"] is None


def test_frame_less_distance_truth_applies_to_every_frame_of_the_point() -> None:
    predictions = [
        _prediction(0, z_m=2.1),
        _prediction(1, z_m=1.9),
    ]
    ground_truth = [{"point_id": "P1", "distance_m": 2.0}]

    summary = evaluate_rows(predictions, ground_truth)

    assert summary["distance_sample_count"] == 2
    assert math.isclose(summary["distance_mae_m"], 0.1)


def test_jump_rate_detects_constant_velocity_residual() -> None:
    predictions = [
        _prediction(0, x_m=0.00),
        _prediction(1, x_m=0.01),
        _prediction(2, x_m=0.50),
        _prediction(3, x_m=0.51),
    ]

    summary = evaluate_rows(predictions)

    assert summary["jump_checks"] == 2
    assert summary["jump_count"] == 2
    assert summary["jump_rate_pct"] == 100.0


def test_jump_threshold_sensitivity_reports_two_five_and_ten_percent() -> None:
    predictions = [
        _prediction(0, x_m=0.00),
        _prediction(1, x_m=0.00),
        _prediction(2, x_m=0.08),
    ]

    summary = evaluate_rows(predictions)

    assert summary["jump_rate_2pct"] == 100.0
    assert summary["jump_rate_5pct"] == 0.0
    assert summary["jump_rate_10pct"] == 0.0


def test_jump_rate_does_not_join_independent_runtime_repeats() -> None:
    predictions = []
    for repeat in (0, 1):
        for frame, x_m in enumerate((0.0, 0.01, 0.02)):
            row = _prediction(frame, x_m=x_m)
            row["repeat"] = repeat
            predictions.append(row)

    summary = evaluate_rows(predictions)

    assert summary["jump_checks"] == 2
    assert summary["jump_count"] == 0


def test_paired_bootstrap_is_deterministic_and_orders_interval() -> None:
    lower, estimate, upper = paired_bootstrap_ci(
        baseline=[5.0, 6.0, 7.0, 8.0],
        improved=[4.0, 4.5, 6.0, 7.0],
        samples=1000,
        seed=123,
    )

    assert lower <= estimate <= upper
    assert estimate > 0


def test_runtime_summary_counts_each_frame_once_for_multiple_points() -> None:
    rows = []
    for frame, frame_ms in ((0, 10.0), (1, 20.0)):
        for point_id in ("P1", "P2"):
            row = _prediction(frame, total_ms=3.0)
            row["point_id"] = point_id
            row["frame_total_ms"] = frame_ms
            rows.append(row)

    summary = evaluate_rows(rows)

    assert summary["runtime_sample_count"] == 2
    assert summary["runtime_median_ms"] == 15.0
    assert frame_runtime_values(rows) == [10.0, 20.0]


def test_runtime_summary_does_not_fall_back_to_point_time_for_warmup_rows() -> None:
    rows = [
        {
            "method": "full",
            "repeat": 0,
            "frame": 0,
            "point_id": "P1",
            "status": "valid",
            "frame_total_ms": "",
            "total_ms": 5.0,
        }
    ]

    summary = evaluate_rows(rows)

    assert summary["runtime_sample_count"] == 0
    assert summary["runtime_mean_ms"] is None


def test_quality_summary_reports_recovery_and_stabilization_stages() -> None:
    rows = [
        {**_prediction(0), "quality_stage": "primary"},
        {**_prediction(1), "quality_stage": "context_recovery"},
        {**_prediction(2), "quality_stage": "temporal_stabilized"},
        {**_prediction(3), "quality_stage": "context_temporal_stabilized"},
    ]

    summary = evaluate_rows(rows)

    assert summary["quality_stage_counts"] == {
        "primary": 1,
        "context_recovery": 1,
        "temporal_stabilized": 1,
        "context_temporal_stabilized": 1,
    }
    assert summary["quality_context_recovery_count"] == 2
    assert summary["quality_temporal_stabilization_count"] == 2


def test_metrics_report_pixel_disparity_and_millimetre_xyz_errors() -> None:
    predictions = [
        _prediction(0, left_x=10.5, right_x=7.5, x_m=0.001, z_m=2.002),
        _prediction(1, left_x=9.5, right_x=8.5, x_m=-0.001, z_m=1.998),
    ]
    ground_truth = [
        {
            "frame": frame,
            "point_id": "P1",
            "left_x": 10.0,
            "left_y": 20.0,
            "right_x": 8.0,
            "right_y": 20.0,
            "X_m": 0.0,
            "Y_m": 0.0,
            "Z_m": 2.0,
            "distance_m": 2.0,
        }
        for frame in range(2)
    ]

    summary = evaluate_rows(predictions, ground_truth)

    assert math.isclose(summary["left_tracking_mae_px"], 0.5)
    assert math.isclose(summary["right_matching_mae_px"], 0.5)
    assert math.isclose(summary["disparity_mae_px"], 1.0)
    assert math.isclose(summary["x_mae_mm"], 1.0)
    assert math.isclose(summary["z_mae_mm"], 2.0)


def test_metrics_report_relative_displacement_error_in_millimetres() -> None:
    predictions = [
        _prediction(0, x_m=0.000, z_m=2.000),
        _prediction(1, x_m=0.012, z_m=1.995),
    ]
    ground_truth = [
        {"frame": 0, "point_id": "P1", "X_m": 0.0, "Y_m": 0.0, "Z_m": 2.0},
        {"frame": 1, "point_id": "P1", "X_m": 0.010, "Y_m": 0.0, "Z_m": 1.996},
    ]

    summary = evaluate_rows(predictions, ground_truth)

    assert math.isclose(summary["delta_x_mae_mm"], 1.0)
    assert math.isclose(summary["delta_z_mae_mm"], 0.5)
    assert summary["delta_xyz_rmse_mm"] > 0


def test_metrics_report_static_stability_jumps_and_recovery() -> None:
    rows = []
    for frame, x_m in enumerate((0.000, 0.001, 0.002, 0.010)):
        row = _prediction(frame, x_m=x_m)
        row["measured_disparity"] = (8.0, 8.1, 7.9, 8.2)[frame]
        row["recovery_attempt_count"] = 2
        row["recovery_success_count"] = 1
        rows.append(row)
    rows.append(_prediction(4, status="lost", x_m="", z_m=""))

    summary = evaluate_rows(rows)

    assert summary["x_std_mm"] > 0
    assert math.isclose(summary["peak_to_peak_x_mm"], 10.0)
    assert math.isclose(summary["drift_x_mm"], 10.0)
    assert summary["jump_rate_1mm"] is not None
    assert summary["innovation_rmse_mm"] is not None
    assert summary["lost_count"] == 1
    assert summary["recovery_attempt_count"] == 2
    assert summary["recovery_success_count"] == 1
    assert summary["recovery_success_rate_pct"] == 50.0


def test_static_stability_does_not_mix_absolute_positions_of_different_points() -> None:
    rows = []
    for point_id, x_m in (("P1", 0.0), ("P2", 1.0)):
        for frame in (0, 1, 2):
            row = _prediction(frame, x_m=x_m)
            row["point_id"] = point_id
            rows.append(row)

    summary = evaluate_rows(rows)

    assert summary["x_std_mm"] == 0.0
    assert summary["peak_to_peak_x_mm"] == 0.0


def test_research_quality_metrics_are_reported_from_raw_fields() -> None:
    rows = []
    for frame, values in enumerate(
        (
            (0.2, "cycle_valid", "valid", 2, 0.03, 8.0, 8.0, True),
            (0.8, "cycle_soft", "valid", 4, 0.08, 8.2, 8.1, True),
            (2.0, "cycle_recovered", "icgn_not_converged", 6, 0.20, 7.8, 7.95, False),
        )
    ):
        cycle_error, cycle_status, icgn_status, iterations, residual, measured, estimated, applied = values
        row = _prediction(frame)
        row.update(
            {
                "method": "research_full",
                "cycle_error_px": cycle_error,
                "cycle_status": cycle_status,
                "icgn_status": icgn_status,
                "icgn_iterations": iterations,
                "icgn_residual": residual,
                "measured_disparity": measured,
                "estimated_disparity": estimated,
                "compensation_applied": applied,
                "point_role": "reference",
                "raw_delta_X_mm": float(frame * 2),
                "compensated_delta_X_mm": float(frame) if applied else "",
            }
        )
        rows.append(row)
    rows.append({**_prediction(3, status="cycle_failed"), "cycle_status": "cycle_failed"})

    summary = evaluate_rows(rows)

    assert summary["cycle_sample_count"] == 3
    assert summary["cycle_failure_count"] == 1
    assert summary["cycle_recovery_count"] == 1
    assert summary["cycle_error_median_px"] == 0.8
    assert summary["icgn_attempt_count"] == 3
    assert summary["icgn_success_count"] == 2
    assert math.isclose(summary["icgn_success_rate_pct"], 200.0 / 3.0)
    assert summary["measured_disparity_std_px"] > summary["estimated_disparity_std_px"]
    assert math.isclose(summary["camera_compensation_success_rate_pct"], 200.0 / 3.0)
