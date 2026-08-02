from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


def evaluate_rows(
    predictions: Iterable[dict[str, Any]],
    ground_truth: Iterable[dict[str, Any]] | None = None,
    pixel_threshold_px: float = 2.0,
    jump_fraction: float = 0.05,
    jump_floor_m: float = 0.05,
) -> dict[str, Any]:
    rows = list(predictions)
    gt_rows = [] if ground_truth is None else list(ground_truth)
    gt_by_key = {
        (_as_int(row.get("frame")), str(row.get("point_id"))): row
        for row in gt_rows
        if _as_int(row.get("frame")) >= 0
    }
    gt_by_point = {
        str(row.get("point_id")): row
        for row in gt_rows
        if _as_int(row.get("frame")) < 0
    }
    valid_rows = [row for row in rows if str(row.get("status")) == "valid"]
    quality_stages = Counter(
        str(row.get("quality_stage"))
        for row in valid_rows
        if str(row.get("quality_stage", ""))
    )
    runtimes = frame_runtime_values(rows)
    summary: dict[str, Any] = {
        "method": str(rows[0].get("method", "")) if rows else "",
        "row_count": len(rows),
        "valid_count": len(valid_rows),
        "coverage_pct": _percent(len(valid_rows), len(rows)),
        "effective_tracking_rate_pct": _percent(len(valid_rows), len(rows)),
        "status_counts": dict(Counter(str(row.get("status", "")) for row in rows)),
        "quality_stage_counts": dict(quality_stages),
        "quality_context_recovery_count": sum(
            count for stage, count in quality_stages.items() if "context" in stage
        ),
        "quality_temporal_stabilization_count": sum(
            count for stage, count in quality_stages.items() if "temporal" in stage
        ),
        "runtime_sample_count": len(runtimes),
        "runtime_mean_ms": float(np.mean(runtimes)) if runtimes else None,
        "runtime_median_ms": float(np.median(runtimes)) if runtimes else None,
        "runtime_p95_ms": float(np.percentile(runtimes, 95)) if runtimes else None,
        "gt_annotation_count": len(gt_rows),
    }

    jump_count, jump_checks = _trajectory_jumps(
        valid_rows,
        jump_fraction=jump_fraction,
        jump_floor_m=jump_floor_m,
    )
    summary.update(
        {
            "jump_count": jump_count,
            "jump_checks": jump_checks,
            "jump_rate_pct": _percent(jump_count, jump_checks),
        }
    )
    for label, fraction in (("2pct", 0.02), ("5pct", 0.05), ("10pct", 0.10)):
        sensitivity_count, sensitivity_checks = _trajectory_jumps(
            valid_rows,
            jump_fraction=fraction,
            jump_floor_m=jump_floor_m,
        )
        summary[f"jump_rate_{label}"] = _percent(
            sensitivity_count,
            sensitivity_checks,
        )

    pixel_gt_count = 0
    accepted_with_pixel_gt = 0
    false_matches = 0
    correct_tracks = 0
    left_errors: list[float] = []
    right_errors: list[float] = []
    disparity_errors: list[float] = []
    for row in rows:
        gt = _lookup_ground_truth(row, gt_by_key, gt_by_point)
        if gt is not None and _pixel_truth(gt) is not None:
            pixel_gt_count += 1
    for row in valid_rows:
        gt = _lookup_ground_truth(row, gt_by_key, gt_by_point)
        if gt is None:
            continue
        truth = _pixel_truth(gt)
        prediction = _pixel_prediction(row)
        if truth is None or prediction is None:
            continue
        accepted_with_pixel_gt += 1
        left_error = float(np.hypot(prediction[0] - truth[0], prediction[1] - truth[1]))
        right_error = float(np.hypot(prediction[2] - truth[2], prediction[3] - truth[3]))
        left_errors.append(left_error)
        right_errors.append(right_error)
        measured_disparity = _as_float(row.get("measured_disparity"))
        if measured_disparity is None:
            measured_disparity = prediction[0] - prediction[2]
        truth_disparity = _as_float(gt.get("gt_disparity"))
        if truth_disparity is None:
            truth_disparity = truth[0] - truth[2]
        disparity_errors.append(float(measured_disparity - truth_disparity))
        if left_error > pixel_threshold_px or right_error > pixel_threshold_px:
            false_matches += 1
        else:
            correct_tracks += 1
    summary.update(
        {
            "pixel_gt_count": pixel_gt_count,
            "accepted_with_pixel_gt": accepted_with_pixel_gt,
            "false_match_count": false_matches,
            "false_match_rate_pct": (
                _percent(false_matches, accepted_with_pixel_gt)
                if pixel_gt_count > 0
                else None
            ),
            "accurate_tracking_rate_pct": (
                _percent(correct_tracks, pixel_gt_count)
                if pixel_gt_count > 0
                else None
            ),
            "left_tracking_mae_px": _mae(left_errors),
            "right_matching_mae_px": _mae(right_errors),
            "disparity_mae_px": _mae(disparity_errors),
            "disparity_rmse_px": _rmse(disparity_errors),
        }
    )

    xyz_errors: list[np.ndarray] = []
    distance_errors: list[float] = []
    for row in valid_rows:
        gt = _lookup_ground_truth(row, gt_by_key, gt_by_point)
        if gt is None:
            continue
        predicted_xyz = _measured_xyz(row)
        truth_xyz = _xyz(gt)
        if predicted_xyz is not None and truth_xyz is not None:
            xyz_errors.append(predicted_xyz - truth_xyz)
        predicted_distance = _as_float(row.get("distance_m"))
        truth_distance = _as_float(gt.get("distance_m"))
        if predicted_distance is not None and truth_distance is not None:
            distance_errors.append(predicted_distance - truth_distance)

    if xyz_errors:
        errors = np.vstack(xyz_errors)
        norms = np.linalg.norm(errors, axis=1)
        summary.update(
            {
                "xyz_metric_available": True,
                "xyz_sample_count": int(errors.shape[0]),
                "x_mae_m": float(np.mean(np.abs(errors[:, 0]))),
                "y_mae_m": float(np.mean(np.abs(errors[:, 1]))),
                "z_mae_m": float(np.mean(np.abs(errors[:, 2]))),
                "xyz_rmse_m": float(np.sqrt(np.mean(norms**2))),
                "x_mae_mm": float(np.mean(np.abs(errors[:, 0])) * 1000.0),
                "y_mae_mm": float(np.mean(np.abs(errors[:, 1])) * 1000.0),
                "z_mae_mm": float(np.mean(np.abs(errors[:, 2])) * 1000.0),
                "xyz_rmse_mm": float(np.sqrt(np.mean(norms**2)) * 1000.0),
            }
        )
    else:
        summary.update(
            {
                "xyz_metric_available": False,
                "xyz_sample_count": 0,
                "x_mae_m": None,
                "y_mae_m": None,
                "z_mae_m": None,
                "xyz_rmse_m": None,
                "x_mae_mm": None,
                "y_mae_mm": None,
                "z_mae_mm": None,
                "xyz_rmse_mm": None,
            }
        )
    if distance_errors:
        errors = np.asarray(distance_errors, dtype=np.float64)
        summary.update(
            {
                "distance_metric_available": True,
                "distance_sample_count": int(errors.size),
                "distance_mae_m": float(np.mean(np.abs(errors))),
                "distance_rmse_m": float(np.sqrt(np.mean(errors**2))),
                "distance_mae_mm": float(np.mean(np.abs(errors)) * 1000.0),
            }
        )
    else:
        summary.update(
            {
                "distance_metric_available": False,
                "distance_sample_count": 0,
                "distance_mae_m": None,
                "distance_rmse_m": None,
                "distance_mae_mm": None,
            }
        )
    summary.update(_relative_displacement_metrics(valid_rows, gt_by_key, gt_by_point))
    summary.update(_stability_metrics(valid_rows))
    innovations = _innovation_residuals_mm(valid_rows)
    summary["innovation_rmse_mm"] = _rmse(innovations)
    summary["innovation_outlier_rate_pct"] = (
        _percent(sum(value > 5.0 for value in innovations), len(innovations))
        if innovations
        else None
    )
    for threshold in (1, 2, 5, 10):
        summary[f"jump_rate_{threshold}mm"] = (
            _percent(sum(value > threshold for value in innovations), len(innovations))
            if innovations
            else None
        )
    summary["lost_count"] = sum(str(row.get("status")) == "lost" for row in rows)
    recovery_attempts, recovery_successes = _recovery_counts(rows)
    summary["recovery_attempt_count"] = recovery_attempts
    summary["recovery_success_count"] = recovery_successes
    summary["recovery_success_rate_pct"] = (
        _percent(recovery_successes, recovery_attempts) if recovery_attempts else None
    )
    summary["mean_recovery_frames"] = _mean_recovery_frames(rows)
    summary.update(_research_quality_metrics(rows, valid_rows, gt_by_key, gt_by_point))
    return summary


def _research_quality_metrics(
    rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    gt_by_key: dict[tuple[int, str], dict[str, Any]],
    gt_by_point: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cycle_unavailable = {"", "none", "not_attempted", "not_used", "not_enabled"}
    cycle_rows = [
        row
        for row in rows
        if str(row.get("cycle_status", "")).strip().lower() not in cycle_unavailable
    ]
    cycle_errors = [
        value
        for row in cycle_rows
        if (value := _as_float(row.get("cycle_error_px"))) is not None
    ]
    cycle_status_counts = Counter(str(row.get("cycle_status")) for row in cycle_rows)
    cycle_failures = cycle_status_counts["cycle_failed"]
    cycle_recoveries = cycle_status_counts["cycle_recovered"]
    icgn_rows = [
        row
        for row in rows
        if str(row.get("icgn_status", "")) not in {"", "not_attempted"}
    ]
    icgn_successes = sum(
        str(row.get("icgn_status")) in {"valid", "icgn_converged"}
        and _as_bool(row.get("icgn_converged", True))
        for row in icgn_rows
    )
    icgn_iterations = [
        value
        for row in icgn_rows
        if (value := _as_float(row.get("icgn_iterations"))) is not None
    ]
    icgn_residuals = [
        value
        for row in icgn_rows
        if (value := _as_float(row.get("icgn_residual"))) is not None
    ]
    measured_stds = grouped_temporal_std(valid_rows, "measured_disparity")
    estimated_stds = grouped_temporal_std(valid_rows, "estimated_disparity")
    integer_errors: list[float] = []
    subpixel_errors: list[float] = []
    icgn_errors: list[float] = []
    measured_xyz_errors: list[float] = []
    estimated_xyz_errors: list[float] = []
    amplitude_ratios: list[float] = []
    amplitude_groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in valid_rows:
        gt = _lookup_ground_truth(row, gt_by_key, gt_by_point)
        if gt is None:
            continue
        truth_disparity = _as_float(gt.get("gt_disparity"))
        if truth_disparity is None:
            truth_pixels = _pixel_truth(gt)
            truth_disparity = None if truth_pixels is None else truth_pixels[0] - truth_pixels[2]
        if truth_disparity is not None:
            integer_value = _as_float(row.get("integer_disparity"))
            subpixel_value = _as_float(row.get("measured_disparity"))
            estimated_value = _as_float(row.get("estimated_disparity"))
            if integer_value is not None:
                integer_errors.append(integer_value - truth_disparity)
            if subpixel_value is not None:
                subpixel_errors.append(subpixel_value - truth_disparity)
            if str(row.get("icgn_status")) in {"valid", "icgn_converged"} and subpixel_value is not None:
                icgn_errors.append(subpixel_value - truth_disparity)
            if estimated_value is not None:
                amplitude_groups[
                    (str(row.get("repeat", "0")), str(row.get("point_id")))
                ].append((truth_disparity, estimated_value))
        truth_xyz = _xyz(gt)
        if truth_xyz is not None:
            measured_xyz = _xyz_from_prefix(row, "measured_")
            estimated_xyz = _xyz_from_prefix(row, "estimated_")
            if measured_xyz is not None:
                measured_xyz_errors.append(float(np.linalg.norm(measured_xyz - truth_xyz) * 1000.0))
            if estimated_xyz is not None:
                estimated_xyz_errors.append(float(np.linalg.norm(estimated_xyz - truth_xyz) * 1000.0))
    for pairs in amplitude_groups.values():
        if len(pairs) < 3:
            continue
        truth = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        estimate = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        truth_amplitude = 0.5 * float(np.ptp(truth))
        if truth_amplitude > 1e-12:
            amplitude_ratios.append(0.5 * float(np.ptp(estimate)) / truth_amplitude)

    compensation_unavailable = {"", "none", "not_enabled", "not_attempted", "not_used"}
    compensation_rows = [
        row
        for row in rows
        if str(row.get("compensation_status", "")).strip().lower()
        not in compensation_unavailable
    ]
    compensated_valid = [row for row in compensation_rows if _as_bool(row.get("compensation_applied"))]
    reference_residuals: list[float] = []
    for row in valid_rows:
        if str(row.get("point_role", "measurement")) != "reference":
            continue
        compensated = _delta_vector(row, "compensated_delta_")
        if compensated is not None:
            residual = float(np.linalg.norm(compensated))
            reference_residuals.append(residual)
    raw_static_values, compensated_static_values, raw_drifts, compensated_drifts = (
        _static_reference_series_metrics(valid_rows)
    )
    measured_std_stats = _distribution_metrics(measured_stds)
    estimated_std_stats = _distribution_metrics(estimated_stds)
    raw_static_stats = _distribution_metrics(raw_static_values)
    compensated_static_stats = _distribution_metrics(compensated_static_values)
    raw_drift_stats = _distribution_metrics(raw_drifts)
    compensated_drift_stats = _distribution_metrics(compensated_drifts)
    return {
        "cycle_sample_count": len(cycle_errors),
        "cycle_attempt_count": len(cycle_rows),
        "cycle_valid_count": cycle_status_counts["cycle_valid"],
        "cycle_soft_count": cycle_status_counts["cycle_soft"],
        "cycle_recovery_required_count": cycle_status_counts["cycle_recovery_required"],
        "cycle_recovered_count": cycle_recoveries,
        "cycle_error_mean_px": float(np.mean(cycle_errors)) if cycle_errors else None,
        "cycle_error_median_px": float(np.median(cycle_errors)) if cycle_errors else None,
        "cycle_error_p95_px": float(np.percentile(cycle_errors, 95)) if cycle_errors else None,
        "cycle_failure_count": cycle_failures,
        "cycle_failure_rate_pct": _percent(cycle_failures, len(cycle_rows)),
        "cycle_recovery_count": cycle_recoveries,
        "icgn_attempt_count": len(icgn_rows),
        "icgn_success_count": icgn_successes,
        "icgn_success_rate_pct": _percent(icgn_successes, len(icgn_rows)),
        "icgn_iteration_mean": float(np.mean(icgn_iterations)) if icgn_iterations else None,
        "icgn_residual_mean": float(np.mean(icgn_residuals)) if icgn_residuals else None,
        "integer_disparity_rmse_px": _rmse(integer_errors),
        "subpixel_disparity_rmse_px": _rmse(subpixel_errors),
        "icgn_disparity_rmse_px": _rmse(icgn_errors),
        "measured_disparity_std_px": measured_std_stats["mean"],
        "estimated_disparity_std_px": estimated_std_stats["mean"],
        "measured_disparity_temporal_std_mean_px": measured_std_stats["mean"],
        "measured_disparity_temporal_std_median_px": measured_std_stats["median"],
        "measured_disparity_temporal_std_p95_px": measured_std_stats["p95"],
        "estimated_disparity_temporal_std_mean_px": estimated_std_stats["mean"],
        "estimated_disparity_temporal_std_median_px": estimated_std_stats["median"],
        "estimated_disparity_temporal_std_p95_px": estimated_std_stats["p95"],
        "measured_xyz_rmse_mm": _rmse(measured_xyz_errors),
        "estimated_xyz_rmse_mm": _rmse(estimated_xyz_errors),
        "measured_innovation_rmse_mm": _series_innovation_rmse(valid_rows, "measured_"),
        "estimated_innovation_rmse_mm": _series_innovation_rmse(valid_rows, "estimated_"),
        "amplitude_ratio": float(np.mean(amplitude_ratios)) if amplitude_ratios else None,
        "camera_compensation_attempt_count": len(compensation_rows),
        "camera_compensation_available": bool(compensation_rows),
        "camera_compensation_success_rate_pct": _percent(len(compensated_valid), len(compensation_rows)),
        "camera_reference_residual_mean_mm": float(np.mean(reference_residuals)) if reference_residuals else None,
        "camera_reference_residual_p95_mm": float(np.percentile(reference_residuals, 95)) if reference_residuals else None,
        "raw_static_point_std_mm": raw_static_stats["mean"],
        "compensated_static_point_std_mm": compensated_static_stats["mean"],
        "raw_static_point_std_mean_mm": raw_static_stats["mean"],
        "raw_static_point_std_median_mm": raw_static_stats["median"],
        "raw_static_point_std_p95_mm": raw_static_stats["p95"],
        "compensated_static_point_std_mean_mm": compensated_static_stats["mean"],
        "compensated_static_point_std_median_mm": compensated_static_stats["median"],
        "compensated_static_point_std_p95_mm": compensated_static_stats["p95"],
        "raw_drift_mm": raw_drift_stats["mean"],
        "compensated_drift_mm": compensated_drift_stats["mean"],
        "raw_static_drift_mean_mm": raw_drift_stats["mean"],
        "raw_static_drift_median_mm": raw_drift_stats["median"],
        "raw_static_drift_p95_mm": raw_drift_stats["p95"],
        "compensated_static_drift_mean_mm": compensated_drift_stats["mean"],
        "compensated_static_drift_median_mm": compensated_drift_stats["median"],
        "compensated_static_drift_p95_mm": compensated_drift_stats["p95"],
    }


def grouped_temporal_std(
    rows: Iterable[dict[str, Any]],
    field: str,
    valid_only: bool = True,
) -> list[float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if valid_only and str(row.get("status")) != "valid":
            continue
        value = _as_float(row.get(field))
        if value is None:
            continue
        grouped[(str(row.get("repeat", "0")), str(row.get("point_id")))].append(value)
    return [
        float(np.std(grouped[key]))
        for key in sorted(grouped)
        if len(grouped[key]) >= 2
    ]


def _distribution_metrics(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _xyz_from_prefix(row: dict[str, Any], prefix: str) -> np.ndarray | None:
    values = [_as_float(row.get(f"{prefix}{axis}_m")) for axis in ("X", "Y", "Z")]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float64)


def _delta_vector(row: dict[str, Any], prefix: str) -> np.ndarray | None:
    values = [_as_float(row.get(f"{prefix}{axis}_mm")) for axis in ("X", "Y", "Z")]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float64)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _series_innovation_rmse(rows: list[dict[str, Any]], prefix: str) -> float | None:
    grouped: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for row in rows:
        xyz = _xyz_from_prefix(row, prefix)
        if xyz is not None:
            grouped[(str(row.get("repeat", "0")), str(row.get("point_id")))].append(
                (_as_int(row.get("frame")), xyz)
            )
    residuals: list[float] = []
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
        for first, second, third in zip(values, values[1:], values[2:]):
            if second[0] - first[0] == 1 and third[0] - second[0] == 1:
                residuals.append(float(np.linalg.norm(third[1] - (2 * second[1] - first[1])) * 1000.0))
    return _rmse(residuals)


def _static_reference_series_metrics(
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("point_role", "measurement")) == "reference":
            grouped[(str(row.get("repeat", "0")), str(row.get("point_id")))].append(row)
    raw_volatility: list[float] = []
    compensated_volatility: list[float] = []
    raw_drift: list[float] = []
    compensated_drift: list[float] = []
    for values in grouped.values():
        values.sort(key=lambda row: _as_int(row.get("frame")))
        if len(values) < 2:
            continue
        raw_xyz: list[np.ndarray | None] = []
        for row in values:
            point = _xyz_from_prefix(row, "estimated_")
            raw_xyz.append(point if point is not None else _xyz(row))
        compensated_xyz = [_xyz_from_prefix(row, "compensated_") for row in values]
        for series, volatility_target, drift_target in (
            (raw_xyz, raw_volatility, raw_drift),
            (compensated_xyz, compensated_volatility, compensated_drift),
        ):
            finite = [point for point in series if point is not None]
            if len(finite) < 2:
                continue
            array = np.vstack(finite)
            center = np.mean(array, axis=0)
            residual_norm = np.linalg.norm(array - center, axis=1)
            volatility_target.append(
                float(np.sqrt(np.mean(residual_norm**2)) * 1000.0)
            )
            drift_target.append(float(np.linalg.norm(array[-1] - array[0]) * 1000.0))
    return raw_volatility, compensated_volatility, raw_drift, compensated_drift


def _relative_displacement_metrics(
    valid_rows: list[dict[str, Any]],
    gt_by_key: dict[tuple[int, str], dict[str, Any]],
    gt_by_point: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], np.ndarray, np.ndarray]]] = defaultdict(list)
    for row in valid_rows:
        gt = _lookup_ground_truth(row, gt_by_key, gt_by_point)
        prediction_xyz = _measured_xyz(row)
        truth_xyz = None if gt is None else _xyz(gt)
        if prediction_xyz is None or truth_xyz is None:
            continue
        grouped[(str(row.get("repeat", "0")), str(row.get("point_id")))].append(
            (row, prediction_xyz, truth_xyz)
        )
    errors: list[np.ndarray] = []
    for values in grouped.values():
        values.sort(key=lambda item: _as_int(item[0].get("frame")))
        prediction_reference = values[0][1]
        truth_reference = values[0][2]
        for _row, prediction, truth in values:
            errors.append((prediction - prediction_reference) - (truth - truth_reference))
    if not errors:
        return {
            "delta_metric_available": False,
            "delta_sample_count": 0,
            "delta_x_mae_mm": None,
            "delta_y_mae_mm": None,
            "delta_z_mae_mm": None,
            "delta_xyz_rmse_mm": None,
        }
    array = np.vstack(errors) * 1000.0
    return {
        "delta_metric_available": True,
        "delta_sample_count": int(array.shape[0]),
        "delta_x_mae_mm": float(np.mean(np.abs(array[:, 0]))),
        "delta_y_mae_mm": float(np.mean(np.abs(array[:, 1]))),
        "delta_z_mae_mm": float(np.mean(np.abs(array[:, 2]))),
        "delta_xyz_rmse_mm": float(np.sqrt(np.mean(np.sum(array**2, axis=1)))),
    }


def _stability_metrics(valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        if _xyz(row) is not None:
            grouped[(str(row.get("repeat", "0")), str(row.get("point_id")))].append(row)
    standard_deviations: list[list[float]] = [[], [], []]
    peaks: list[list[float]] = [[], [], []]
    drifts: list[list[float]] = [[], [], []]
    disparities: list[float] = []
    for values in grouped.values():
        values.sort(key=lambda row: _as_int(row.get("frame")))
        xyz_values = [_xyz(row) for row in values]
        xyz_values = [value for value in xyz_values if value is not None]
        if not xyz_values:
            continue
        array = np.vstack(xyz_values) * 1000.0
        for axis in range(3):
            standard_deviations[axis].append(float(np.std(array[:, axis])))
            peaks[axis].append(float(np.ptp(array[:, axis])))
            drifts[axis].append(float(array[-1, axis] - array[0, axis]))
        disparities.extend(
            value
            for row in values
            if (value := _as_float(row.get("measured_disparity") or row.get("disparity"))) is not None
        )
    labels = ("x", "y", "z")
    result: dict[str, Any] = {}
    for axis, label in enumerate(labels):
        result[f"{label}_std_mm"] = (
            float(np.mean(standard_deviations[axis]))
            if standard_deviations[axis]
            else None
        )
        result[f"peak_to_peak_{label}_mm"] = (
            float(np.mean(peaks[axis])) if peaks[axis] else None
        )
        result[f"drift_{label}_mm"] = (
            float(np.mean(drifts[axis])) if drifts[axis] else None
        )
    result["disparity_std_px"] = float(np.std(disparities)) if disparities else None
    return result


def _innovation_residuals_mm(valid_rows: list[dict[str, Any]]) -> list[float]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        if _xyz(row) is not None:
            grouped[(str(row.get("repeat", "0")), str(row.get("point_id")))].append(row)
    residuals: list[float] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: _as_int(row.get("frame")))
        for first, second, third in zip(rows, rows[1:], rows[2:]):
            frames = [_as_int(item.get("frame")) for item in (first, second, third)]
            if frames[1] - frames[0] != 1 or frames[2] - frames[1] != 1:
                continue
            p0, p1, p2 = _xyz(first), _xyz(second), _xyz(third)
            if p0 is not None and p1 is not None and p2 is not None:
                residuals.append(float(np.linalg.norm(p2 - (2.0 * p1 - p0)) * 1000.0))
    return residuals


def _recovery_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    grouped: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        key = (str(row.get("repeat", "0")), str(row.get("point_id")))
        attempts = _as_int(row.get("recovery_attempt_count"))
        successes = _as_int(row.get("recovery_success_count"))
        previous = grouped.get(key, (0, 0))
        grouped[key] = (max(previous[0], max(0, attempts)), max(previous[1], max(0, successes)))
    return sum(value[0] for value in grouped.values()), sum(value[1] for value in grouped.values())


def _mean_recovery_frames(rows: list[dict[str, Any]]) -> float | None:
    finals: dict[tuple[str, str], tuple[int, float]] = {}
    for row in rows:
        successes = max(0, _as_int(row.get("recovery_success_count")))
        mean_frames = _as_float(row.get("mean_recovery_frames"))
        if successes <= 0 or mean_frames is None:
            continue
        key = (str(row.get("repeat", "0")), str(row.get("point_id")))
        if key not in finals or successes >= finals[key][0]:
            finals[key] = (successes, mean_frames)
    total_successes = sum(value[0] for value in finals.values())
    if total_successes <= 0:
        return None
    return float(
        sum(successes * mean_frames for successes, mean_frames in finals.values())
        / total_successes
    )


def frame_runtime_values(rows: Iterable[dict[str, Any]]) -> list[float]:
    row_list = list(rows)
    frame_runtimes: dict[tuple[str, int], float] = {}
    has_frame_runtime_column = any("frame_total_ms" in row for row in row_list)
    for row in row_list:
        value = _as_float(row.get("frame_total_ms"))
        if value is not None:
            frame_runtimes[
                (str(row.get("repeat", "0")), _as_int(row.get("frame")))
            ] = value
    if has_frame_runtime_column:
        return [frame_runtimes[key] for key in sorted(frame_runtimes)]
    return [
        value
        for row in row_list
        if (value := _as_float(row.get("total_ms"))) is not None
    ]


def paired_bootstrap_ci(
    baseline: Sequence[float],
    improved: Sequence[float],
    samples: int = 10_000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    improved_values = np.asarray(improved, dtype=np.float64)
    if baseline_values.shape != improved_values.shape or baseline_values.ndim != 1:
        raise ValueError("baseline and improved must be one-dimensional paired arrays")
    if baseline_values.size < 2 or samples < 1:
        raise ValueError("At least two pairs and one bootstrap sample are required")
    differences = baseline_values - improved_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    estimates = differences[indices].mean(axis=1)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(differences.mean()), float(upper)


def _trajectory_jumps(
    rows: list[dict[str, Any]],
    jump_fraction: float,
    jump_floor_m: float,
) -> tuple[int, int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _xyz(row) is not None:
            grouped[
                (str(row.get("repeat", "0")), str(row.get("point_id")))
            ].append(row)
    jumps = 0
    checks = 0
    for point_rows in grouped.values():
        point_rows.sort(key=lambda row: _as_int(row.get("frame")))
        for first, second, third in zip(point_rows, point_rows[1:], point_rows[2:]):
            frames = [_as_int(item.get("frame")) for item in (first, second, third)]
            if frames[1] - frames[0] != 1 or frames[2] - frames[1] != 1:
                continue
            p0 = _xyz(first)
            p1 = _xyz(second)
            p2 = _xyz(third)
            assert p0 is not None and p1 is not None and p2 is not None
            residual = float(np.linalg.norm(p2 - (2.0 * p1 - p0)))
            threshold = max(jump_floor_m, jump_fraction * abs(float(p1[2])))
            checks += 1
            if residual > threshold:
                jumps += 1
    return jumps, checks


def _pixel_truth(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    return _pixel_values(row)


def _lookup_ground_truth(
    prediction: dict[str, Any],
    by_key: dict[tuple[int, str], dict[str, Any]],
    by_point: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    point_id = str(prediction.get("point_id"))
    return by_key.get((_as_int(prediction.get("frame")), point_id)) or by_point.get(point_id)


def _pixel_prediction(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    return _pixel_values(row)


def _pixel_values(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values = tuple(
        _as_float(row.get(key))
        for key in ("left_x", "left_y", "right_x", "right_y")
    )
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[arg-type,return-value]


def _xyz(row: dict[str, Any]) -> np.ndarray | None:
    values = tuple(_as_float(row.get(key)) for key in ("X_m", "Y_m", "Z_m"))
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float64)


def _measured_xyz(row: dict[str, Any]) -> np.ndarray | None:
    measured_keys = ("measured_X_m", "measured_Y_m", "measured_Z_m")
    if any(key in row and row.get(key) not in (None, "") for key in measured_keys):
        values = tuple(_as_float(row.get(key)) for key in measured_keys)
        if all(value is not None for value in values):
            return np.asarray(values, dtype=np.float64)
    return _xyz(row)


def _mae(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64))))


def _rmse(values: Sequence[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2)))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator * 100.0)
