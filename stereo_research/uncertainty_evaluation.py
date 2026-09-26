"""Evaluation-only uncertainty contracts; never imported by the runtime pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Iterable, Mapping

import numpy as np


def _number(row: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                return parsed
    return None


@dataclass(frozen=True)
class GroundTruthRecord:
    frame: int | None = None
    timestamp_s: float | None = None
    point_id: str = ""
    x_mm: float | None = None
    y_mm: float | None = None
    z_mm: float | None = None
    disparity_px: float | None = None
    sigma_x_mm: float | None = None
    sigma_y_mm: float | None = None
    sigma_z_mm: float | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "GroundTruthRecord":
        frame = _number(row, "frame", "sample")
        return cls(
            frame=None if frame is None else int(frame), timestamp_s=_number(row, "timestamp_s", "timestamp"),
            point_id=str(row.get("point_id", "")), x_mm=_number(row, "gt_x_mm"),
            y_mm=_number(row, "gt_y_mm"), z_mm=_number(row, "gt_z_mm"),
            disparity_px=_number(row, "gt_disparity_px"), sigma_x_mm=_number(row, "gt_sigma_x_mm"),
            sigma_y_mm=_number(row, "gt_sigma_y_mm"), sigma_z_mm=_number(row, "gt_sigma_z_mm"),
        )


@dataclass(frozen=True)
class UncertaintyEvaluationRecord:
    frame: int | None
    point_id: str
    method: str
    evaluated_xyz_stage: str
    estimated_disparity_px: float | None
    gt_disparity_px: float | None
    disparity_error_px: float | None
    estimated_sigma_d_px: float | None
    estimated_z_mm: float | None
    gt_z_mm: float | None
    z_error_mm: float | None
    estimated_sigma_z_mm: float | None
    estimated_x_mm: float | None
    gt_x_mm: float | None
    x_error_mm: float | None
    estimated_y_mm: float | None
    gt_y_mm: float | None
    y_error_mm: float | None
    refinement_level: int | None
    precision_retry_count: int | None
    search_radius: int | None
    vision_state: str
    matching_cost: float | None
    matching_margin: float | None
    lr_error: float | None
    fb_error: float | None
    texture_quality: float | None
    precision_status: str

    @classmethod
    def from_measurement_and_gt(
        cls,
        measurement: Mapping[str, object],
        gt: GroundTruthRecord,
        *,
        xyz_stage: str = "estimated",
    ) -> "UncertaintyEvaluationRecord":
        stage_fields = {
            "measured": ("measured_X_m", "measured_Y_m", "measured_Z_m"),
            "estimated": ("estimated_X_m", "estimated_Y_m", "estimated_Z_m"),
            "compensated": ("compensated_X_m", "compensated_Y_m", "compensated_Z_m"),
            "final": ("final_X_m", "final_Y_m", "final_Z_m"),
        }
        if xyz_stage not in stage_fields:
            raise ValueError(
                "xyz_stage must be measured, estimated, compensated, or final"
            )
        x_field, y_field, z_field = stage_fields[xyz_stage]
        disparity = _number(measurement, "estimated_disparity", "disparity")
        z_m = _number(measurement, z_field)
        x_m = _number(measurement, x_field)
        y_m = _number(measurement, y_field)
        def error(value: float | None, reference: float | None) -> float | None:
            return None if value is None or reference is None else value - reference
        return cls(
            frame=None if _number(measurement, "frame") is None else int(_number(measurement, "frame") or 0),
            point_id=str(measurement.get("point_id", "")), method=str(measurement.get("method", "")),
            evaluated_xyz_stage=xyz_stage,
            estimated_disparity_px=disparity, gt_disparity_px=gt.disparity_px,
            disparity_error_px=error(disparity, gt.disparity_px), estimated_sigma_d_px=_number(measurement, "estimated_sigma_d_px"),
            estimated_z_mm=None if z_m is None else z_m * 1000.0, gt_z_mm=gt.z_mm,
            z_error_mm=error(None if z_m is None else z_m * 1000.0, gt.z_mm), estimated_sigma_z_mm=_number(measurement, "estimated_sigma_z_mm"),
            estimated_x_mm=None if x_m is None else x_m * 1000.0, gt_x_mm=gt.x_mm,
            x_error_mm=error(None if x_m is None else x_m * 1000.0, gt.x_mm),
            estimated_y_mm=None if y_m is None else y_m * 1000.0, gt_y_mm=gt.y_mm,
            y_error_mm=error(None if y_m is None else y_m * 1000.0, gt.y_mm),
            refinement_level=_int_or_none(measurement.get("policy_refinement_level")), precision_retry_count=_int_or_none(measurement.get("policy_precision_retry_count")),
            search_radius=_int_or_none(measurement.get("adaptive_search_radius", measurement.get("used_search_radius"))),
            vision_state=str(measurement.get("confidence_state", "")), matching_cost=_number(measurement, "match_cost"),
            matching_margin=_number(measurement, "uniqueness_margin_value"), lr_error=_number(measurement, "lr_error_px"),
            fb_error=_number(measurement, "flow_fb_error_px"), texture_quality=_number(measurement, "texture_std"),
            precision_status=str(measurement.get("precision_status", "")),
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _int_or_none(value: object) -> int | None:
    try:
        return None if value in (None, "") else int(float(str(value)))
    except (TypeError, ValueError):
        return None


def align_ground_truth(measurement: Mapping[str, object], records: Iterable[GroundTruthRecord], *, alignment_mode: str, max_time_offset_ms: float = 20.0) -> GroundTruthRecord | None:
    """Return an independent GT record only when its identity/time is admissible."""
    candidates = [record for record in records if record.point_id in {"", str(measurement.get("point_id", ""))}]
    if alignment_mode == "frame":
        frame = _number(measurement, "frame", "sample")
        return next((record for record in candidates if frame is not None and record.frame == int(frame)), None)
    if alignment_mode != "timestamp":
        raise ValueError("alignment_mode must be frame or timestamp")
    timestamp = _number(measurement, "timestamp_s", "frame_timestamp_s")
    timed = [record for record in candidates if record.timestamp_s is not None]
    if timestamp is None or not timed or max_time_offset_ms < 0:
        return None
    nearest = min(timed, key=lambda record: abs(float(record.timestamp_s) - timestamp))
    return nearest if abs(float(nearest.timestamp_s) - timestamp) * 1000.0 <= max_time_offset_ms else None


def _paired(records: Iterable[UncertaintyEvaluationRecord], error_name: str, sigma_name: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(getattr(record, error_name), getattr(record, sigma_name)) for record in records]
    pairs = [(float(error), float(sigma)) for error, sigma in pairs if error is not None and sigma is not None and np.isfinite(error) and np.isfinite(sigma) and sigma > 0]
    if not pairs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float); ranks[order] = np.arange(values.size, dtype=float)
    for value in np.unique(values):
        same = values == value
        ranks[same] = ranks[same].mean()
    return ranks


def _metrics(errors: np.ndarray, sigmas: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    if errors.size == 0:
        return {f"{prefix}_{key}": None for key in ("n", "rmse", "mean_sigma", "coverage_1sigma", "coverage_2sigma", "coverage_3sigma", "calibration_ratio", "pearson", "spearman")}
    absolute = np.abs(errors)
    rms_sigma = sqrt(float(np.mean(sigmas ** 2)))
    result: dict[str, float | int | None] = {
        f"{prefix}_n": int(errors.size), f"{prefix}_rmse": sqrt(float(np.mean(errors ** 2))), f"{prefix}_mean_sigma": float(np.mean(sigmas)),
        f"{prefix}_coverage_1sigma": float(np.mean(absolute <= sigmas)), f"{prefix}_coverage_2sigma": float(np.mean(absolute <= 2 * sigmas)), f"{prefix}_coverage_3sigma": float(np.mean(absolute <= 3 * sigmas)),
        f"{prefix}_calibration_ratio": sqrt(float(np.mean(errors ** 2))) / rms_sigma if rms_sigma else None,
        f"{prefix}_pearson": None, f"{prefix}_spearman": None,
    }
    if errors.size >= 2 and np.std(sigmas) > 0 and np.std(absolute) > 0:
        result[f"{prefix}_pearson"] = float(np.corrcoef(sigmas, absolute)[0, 1])
        ranked_sigmas, ranked_errors = _rank(sigmas), _rank(absolute)
        if np.std(ranked_sigmas) > 0 and np.std(ranked_errors) > 0:
            result[f"{prefix}_spearman"] = float(np.corrcoef(ranked_sigmas, ranked_errors)[0, 1])
    return result


def evaluate_records(records: Iterable[UncertaintyEvaluationRecord]) -> dict[str, float | int | None]:
    records = list(records)
    d_errors, d_sigmas = _paired(records, "disparity_error_px", "estimated_sigma_d_px")
    z_errors, z_sigmas = _paired(records, "z_error_mm", "estimated_sigma_z_mm")
    metrics = _metrics(d_errors, d_sigmas, "disparity"); metrics.update(_metrics(z_errors, z_sigmas, "z"))
    return metrics


@dataclass(frozen=True)
class UncertaintyCalibrationModel:
    scale_factor_d: float
    scale_factor_z: float
    dataset_id: str
    fit_count: int
    created_at: str
    method: str = "rms_scale"
    calibration_status: str = "CALIBRATED"

    def calibrate_d(self, sigma_raw: float | None) -> float | None:
        return None if sigma_raw is None else self.scale_factor_d * sigma_raw

    def calibrate_z(self, sigma_raw: float | None) -> float | None:
        return None if sigma_raw is None else self.scale_factor_z * sigma_raw


def fit_scale_calibration(records: Iterable[UncertaintyEvaluationRecord], *, dataset_id: str) -> UncertaintyCalibrationModel:
    records = list(records); d_errors, d_sigmas = _paired(records, "disparity_error_px", "estimated_sigma_d_px"); z_errors, z_sigmas = _paired(records, "z_error_mm", "estimated_sigma_z_mm")
    if not d_errors.size or not z_errors.size:
        raise ValueError("calibration needs independent disparity and Z error/sigma pairs")
    return UncertaintyCalibrationModel(
        scale_factor_d=sqrt(float(np.mean(d_errors ** 2)) / float(np.mean(d_sigmas ** 2))),
        scale_factor_z=sqrt(float(np.mean(z_errors ** 2)) / float(np.mean(z_sigmas ** 2))),
        dataset_id=dataset_id, fit_count=len(records), created_at=datetime.now(timezone.utc).isoformat(),
    )
