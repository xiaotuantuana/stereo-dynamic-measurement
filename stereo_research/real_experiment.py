"""Pre-flight tooling for real E0/E1 experiments and labelled synthetic dry-runs."""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .calibration import StereoCalibration
from .evaluation_runner import evaluate_e0_static
from .models import MatcherConfig


REAL_DATA_REQUIRED = "REAL_DATA_REQUIRED"
REAL_EVALUATION_DATA_REQUIRED = "REAL_EVALUATION_DATA_REQUIRED"
SYNTHETIC_DRY_RUN = "SYNTHETIC_DRY_RUN"
VALID_SPLITS = {"calibration", "evaluation"}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    status: str
    issues: tuple[ValidationIssue, ...]
    alignment_success_rate: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "alignment_success_rate": self.alignment_success_rate, "issues": [asdict(issue) for issue in self.issues]}


@dataclass(frozen=True)
class WorkflowResult:
    status: str
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    message: str = ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return path


def _resolve(base: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def load_real_experiment_manifest(path: str | Path, experiment_type: str | None = None) -> dict[str, object]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = str(payload.get("experiment_type", "")).upper()
    expected = actual if experiment_type is None else experiment_type.upper()
    if actual not in {"E0", "E1"} or actual != expected:
        raise ValueError(f"manifest experiment_type must be {expected}")
    if payload.get("gt_unit") != "mm":
        raise ValueError("manifest gt_unit must explicitly be mm")
    key = "sequences" if actual == "E0" else "conditions"
    entries = payload.get(key)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{actual} manifest requires a non-empty {key} list")
    required = (
        {"sequence_id", "split", "measurement_csv", "gt_csv", "alignment_mode", "max_time_offset_ms", "calibration_id", "distance_m", "baseline_mm", "method", "repeat", "notes"}
        if actual == "E0"
        else {"sequence_id", "split", "distance_m", "baseline_mm", "target_sigma_z_mm", "method", "repeat", "calibration_id", "calibration", "measurement", "gt"}
    )
    split_by_sequence: dict[str, set[str]] = defaultdict(set)
    for index, entry in enumerate(entries):
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(f"{actual} entry {index} missing fields: {', '.join(missing)}")
        split = str(entry["split"])
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be calibration or evaluation, got {split!r}")
        split_by_sequence[str(entry["sequence_id"])].add(split)
        alignment = str(entry.get("alignment_mode", "frame"))
        if alignment not in {"frame", "timestamp"}:
            raise ValueError("alignment_mode must be frame or timestamp")
    overlap = sorted(sequence for sequence, splits in split_by_sequence.items() if len(splits) > 1)
    if overlap:
        raise ValueError(f"sequence_id occurs in both calibration and evaluation split: {', '.join(overlap)}")
    payload["_manifest_path"] = str(manifest_path)
    return payload


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(str(value)))
    except (TypeError, ValueError):
        return False


def _validate_gt(rows: list[dict[str, str]], alignment_mode: str, label: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = {"point_id", "gt_x_mm", "gt_y_mm", "gt_z_mm"}
    header = set(rows[0]) if rows else set()
    if not rows:
        return [ValidationIssue("FAIL", "GT_EMPTY", f"{label}: GT has no records")]
    missing = required - header
    if missing:
        return [ValidationIssue("FAIL", "GT_FIELDS", f"{label}: missing GT fields {sorted(missing)}")]
    if alignment_mode not in header:
        column = "timestamp_s" if alignment_mode == "timestamp" else "frame"
        if column not in header:
            issues.append(ValidationIssue("FAIL", "GT_ALIGNMENT_FIELD", f"{label}: missing {column}"))
    frame_keys: set[tuple[int, str]] = set()
    time_keys: set[tuple[float, str]] = set()
    last_time: dict[str, float] = {}
    for index, row in enumerate(rows, start=2):
        point = str(row.get("point_id", "")).strip()
        if not point:
            issues.append(ValidationIssue("FAIL", "GT_POINT_ID", f"{label}:{index}: point_id is empty"))
        if any(not _finite(row.get(name)) for name in ("gt_x_mm", "gt_y_mm", "gt_z_mm")):
            issues.append(ValidationIssue("FAIL", "GT_XYZ", f"{label}:{index}: GT XYZ must be finite mm"))
        if row.get("frame", "") != "":
            try:
                raw_frame = float(row["frame"]); frame = int(raw_frame)
                if frame < 0 or raw_frame != frame:
                    raise ValueError
                key = (frame, point)
                if key in frame_keys:
                    issues.append(ValidationIssue("FAIL", "GT_DUPLICATE_FRAME", f"{label}:{index}: duplicate frame + point"))
                frame_keys.add(key)
            except ValueError:
                issues.append(ValidationIssue("FAIL", "GT_FRAME", f"{label}:{index}: frame must be a non-negative integer"))
        if row.get("timestamp_s", "") != "":
            if not _finite(row["timestamp_s"]):
                issues.append(ValidationIssue("FAIL", "GT_TIMESTAMP", f"{label}:{index}: timestamp must be finite"))
            else:
                timestamp = float(row["timestamp_s"]); key = (timestamp, point)
                if key in time_keys:
                    issues.append(ValidationIssue("FAIL", "GT_DUPLICATE_TIMESTAMP", f"{label}:{index}: duplicate timestamp + point"))
                if point in last_time and timestamp <= last_time[point]:
                    issues.append(ValidationIssue("FAIL", "GT_TIMESTAMP_ORDER", f"{label}:{index}: timestamp must be strictly increasing per point"))
                time_keys.add(key); last_time[point] = timestamp
    return issues


def _alignment_rate(measurement: list[dict[str, str]], gt: list[dict[str, str]], mode: str, max_offset_ms: float) -> float | None:
    if not measurement:
        return None
    matched = 0
    if mode == "frame":
        keys = {(row.get("frame", ""), row.get("point_id", "")) for row in gt}
        matched = sum((row.get("frame", ""), row.get("point_id", "")) in keys for row in measurement)
    else:
        by_point: dict[str, list[float]] = defaultdict(list)
        for row in gt:
            if _finite(row.get("timestamp_s")):
                by_point[row.get("point_id", "")].append(float(row["timestamp_s"]))
        for row in measurement:
            value = row.get("frame_timestamp_s", row.get("timestamp_s", ""))
            if _finite(value) and by_point[row.get("point_id", "")]:
                offset = min(abs(float(value) - candidate) for candidate in by_point[row.get("point_id", "")]) * 1000.0
                matched += offset <= max_offset_ms
    return matched / len(measurement)


def _calibration_baseline_mm(path: Path) -> float:
    calibration = StereoCalibration.from_json(path)
    tx = abs(float(np.asarray(calibration.translation).reshape(-1)[0]))
    return tx if calibration.unit == "mm" else tx * 1000.0


def validate_real_experiment(path: str | Path, *, baseline_tolerance_mm: float = 1.0) -> ValidationResult:
    try:
        payload = load_real_experiment_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return ValidationResult("FAIL", (ValidationIssue("FAIL", "MANIFEST", str(error)),))
    manifest_path = Path(str(payload["_manifest_path"])); base = manifest_path.parent
    actual = str(payload["experiment_type"]); entries = payload["sequences" if actual == "E0" else "conditions"]
    issues: list[ValidationIssue] = []; rates: list[float] = []
    for entry in entries:
        label = str(entry["sequence_id"])
        measurement = _resolve(base, entry["measurement_csv" if actual == "E0" else "measurement"])
        gt = _resolve(base, entry["gt_csv" if actual == "E0" else "gt"])
        if measurement is None or not measurement.exists():
            issues.append(ValidationIssue("FAIL", "MEASUREMENT_FILE", f"{label}: measurement CSV does not exist")); continue
        measurement_rows = _read_csv(measurement)
        if not measurement_rows:
            issues.append(ValidationIssue("FAIL", "MEASUREMENT_EMPTY", f"{label}: measurement CSV is empty")); continue
        if actual == "E1":
            calibration = _resolve(base, entry["calibration"])
            if calibration is None or not calibration.exists():
                issues.append(ValidationIssue("FAIL", "CALIBRATION_FILE", f"{label}: calibration does not exist"))
            else:
                baseline = _calibration_baseline_mm(calibration)
                if abs(float(entry["baseline_mm"]) - baseline) > baseline_tolerance_mm:
                    issues.append(ValidationIssue("FAIL", "BASELINE_IDENTITY", f"{label}: manifest baseline={entry['baseline_mm']} mm, calibration |T_x|={baseline:.6f} mm"))
        if gt is None or not gt.exists():
            issues.append(ValidationIssue("WARNING", "REAL_DATA_REQUIRED", f"{label}: independent GT is missing")); continue
        gt_rows = _read_csv(gt); mode = str(entry.get("alignment_mode", "frame"))
        issues.extend(_validate_gt(gt_rows, mode, label))
        rate = _alignment_rate(measurement_rows, gt_rows, mode, float(entry.get("max_time_offset_ms", 20.0)))
        if rate is not None:
            rates.append(rate)
            if rate == 0:
                issues.append(ValidationIssue("FAIL", "ALIGNMENT", f"{label}: no measurement/GT records align"))
            elif rate < 1:
                issues.append(ValidationIssue("WARNING", "ALIGNMENT_PARTIAL", f"{label}: alignment success rate={rate:.3f}"))
    status = "FAIL" if any(issue.level == "FAIL" for issue in issues) else "WARNING" if issues else "PASS"
    return ValidationResult(status, tuple(issues), None if not rates else float(np.mean(rates)))


E0_PER_FIELDS = ["data_source", "status", "data_split", "evaluated_xyz_stage", "sequence_id", "split", "distance_m", "baseline_mm", "method", "repeat", "N", "X_Bias_mm", "Y_Bias_mm", "Z_Bias_mm", "X_MAE_mm", "Y_MAE_mm", "Z_MAE_mm", "X_RMSE_mm", "Y_RMSE_mm", "Z_RMSE_mm", "XYZ_RMSE_mm", "X_Std_mm", "Y_Std_mm", "Z_Std_mm", "Max_Error_mm", "Outlier_Rate", "Mean_sigma_d_px", "Mean_sigma_Z_mm", "Coverage_1sigma", "Coverage_2sigma", "Coverage_3sigma", "Calibration_Ratio", "Spearman", "alignment_success_rate", "Valid_Rate"]
UNCERTAINTY_FIELDS = ["data_source", "status", "Sequence", "Distance", "N", "RMSE_d_px", "Mean_sigma_d_px", "Coverage_1sigma", "Coverage_2sigma", "Coverage_3sigma", "Calibration_Ratio", "Spearman"]
E1_FIELDS = ["data_source", "status", "data_split", "evaluated_xyz_stage", "sequence_id", "split", "distance_m", "baseline_mm", "target_sigma_z_mm", "method", "repeat", "N", "Z_RMSE_mm", "XYZ_RMSE_mm", "Estimated_sigma_Z_mm", "Required_sigma_d_px", "Estimated_sigma_d_px", "Precision_Status_Distribution", "Target_Attainment", "Refinement_Level", "Retry_Count", "Runtime_ms", "Valid_Rate", "Coverage_1sigma", "Coverage_2sigma", "Coverage_3sigma", "alignment_success_rate"]
E0_SUMMARY_FIELDS = ["data_source", "status", "data_split", "evaluated_xyz_stage", "distance_m", "baseline_mm", "method", "sequence_count", "sample_count", "X_bias_mean", "Y_bias_mean", "Z_bias_mean", "X_MAE_mean", "Y_MAE_mean", "Z_MAE_mean", "X_RMSE_mean", "Y_RMSE_mean", "Z_RMSE_mean", "XYZ_RMSE_mean", "X_RMSE_std", "Y_RMSE_std", "Z_RMSE_std", "XYZ_RMSE_std", "valid_rate_mean", "coverage_1sigma_mean", "coverage_2sigma_mean", "coverage_3sigma_mean", "calibration_ratio_mean", "Spearman_mean", "alignment_success_rate_mean"]
E1_SUMMARY_FIELDS = ["data_source", "status", "data_split", "evaluated_xyz_stage", "distance_m", "baseline_mm", "target_sigma_z_mm", "method", "repeat_count", "sample_count", "Z_RMSE_mean", "Z_RMSE_std", "XYZ_RMSE_mean", "XYZ_RMSE_std", "estimated_sigma_Z_mean", "target_attainment_mean", "valid_rate_mean", "runtime_mean", "runtime_std", "retry_count_mean", "refinement_level_mean", "coverage_1sigma_mean", "coverage_2sigma_mean", "coverage_3sigma_mean", "Precision_Status_Distribution", "MET_rate", "VALID_BUT_PRECISION_UNMET_rate", "INFEASIBLE_rate", "WARMUP_rate"]


def _values(rows: list[dict[str, str]], field_name: str) -> np.ndarray:
    return np.asarray([float(row[field_name]) for row in rows if _finite(row.get(field_name))], dtype=float)


def _mean_measurement(rows: list[dict[str, str]], *keys: str) -> float | None:
    for key in keys:
        values = _values(rows, key)
        if values.size:
            return float(np.mean(values))
    return None


def _aggregate_value(
    rows: list[dict[str, object]], field_name: str, operation: str = "mean"
) -> float | None:
    values = np.asarray(
        [float(row[field_name]) for row in rows if _finite(row.get(field_name))],
        dtype=float,
    )
    if not values.size:
        return None
    return float(np.mean(values) if operation == "mean" else np.std(values))


def _group_rows(
    rows: list[dict[str, object]], fields: tuple[str, ...]
) -> list[list[dict[str, object]]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    return list(groups.values())


def _aggregate_e0(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for group in _group_rows(rows, ("distance_m", "baseline_mm", "method")):
        first = group[0]
        output.append({
            "data_source": first["data_source"], "status": first["status"],
            "data_split": "evaluation", "evaluated_xyz_stage": "final",
            "distance_m": first["distance_m"], "baseline_mm": first["baseline_mm"],
            "method": first["method"], "sequence_count": len(group),
            "sample_count": int(sum(float(row["N"]) for row in group if _finite(row.get("N")))),
            "X_bias_mean": _aggregate_value(group, "X_Bias_mm"),
            "Y_bias_mean": _aggregate_value(group, "Y_Bias_mm"),
            "Z_bias_mean": _aggregate_value(group, "Z_Bias_mm"),
            "X_MAE_mean": _aggregate_value(group, "X_MAE_mm"),
            "Y_MAE_mean": _aggregate_value(group, "Y_MAE_mm"),
            "Z_MAE_mean": _aggregate_value(group, "Z_MAE_mm"),
            "X_RMSE_mean": _aggregate_value(group, "X_RMSE_mm"),
            "Y_RMSE_mean": _aggregate_value(group, "Y_RMSE_mm"),
            "Z_RMSE_mean": _aggregate_value(group, "Z_RMSE_mm"),
            "XYZ_RMSE_mean": _aggregate_value(group, "XYZ_RMSE_mm"),
            "X_RMSE_std": _aggregate_value(group, "X_RMSE_mm", "std"),
            "Y_RMSE_std": _aggregate_value(group, "Y_RMSE_mm", "std"),
            "Z_RMSE_std": _aggregate_value(group, "Z_RMSE_mm", "std"),
            "XYZ_RMSE_std": _aggregate_value(group, "XYZ_RMSE_mm", "std"),
            "valid_rate_mean": _aggregate_value(group, "Valid_Rate"),
            "coverage_1sigma_mean": _aggregate_value(group, "Coverage_1sigma"),
            "coverage_2sigma_mean": _aggregate_value(group, "Coverage_2sigma"),
            "coverage_3sigma_mean": _aggregate_value(group, "Coverage_3sigma"),
            "calibration_ratio_mean": _aggregate_value(group, "Calibration_Ratio"),
            "Spearman_mean": _aggregate_value(group, "Spearman"),
            "alignment_success_rate_mean": _aggregate_value(group, "alignment_success_rate"),
        })
    return output


def _aggregate_e1(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    keys = ("distance_m", "baseline_mm", "target_sigma_z_mm", "method")
    for group in _group_rows(rows, keys):
        first = group[0]
        counts: Counter[str] = Counter()
        for row in group:
            raw = row.get("Precision_Status_Distribution")
            if raw:
                counts.update(json.loads(str(raw)))
        total = sum(counts.values())
        rate = lambda name: None if total == 0 else counts[name] / total
        output.append({
            "data_source": first["data_source"], "status": first["status"],
            "data_split": "evaluation", "evaluated_xyz_stage": "final",
            "distance_m": first["distance_m"], "baseline_mm": first["baseline_mm"],
            "target_sigma_z_mm": first["target_sigma_z_mm"], "method": first["method"],
            "repeat_count": len(group),
            "sample_count": int(sum(float(row["N"]) for row in group if _finite(row.get("N")))),
            "Z_RMSE_mean": _aggregate_value(group, "Z_RMSE_mm"),
            "Z_RMSE_std": _aggregate_value(group, "Z_RMSE_mm", "std"),
            "XYZ_RMSE_mean": _aggregate_value(group, "XYZ_RMSE_mm"),
            "XYZ_RMSE_std": _aggregate_value(group, "XYZ_RMSE_mm", "std"),
            "estimated_sigma_Z_mean": _aggregate_value(group, "Estimated_sigma_Z_mm"),
            "target_attainment_mean": _aggregate_value(group, "Target_Attainment"),
            "valid_rate_mean": _aggregate_value(group, "Valid_Rate"),
            "runtime_mean": _aggregate_value(group, "Runtime_ms"),
            "runtime_std": _aggregate_value(group, "Runtime_ms", "std"),
            "retry_count_mean": _aggregate_value(group, "Retry_Count"),
            "refinement_level_mean": _aggregate_value(group, "Refinement_Level"),
            "coverage_1sigma_mean": _aggregate_value(group, "Coverage_1sigma"),
            "coverage_2sigma_mean": _aggregate_value(group, "Coverage_2sigma"),
            "coverage_3sigma_mean": _aggregate_value(group, "Coverage_3sigma"),
            "Precision_Status_Distribution": json.dumps(counts, ensure_ascii=False, sort_keys=True),
            "MET_rate": rate("MET"),
            "VALID_BUT_PRECISION_UNMET_rate": rate("VALID_BUT_PRECISION_UNMET"),
            "INFEASIBLE_rate": rate("INFEASIBLE"),
            "WARMUP_rate": rate("WARMUP"),
        })
    return output


def _sequence_metrics(entry: Mapping[str, object], measurement_path: Path, gt_path: Path, temp_output: Path, data_source: str) -> tuple[dict[str, object], dict[str, object]]:
    result = evaluate_e0_static(measurement_path, gt_path, temp_output, alignment_mode=str(entry.get("alignment_mode", "frame")), max_time_offset_ms=float(entry.get("max_time_offset_ms", 20.0)))
    if result.status != "OK":
        raise ValueError(f"{result.status}: {result.message}")
    records = _read_csv(result.records_path) if result.records_path and result.records_path.exists() else []
    if data_source == SYNTHETIC_DRY_RUN:
        for record in records:
            record["data_source"] = SYNTHETIC_DRY_RUN
        if result.records_path and records:
            _write_csv(result.records_path, ["data_source", *[key for key in records[0] if key != "data_source"]], records)
        if result.report_path and result.report_path.exists():
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            report["data_source"] = SYNTHETIC_DRY_RUN
            result.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    measurement = _read_csv(measurement_path)
    axis_errors = {axis: _values(records, f"{axis}_error_mm") for axis in "xyz"}
    xyz_rows = [row for row in records if all(_finite(row.get(f"{axis}_error_mm")) for axis in "xyz")]
    xyz_norm = np.asarray([math.sqrt(sum(float(row[f"{axis}_error_mm"]) ** 2 for axis in "xyz")) for row in xyz_rows], dtype=float)
    disparity_errors, sigma_d = _values(records, "disparity_error_px"), _values(records, "estimated_sigma_d_px")
    z_errors, sigma_z = _values(records, "z_error_mm"), _values(records, "estimated_sigma_z_mm")
    valid_rate = sum(row.get("status", "valid") == "valid" for row in measurement) / len(measurement) if measurement else None
    def coverage(n: int) -> float | None:
        pairs = [(abs(float(row["z_error_mm"])), float(row["estimated_sigma_z_mm"])) for row in records if _finite(row.get("z_error_mm")) and _finite(row.get("estimated_sigma_z_mm")) and float(row["estimated_sigma_z_mm"]) > 0]
        return None if not pairs else float(np.mean([error <= n * sigma for error, sigma in pairs]))
    def axis_stat(axis: str, op: str) -> float | None:
        values = axis_errors[axis]
        if not values.size: return None
        return {"bias": lambda: np.mean(values), "mae": lambda: np.mean(abs(values)), "rmse": lambda: np.sqrt(np.mean(values ** 2)), "std": lambda: np.std(values)}[op]().item()
    cal_ratio = None if not z_errors.size or not sigma_z.size or np.mean(sigma_z ** 2) == 0 else float(np.sqrt(np.mean(z_errors ** 2) / np.mean(sigma_z ** 2)))
    spear = None
    if len(z_errors) >= 2 and len(sigma_z) == len(z_errors) and np.std(z_errors) > 0 and np.std(sigma_z) > 0:
        spear = float(np.corrcoef(np.argsort(np.argsort(sigma_z)), np.argsort(np.argsort(abs(z_errors))))[0, 1])
    row = {"data_source": data_source, "status": SYNTHETIC_DRY_RUN if data_source == SYNTHETIC_DRY_RUN else "OK", "data_split": "evaluation", "evaluated_xyz_stage": "final", "sequence_id": entry["sequence_id"], "split": entry["split"], "distance_m": entry["distance_m"], "baseline_mm": entry["baseline_mm"], "method": entry["method"], "repeat": entry["repeat"], "N": len(records), "X_Bias_mm": axis_stat("x", "bias"), "Y_Bias_mm": axis_stat("y", "bias"), "Z_Bias_mm": axis_stat("z", "bias"), "X_MAE_mm": axis_stat("x", "mae"), "Y_MAE_mm": axis_stat("y", "mae"), "Z_MAE_mm": axis_stat("z", "mae"), "X_RMSE_mm": axis_stat("x", "rmse"), "Y_RMSE_mm": axis_stat("y", "rmse"), "Z_RMSE_mm": axis_stat("z", "rmse"), "XYZ_RMSE_mm": None if not xyz_norm.size else float(np.sqrt(np.mean(xyz_norm ** 2))), "X_Std_mm": axis_stat("x", "std"), "Y_Std_mm": axis_stat("y", "std"), "Z_Std_mm": axis_stat("z", "std"), "Max_Error_mm": None if not xyz_norm.size else float(np.max(xyz_norm)), "Outlier_Rate": None if not xyz_norm.size else float(np.mean(xyz_norm > 3 * np.std(xyz_norm))) if np.std(xyz_norm) else 0.0, "Mean_sigma_d_px": None if not sigma_d.size else float(np.mean(sigma_d)), "Mean_sigma_Z_mm": None if not sigma_z.size else float(np.mean(sigma_z)), "Coverage_1sigma": coverage(1), "Coverage_2sigma": coverage(2), "Coverage_3sigma": coverage(3), "Calibration_Ratio": cal_ratio, "Spearman": spear, "alignment_success_rate": len(records) / len(measurement) if measurement else None, "Valid_Rate": valid_rate}
    uncertainty = {"data_source": data_source, "status": row["status"], "Sequence": entry["sequence_id"], "Distance": entry["distance_m"], "N": len(records), "RMSE_d_px": None if not disparity_errors.size else float(np.sqrt(np.mean(disparity_errors ** 2))), "Mean_sigma_d_px": row["Mean_sigma_d_px"], "Coverage_1sigma": None if not disparity_errors.size or len(disparity_errors) != len(sigma_d) else float(np.mean(abs(disparity_errors) <= sigma_d)), "Coverage_2sigma": None if not disparity_errors.size or len(disparity_errors) != len(sigma_d) else float(np.mean(abs(disparity_errors) <= 2 * sigma_d)), "Coverage_3sigma": None if not disparity_errors.size or len(disparity_errors) != len(sigma_d) else float(np.mean(abs(disparity_errors) <= 3 * sigma_d)), "Calibration_Ratio": None if not disparity_errors.size or not sigma_d.size or np.mean(sigma_d ** 2) == 0 else float(np.sqrt(np.mean(disparity_errors ** 2) / np.mean(sigma_d ** 2))), "Spearman": None}
    return row, uncertainty


def _empty_real_row(entry: Mapping[str, object], fields: list[str], data_source: str) -> dict[str, object]:
    row = {field: "" for field in fields}; row.update({"data_source": data_source, "status": REAL_DATA_REQUIRED, "data_split": "evaluation", "evaluated_xyz_stage": "final", "sequence_id": entry["sequence_id"], "split": entry["split"], "distance_m": entry["distance_m"], "baseline_mm": entry["baseline_mm"], "method": entry["method"], "repeat": entry["repeat"]})
    return row


def _paper_tables(output: Path, summary_rows: list[dict[str, object]], data_source: str) -> list[Path]:
    table_e0_fields = ["data_source", "status", "data_split", "evaluated_xyz_stage", "Distance", "Baseline", "Method", "sequence_count", "sample_count", "X_RMSE_mean", "Y_RMSE_mean", "Z_RMSE_mean", "XYZ_RMSE_mean", "Z_bias_mean", "valid_rate_mean"]
    table_e0 = [{"data_source": data_source, "status": row["status"], "data_split": "evaluation", "evaluated_xyz_stage": "final", "Distance": row["distance_m"], "Baseline": row["baseline_mm"], "Method": row["method"], **{key: row.get(key, "") for key in table_e0_fields[7:]}} for row in summary_rows]
    uncertainty_fields = ["data_source", "status", "data_split", "evaluated_xyz_stage", "Distance", "Method", "sample_count", "coverage_1sigma_mean", "coverage_2sigma_mean", "coverage_3sigma_mean", "calibration_ratio_mean", "Spearman_mean"]
    uncertainty = [{"data_source": data_source, "status": row["status"], "data_split": "evaluation", "evaluated_xyz_stage": "final", "Distance": row["distance_m"], "Method": row["method"], **{key: row.get(key, "") for key in uncertainty_fields[6:]}} for row in summary_rows]
    refinement_fields = ["data_source", "status", "data_split", "evaluated_xyz_stage", "Method", "Refinement_Level", "Retry_Mode", "Disparity_RMSE", "Z_RMSE", "Runtime", "Improvement_Rate"]
    template = {"data_source": data_source, "status": "TEMPLATE_ONLY", "data_split": "evaluation", "evaluated_xyz_stage": "final"}
    return [
        _write_csv(output / "table_e0_static_accuracy.csv", table_e0_fields, table_e0),
        _write_csv(output / "table_uncertainty_calibration.csv", uncertainty_fields, uncertainty),
        _write_csv(output / "table_e1_distance_baseline.csv", ["data_source", "status", "data_split", "evaluated_xyz_stage", "Distance", "Baseline", "Target_sigma_Z", "Method", "Z_RMSE", "Estimated_sigma_Z", "Target_Attainment", "Retry_Count", "Runtime", "Valid_Rate"], [template]),
        _write_csv(output / "table_refinement_retry.csv", refinement_fields, [template]),
    ]


def evaluate_e0_manifest(manifest_path: str | Path, output_dir: str | Path) -> WorkflowResult:
    payload = load_real_experiment_manifest(manifest_path, "E0"); base = Path(str(payload["_manifest_path"])).parent; output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    validation = validate_real_experiment(manifest_path)
    if validation.status == "FAIL":
        raise ValueError(next(issue.message for issue in validation.issues if issue.level == "FAIL"))
    data_source = str(payload.get("data_source", "REAL_E0")); rows: list[dict[str, object]] = []; uncertainty: list[dict[str, object]] = []; missing_real = False
    evaluation_entries = [
        entry for entry in payload["sequences"] if entry["split"] == "evaluation"
    ]
    for entry in evaluation_entries:
        measurement = _resolve(base, entry["measurement_csv"]); gt = _resolve(base, entry["gt_csv"])
        if measurement is None or not measurement.exists():
            raise FileNotFoundError(f"measurement CSV missing for {entry['sequence_id']}")
        if gt is None or not gt.exists():
            missing_real = True; rows.append(_empty_real_row(entry, E0_PER_FIELDS, data_source)); uncertainty.append({field: "" for field in UNCERTAINTY_FIELDS} | {"data_source": data_source, "status": REAL_DATA_REQUIRED, "Sequence": entry["sequence_id"], "Distance": entry["distance_m"]}); continue
        row, unc = _sequence_metrics(entry, measurement, gt, output / "evaluation" / str(entry["sequence_id"]), data_source); rows.append(row); uncertainty.append(unc)
    summary_rows = _aggregate_e0(rows)
    paths = [_write_csv(output / "e0_per_sequence.csv", E0_PER_FIELDS, rows), _write_csv(output / "e0_summary.csv", E0_SUMMARY_FIELDS, summary_rows), _write_csv(output / "uncertainty_calibration.csv", UNCERTAINTY_FIELDS, uncertainty)]
    paths.extend(_paper_tables(output, summary_rows, data_source))
    metadata_path = output / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps({
        "data_source": data_source,
        "formal_results_split": "evaluation",
        "evaluated_xyz_stage": "final",
        "num_calibration_sequences": sum(entry["split"] == "calibration" for entry in payload["sequences"]),
        "num_evaluation_sequences": len(evaluation_entries),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(metadata_path)
    status = (
        REAL_EVALUATION_DATA_REQUIRED
        if not evaluation_entries
        else SYNTHETIC_DRY_RUN
        if data_source == SYNTHETIC_DRY_RUN
        else REAL_DATA_REQUIRED
        if missing_real
        else "OK"
    )
    return WorkflowResult(status, tuple(paths))


def evaluate_e1_manifest(manifest_path: str | Path, output_dir: str | Path, *, baseline_tolerance_mm: float = 1.0) -> WorkflowResult:
    payload = load_real_experiment_manifest(manifest_path, "E1"); base = Path(str(payload["_manifest_path"])).parent; output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); data_source = str(payload.get("data_source", "REAL_E1"))
    validation = validate_real_experiment(manifest_path, baseline_tolerance_mm=baseline_tolerance_mm)
    failures = [issue for issue in validation.issues if issue.level == "FAIL"]
    if failures: raise ValueError(failures[0].message)
    rows: list[dict[str, object]] = []; missing_real = False
    evaluation_entries = [
        entry for entry in payload["conditions"] if entry["split"] == "evaluation"
    ]
    for index, entry in enumerate(evaluation_entries):
        measurement_path = _resolve(base, entry["measurement"]); gt_path = _resolve(base, entry["gt"])
        if measurement_path is None or not measurement_path.exists(): raise FileNotFoundError(f"measurement CSV missing for {entry['sequence_id']}")
        measurements = _read_csv(measurement_path)
        if gt_path is None or not gt_path.exists():
            missing_real = True; rows.append(_empty_real_row(entry, E1_FIELDS, data_source) | {"target_sigma_z_mm": entry["target_sigma_z_mm"]}); continue
        e0_entry = {"sequence_id": entry["sequence_id"], "split": entry["split"], "distance_m": entry["distance_m"], "baseline_mm": entry["baseline_mm"], "method": entry["method"], "repeat": entry["repeat"], "alignment_mode": entry.get("alignment_mode", "frame"), "max_time_offset_ms": entry.get("max_time_offset_ms", 20)}
        base_row, _ = _sequence_metrics(e0_entry, measurement_path, gt_path, output / "evaluation" / f"condition_{index}", data_source)
        status_distribution = Counter(row.get("precision_status", "") for row in measurements if row.get("precision_status", ""))
        records_path = output / "evaluation" / f"condition_{index}" / "e0_static_accuracy.csv"; records = _read_csv(records_path) if records_path.exists() else []
        target = float(entry["target_sigma_z_mm"]); actual = _values(records, "z_error_mm")
        row = {"data_source": data_source, "status": SYNTHETIC_DRY_RUN if data_source == SYNTHETIC_DRY_RUN else "OK", "data_split": "evaluation", "evaluated_xyz_stage": "final", "sequence_id": entry["sequence_id"], "split": entry["split"], "distance_m": entry["distance_m"], "baseline_mm": entry["baseline_mm"], "target_sigma_z_mm": target, "method": entry["method"], "repeat": entry["repeat"], "N": base_row["N"], "Z_RMSE_mm": base_row["Z_RMSE_mm"], "XYZ_RMSE_mm": base_row["XYZ_RMSE_mm"], "Estimated_sigma_Z_mm": _mean_measurement(measurements, "estimated_sigma_z_mm"), "Required_sigma_d_px": _mean_measurement(measurements, "required_sigma_d_px"), "Estimated_sigma_d_px": _mean_measurement(measurements, "estimated_sigma_d_px"), "Precision_Status_Distribution": json.dumps(status_distribution, ensure_ascii=False, sort_keys=True), "Target_Attainment": None if not actual.size else float(np.mean(abs(actual) <= target)), "Refinement_Level": _mean_measurement(measurements, "policy_refinement_level"), "Retry_Count": _mean_measurement(measurements, "policy_precision_retry_count"), "Runtime_ms": _mean_measurement(measurements, "frame_total_ms", "total_ms"), "Valid_Rate": base_row["Valid_Rate"], "Coverage_1sigma": base_row["Coverage_1sigma"], "Coverage_2sigma": base_row["Coverage_2sigma"], "Coverage_3sigma": base_row["Coverage_3sigma"], "alignment_success_rate": base_row["alignment_success_rate"]}; rows.append(row)
    summary_rows = _aggregate_e1(rows)
    per_path = _write_csv(output / "e1_per_condition.csv", E1_FIELDS, rows); summary_path = _write_csv(output / "e1_summary.csv", E1_SUMMARY_FIELDS, summary_rows)
    paper_fields = ["data_source", "status", "data_split", "evaluated_xyz_stage", "Distance", "Baseline", "Target_sigma_Z", "Method", "Z_RMSE_mean", "XYZ_RMSE_mean", "Estimated_sigma_Z_mean", "Target_Attainment_mean", "Retry_Count_mean", "Runtime_mean", "Valid_Rate_mean"]
    paper = [{"data_source": data_source, "status": row["status"], "data_split": "evaluation", "evaluated_xyz_stage": "final", "Distance": row["distance_m"], "Baseline": row["baseline_mm"], "Target_sigma_Z": row["target_sigma_z_mm"], "Method": row["method"], "Z_RMSE_mean": row.get("Z_RMSE_mean", ""), "XYZ_RMSE_mean": row.get("XYZ_RMSE_mean", ""), "Estimated_sigma_Z_mean": row.get("estimated_sigma_Z_mean", ""), "Target_Attainment_mean": row.get("target_attainment_mean", ""), "Retry_Count_mean": row.get("retry_count_mean", ""), "Runtime_mean": row.get("runtime_mean", ""), "Valid_Rate_mean": row.get("valid_rate_mean", "")} for row in summary_rows]
    table_path = _write_csv(output / "table_e1_distance_baseline.csv", paper_fields, paper)
    metadata_path = output / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps({
        "data_source": data_source,
        "formal_results_split": "evaluation",
        "evaluated_xyz_stage": "final",
        "num_calibration_sequences": sum(entry["split"] == "calibration" for entry in payload["conditions"]),
        "num_evaluation_sequences": len(evaluation_entries),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    status = (
        REAL_EVALUATION_DATA_REQUIRED
        if not evaluation_entries
        else SYNTHETIC_DRY_RUN
        if data_source == SYNTHETIC_DRY_RUN
        else REAL_DATA_REQUIRED
        if missing_real
        else "OK"
    )
    return WorkflowResult(status, (per_path, summary_path, table_path, metadata_path))


def write_environment_snapshot(path: str | Path, *, experiment_id: str, sequence_id: str, method: str, resolution: str, fps: float, baseline_mm: float, calibration_id: str, calibration_file: str, target_sigma_z_mm: float | None, config: MatcherConfig | None = None, data_source: str = "REAL") -> Path:
    matcher = config or MatcherConfig()
    def git(*args: str) -> str:
        try: return subprocess.run(["git", *args], capture_output=True, text=True, check=False, timeout=5).stdout.strip()
        except OSError: return "unavailable"
    payload = {"data_source": data_source, "experiment_id": experiment_id, "sequence_id": sequence_id, "date": date.today().isoformat(), "method": method, "resolution": resolution, "fps": fps, "baseline_mm": baseline_mm, "calibration_id": calibration_id, "calibration_file": calibration_file, "target_sigma_z_mm": target_sigma_z_mm, "algorithm": {"enable_target_accuracy_policy": matcher.enable_target_accuracy_policy, "target_metric": matcher.target_metric, "enable_icgn": matcher.enable_icgn, "icgn_max_iterations": matcher.icgn_max_iterations, "icgn_epsilon": matcher.icgn_epsilon, "max_precision_retry": matcher.max_precision_retry, "uncertainty_base_disparity_variance_px2": matcher.uncertainty_base_disparity_variance_px2, "uncertainty_min_disparity_variance_px2": matcher.uncertainty_min_disparity_variance_px2, "uncertainty_max_disparity_variance_px2": matcher.uncertainty_max_disparity_variance_px2}, "git": {"revision": git("rev-parse", "HEAD"), "status": git("status", "--short")}}
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); return output


def create_synthetic_demo(root: str | Path) -> Path:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    measurement_fields = ["frame", "frame_timestamp_s", "point_id", "method", "status", "estimated_disparity", "estimated_sigma_d_px", "required_sigma_d_px", "estimated_X_m", "estimated_Y_m", "estimated_Z_m", "final_X_m", "final_Y_m", "final_Z_m", "estimated_sigma_z_mm", "precision_status", "policy_refinement_level", "policy_precision_retry_count", "frame_total_ms", "data_source"]
    measurements = [
        {"frame": 0, "frame_timestamp_s": 1.000, "point_id": "P1", "method": "M3", "status": "valid", "estimated_disparity": 48.10, "estimated_sigma_d_px": .10, "required_sigma_d_px": .08, "estimated_X_m": .010, "estimated_Y_m": .020, "estimated_Z_m": 2.001, "final_X_m": .010, "final_Y_m": .020, "final_Z_m": 2.001, "estimated_sigma_z_mm": 4.2, "precision_status": "MET", "policy_refinement_level": 1, "policy_precision_retry_count": 0, "frame_total_ms": 12.5, "data_source": SYNTHETIC_DRY_RUN},
        {"frame": 1, "frame_timestamp_s": 1.033, "point_id": "P1", "method": "M3", "status": "valid", "estimated_disparity": 47.95, "estimated_sigma_d_px": .12, "required_sigma_d_px": .08, "estimated_X_m": .011, "estimated_Y_m": .019, "estimated_Z_m": 2.004, "final_X_m": .011, "final_Y_m": .019, "final_Z_m": 2.004, "estimated_sigma_z_mm": 5.1, "precision_status": "VALID_BUT_PRECISION_UNMET", "policy_refinement_level": 2, "policy_precision_retry_count": 1, "frame_total_ms": 18.0, "data_source": SYNTHETIC_DRY_RUN},
    ]
    _write_csv(root / "measurement.csv", measurement_fields, measurements)
    gt_fields = ["frame", "timestamp_s", "point_id", "gt_x_mm", "gt_y_mm", "gt_z_mm", "gt_sigma_x_mm", "gt_sigma_y_mm", "gt_sigma_z_mm", "gt_disparity_px", "data_source"]
    _write_csv(root / "gt.csv", gt_fields, [{"frame": 0, "timestamp_s": 1.000, "point_id": "P1", "gt_x_mm": 10, "gt_y_mm": 20, "gt_z_mm": 2000, "gt_sigma_x_mm": .1, "gt_sigma_y_mm": .1, "gt_sigma_z_mm": .2, "gt_disparity_px": 48.0, "data_source": SYNTHETIC_DRY_RUN}, {"frame": 1, "timestamp_s": 1.033, "point_id": "P1", "gt_x_mm": 11, "gt_y_mm": 19, "gt_z_mm": 2000, "gt_sigma_x_mm": .1, "gt_sigma_y_mm": .1, "gt_sigma_z_mm": .2, "gt_disparity_px": 48.0, "data_source": SYNTHETIC_DRY_RUN}])
    calibration = StereoCalibration("synthetic_demo_120mm", (8, 6), np.eye(3), np.eye(3), np.zeros(5), np.zeros(5), np.eye(3), np.array([-120.0, 0.0, 0.0]), "mm"); calibration.to_json(root / "calibration.json")
    common = {"experiment_id": "synthetic-demo", "gt_unit": "mm", "data_source": SYNTHETIC_DRY_RUN}
    e0 = common | {"experiment_type": "E0", "sequences": [{"sequence_id": "SYN_E0_001", "split": "evaluation", "measurement_csv": "measurement.csv", "gt_csv": "gt.csv", "alignment_mode": "frame", "max_time_offset_ms": 20, "calibration_id": "synthetic_demo_120mm", "distance_m": 2.0, "baseline_mm": 120.0, "method": "M3", "repeat": 1, "notes": "SYNTHETIC EXAMPLE ONLY"}]}
    e1 = common | {"experiment_type": "E1", "conditions": [{"sequence_id": "SYN_E1_001", "split": "evaluation", "distance_m": 2.0, "baseline_mm": 120.0, "target_sigma_z_mm": 5.0, "method": "M3", "repeat": 1, "calibration_id": "synthetic_demo_120mm", "calibration": "calibration.json", "measurement": "measurement.csv", "gt": "gt.csv", "alignment_mode": "timestamp", "max_time_offset_ms": 20}]}
    (root / "e0_manifest.json").write_text(json.dumps(e0, ensure_ascii=False, indent=2), encoding="utf-8"); (root / "e1_manifest.json").write_text(json.dumps(e1, ensure_ascii=False, indent=2), encoding="utf-8")
    write_environment_snapshot(root / "experiment_environment.json", experiment_id="synthetic-demo", sequence_id="SYN_E0_001", method="M3", resolution="8x6", fps=30.0, baseline_mm=120.0, calibration_id="synthetic_demo_120mm", calibration_file="calibration.json", target_sigma_z_mm=5.0, data_source=SYNTHETIC_DRY_RUN)
    return root


def run_synthetic_dry_run(root: str | Path) -> WorkflowResult:
    root = Path(root); validation_paths: list[Path] = []
    for name in ("e0_manifest.json", "e1_manifest.json"):
        result = validate_real_experiment(root / name)
        path = root / "results" / f"validation_{name.replace('_manifest.json', '')}.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({"data_source": SYNTHETIC_DRY_RUN, **result.as_dict()}, ensure_ascii=False, indent=2), encoding="utf-8"); validation_paths.append(path)
        if result.status == "FAIL": return WorkflowResult("FAIL", tuple(validation_paths), f"synthetic validation failed for {name}")
    e0 = evaluate_e0_manifest(root / "e0_manifest.json", root / "results" / "E0"); e1 = evaluate_e1_manifest(root / "e1_manifest.json", root / "results" / "E1")
    return WorkflowResult(SYNTHETIC_DRY_RUN, tuple(validation_paths) + e0.output_paths + e1.output_paths)
