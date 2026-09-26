"""Offline E0/E1 evaluators.  They do not invoke the measurement pipeline."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .calibration import StereoCalibration
from .uncertainty_evaluation import GroundTruthRecord, UncertaintyEvaluationRecord, align_ground_truth, evaluate_records


REAL_DATA_REQUIRED = "REAL_DATA_REQUIRED"
FINAL_XYZ_UNAVAILABLE = "FINAL_XYZ_UNAVAILABLE"


@dataclass(frozen=True)
class EvaluationRunResult:
    status: str
    report_path: Path | None = None
    records_path: Path | None = None
    message: str = ""


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    data = list(rows)
    if not data:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(dict.fromkeys(key for row in data for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(data)


def evaluate_e0_static(
    measurement_csv: str | Path,
    gt_csv: str | Path | None,
    output_dir: str | Path,
    *,
    alignment_mode: str = "frame",
    max_time_offset_ms: float = 20.0,
    xyz_stage: str = "final",
) -> EvaluationRunResult:
    """Join separated inputs after measurement; absent real GT is a valid DATA_REQUIRED state."""
    measurement_path, output = Path(measurement_csv), Path(output_dir)
    if gt_csv is None or not Path(gt_csv).exists():
        return EvaluationRunResult(REAL_DATA_REQUIRED, message="independent E0 gt.csv is required for accuracy evaluation")
    measurement_rows = _rows(measurement_path)
    stage_fields = {
        "measured": ("measured_X_m", "measured_Y_m", "measured_Z_m"),
        "estimated": ("estimated_X_m", "estimated_Y_m", "estimated_Z_m"),
        "compensated": ("compensated_X_m", "compensated_Y_m", "compensated_Z_m"),
        "final": ("final_X_m", "final_Y_m", "final_Z_m"),
    }
    if xyz_stage not in stage_fields:
        raise ValueError("xyz_stage must be measured, estimated, compensated, or final")
    header = set(measurement_rows[0]) if measurement_rows else set()
    missing = [field for field in stage_fields[xyz_stage] if field not in header]
    if missing:
        status = FINAL_XYZ_UNAVAILABLE if xyz_stage == "final" else "XYZ_STAGE_UNAVAILABLE"
        return EvaluationRunResult(
            status,
            message=f"{xyz_stage} XYZ fields unavailable: {', '.join(missing)}",
        )
    output.mkdir(parents=True, exist_ok=True)
    gt_records = [GroundTruthRecord.from_row(row) for row in _rows(Path(gt_csv))]
    evaluation = []
    for row in measurement_rows:
        gt = align_ground_truth(row, gt_records, alignment_mode=alignment_mode, max_time_offset_ms=max_time_offset_ms)
        if gt is not None:
            evaluation.append(
                UncertaintyEvaluationRecord.from_measurement_and_gt(
                    row,
                    gt,
                    xyz_stage=xyz_stage,
                )
            )
    records_path = output / "e0_static_accuracy.csv"; _write_csv(records_path, [record.as_dict() for record in evaluation])
    metrics = evaluate_records(evaluation)
    xyz = {}
    for axis in "xyz":
        values = np.asarray([getattr(record, f"{axis}_error_mm") for record in evaluation if getattr(record, f"{axis}_error_mm") is not None], dtype=float)
        xyz.update({f"{axis}_bias_mm": None if not values.size else float(np.mean(values)), f"{axis}_mae_mm": None if not values.size else float(np.mean(abs(values))), f"{axis}_rmse_mm": None if not values.size else float(np.sqrt(np.mean(values ** 2))), f"{axis}_std_mm": None if not values.size else float(np.std(values)), f"{axis}_max_error_mm": None if not values.size else float(np.max(abs(values)))})
    report = {"status": "OK", "alignment_mode": alignment_mode, "evaluated_xyz_stage": xyz_stage, "matched_count": len(evaluation), "metrics": metrics, "xyz": xyz, "xy_uncertainty": "PARTIAL"}
    report_path = output / "e0_static_report.json"; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return EvaluationRunResult("OK", report_path, records_path)


def _calibration_baseline_mm(path: Path) -> float:
    calibration = StereoCalibration.from_json(path)
    tx = abs(float(np.asarray(calibration.translation).reshape(-1)[0]))
    return tx if calibration.unit == "mm" else tx * 1000.0


def run_e1_manifest(manifest_path: str | Path, output_dir: str | Path, *, baseline_tolerance_mm: float = 1.0) -> EvaluationRunResult:
    """Validate baseline/calibration identity and evaluate any conditions carrying real GT."""
    manifest_file, output = Path(manifest_path), Path(output_dir)
    payload = json.loads(manifest_file.read_text(encoding="utf-8")); conditions = payload.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("E1 manifest needs a non-empty conditions list")
    summary: list[dict[str, object]] = []; has_gt = False
    for index, condition in enumerate(conditions):
        for key in ("distance_m", "baseline_mm", "target_sigma_z_mm", "method", "repeat", "calibration_id", "calibration", "measurement"):
            if key not in condition:
                raise ValueError(f"E1 condition {index} missing {key}")
        calibration = (manifest_file.parent / condition["calibration"]).resolve()
        baseline = _calibration_baseline_mm(calibration)
        if abs(float(condition["baseline_mm"]) - baseline) > baseline_tolerance_mm:
            raise ValueError(f"E1 baseline/calibration mismatch: manifest={condition['baseline_mm']} mm, calibration={baseline:.6f} mm")
        gt_value = condition.get("gt")
        if gt_value:
            has_gt = True
            result = evaluate_e0_static((manifest_file.parent / condition["measurement"]).resolve(), (manifest_file.parent / gt_value).resolve(), output / f"condition_{index}", alignment_mode=str(condition.get("alignment_mode", "frame")), max_time_offset_ms=float(condition.get("max_time_offset_ms", 20.0)))
            summary.append({**condition, "evaluation_status": result.status, "report_path": None if result.report_path is None else str(result.report_path)})
        else:
            summary.append({**condition, "evaluation_status": REAL_DATA_REQUIRED})
    output.mkdir(parents=True, exist_ok=True); report_path = output / "e1_distance_baseline_accuracy.json"; report_path.write_text(json.dumps({"status": "OK" if has_gt else REAL_DATA_REQUIRED, "conditions": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return EvaluationRunResult("OK" if has_gt else REAL_DATA_REQUIRED, report_path=report_path)


def _evaluate_csv(measurement_csv: str | Path, gt_records: list[GroundTruthRecord], *, alignment_mode: str = "frame") -> list[UncertaintyEvaluationRecord]:
    records: list[UncertaintyEvaluationRecord] = []
    for row in _rows(Path(measurement_csv)):
        gt = align_ground_truth(row, gt_records, alignment_mode=alignment_mode)
        if gt is not None:
            records.append(UncertaintyEvaluationRecord.from_measurement_and_gt(row, gt))
    return records


def run_uncertainty_experiment_matrix(*, gt_csv: str | Path | None, output_dir: str | Path, refinement_csvs: dict[int, str | Path], retry_csvs: dict[str, str | Path], quality_csvs: dict[str, str | Path], alignment_mode: str = "frame") -> EvaluationRunResult:
    """Create U0--U3 tables from independently generated measurement CSVs and GT.

    The caller must run every variant beforehand; this function cannot feed GT back to a
    matcher and only joins output files after all measurement has completed.
    """
    if gt_csv is None or not Path(gt_csv).exists():
        return EvaluationRunResult(REAL_DATA_REQUIRED, message="GT is required for U0--U3 evaluation")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    gt_records = [GroundTruthRecord.from_row(row) for row in _rows(Path(gt_csv))]
    def summary(label: str, csv_path: str | Path) -> dict[str, object]:
        records = _evaluate_csv(csv_path, gt_records, alignment_mode=alignment_mode)
        return {"variant": label, "measurement_csv": str(csv_path), "n": len(records), **evaluate_records(records)}
    refinement = [summary(f"R{level}", path) | {"refinement_level": level} for level, path in sorted(refinement_csvs.items())]
    _write_csv(output / "refinement_ablation.csv", refinement)
    quality = [summary(label, path) | {"vision_quality": label} for label, path in quality_csvs.items()]
    _write_csv(output / "vision_quality_stratification.csv", quality)
    retry = [summary(label, path) | {"retry_mode": label} for label, path in retry_csvs.items()]
    retry_by_name = {row["retry_mode"]: row for row in retry}
    no_retry = retry_by_name.get("NO_RETRY"); bounded = retry_by_name.get("BOUNDED_RETRY")
    if no_retry and bounded:
        before = float(no_retry.get("disparity_rmse") or np.nan); after = float(bounded.get("disparity_rmse") or np.nan)
        gain = before - after if np.isfinite(before) and np.isfinite(after) else None
        bounded_rows = _rows(Path(retry_csvs["BOUNDED_RETRY"]))
        triggered = [row for row in bounded_rows if str(row.get("retry_triggered", "")).lower() == "true"]
        accepted = sum(str(row.get("retry_candidate_accepted", "")).lower() == "true" for row in triggered)
        before_records = {(record.frame, record.point_id): record for record in _evaluate_csv(retry_csvs["NO_RETRY"], gt_records, alignment_mode=alignment_mode)}
        after_records = {(record.frame, record.point_id): record for record in _evaluate_csv(retry_csvs["BOUNDED_RETRY"], gt_records, alignment_mode=alignment_mode)}
        gains = [abs(before_records[key].disparity_error_px) - abs(after_records[key].disparity_error_px) for key in before_records.keys() & after_records.keys() if before_records[key].disparity_error_px is not None and after_records[key].disparity_error_px is not None]
        bounded.update({"retry_trigger_count": len(triggered), "retry_accept_count": accepted, "retry_actual_improvement_count": sum(value > 0 for value in gains), "retry_worsened_count": sum(value < 0 for value in gains), "retry_no_change_count": sum(value == 0 for value in gains), "mean_error_gain_px": None if not gains else float(np.mean(gains)), "aggregate_rmse_gain_px": gain, "effectiveness": "LOW_EFFECTIVENESS" if not gains or float(np.mean(gains)) <= 0 else "OBSERVED_GAIN"})
    _write_csv(output / "retry_effectiveness.csv", retry)
    u0 = summary("U0_SYNTHETIC_OR_EXTERNAL", next(iter(refinement_csvs.values()))) if refinement_csvs else {}
    _write_csv(output / "uncertainty_calibration_table.csv", [u0])
    return EvaluationRunResult("OK", report_path=output / "uncertainty_calibration_table.csv")


def run_u0_synthetic_sanity(synthetic_csv: str | Path, output_dir: str | Path, *, focal_length_px: float, baseline_mm: float, localization_sigma_px: float) -> EvaluationRunResult:
    """Evaluate simulator samples with its declared localization-noise reference.

    This is explicitly a simulator-noise sanity check, not a claim that the runtime
    heuristic estimator has been calibrated on real data.
    """
    if focal_length_px <= 0 or baseline_mm <= 0 or localization_sigma_px <= 0:
        raise ValueError("synthetic U0 geometry and localization sigma must be positive")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    records: list[UncertaintyEvaluationRecord] = []
    sigma_d = float(np.sqrt(2.0) * localization_sigma_px)
    for row in _rows(Path(synthetic_csv)):
        z_mm = float(row["Z_estimated_mm"]); gt_z_mm = float(row["Z_gt_mm"])
        estimated_d = float(row["left_x_px"]) - float(row["right_x_px"])
        gt_d = focal_length_px * baseline_mm / gt_z_mm
        sigma_z = z_mm ** 2 * sigma_d / (focal_length_px * baseline_mm)
        measurement = {"frame": row["frame"], "point_id": row["point_id"], "method": "U0_SIMULATOR", "estimated_disparity": estimated_d, "estimated_sigma_d_px": sigma_d, "estimated_X_m": float(row["X_estimated_mm"]) / 1000.0, "estimated_Y_m": float(row["Y_estimated_mm"]) / 1000.0, "estimated_Z_m": z_mm / 1000.0, "estimated_sigma_z_mm": sigma_z}
        gt = GroundTruthRecord(frame=int(float(row["frame"])), point_id=row["point_id"], x_mm=float(row["X_gt_mm"]), y_mm=float(row["Y_gt_mm"]), z_mm=gt_z_mm, disparity_px=gt_d)
        records.append(UncertaintyEvaluationRecord.from_measurement_and_gt(measurement, gt))
    _write_csv(output / "uncertainty_evaluation.csv", [record.as_dict() for record in records])
    summary = {"method": "U0_SIMULATOR_NOISE_REFERENCE", "status": "NOT_RUNTIME_CALIBRATION", "localization_sigma_px": localization_sigma_px, "estimated_sigma_d_px": sigma_d, **evaluate_records(records)}
    _write_csv(output / "uncertainty_calibration_table.csv", [summary])
    return EvaluationRunResult("OK", report_path=output / "uncertainty_calibration_table.csv", records_path=output / "uncertainty_evaluation.csv")
