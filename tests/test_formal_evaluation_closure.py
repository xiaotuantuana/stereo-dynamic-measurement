from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from stereo_research.calibration import StereoCalibration
from stereo_research.real_experiment import (
    REAL_EVALUATION_DATA_REQUIRED,
    evaluate_e0_manifest,
    evaluate_e1_manifest,
)
from stereo_research.evaluation_runner import (
    FINAL_XYZ_UNAVAILABLE,
    evaluate_e0_static,
)
from stereo_research.runner import CSV_FIELDS


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _measurement(path: Path, *, final_z_m: float = 2.0) -> None:
    _write_csv(
        path,
        [{
            "frame": 0,
            "frame_timestamp_s": 1.0,
            "point_id": "P1",
            "method": "M3",
            "status": "valid",
            "estimated_disparity": 48.0,
            "estimated_sigma_d_px": 0.1,
            "required_sigma_d_px": 0.08,
            "estimated_X_m": 0.0,
            "estimated_Y_m": 0.0,
            "estimated_Z_m": 2.1,
            "final_X_m": 0.0,
            "final_Y_m": 0.0,
            "final_Z_m": final_z_m,
            "estimated_sigma_z_mm": 5.0,
            "precision_status": "MET",
            "policy_refinement_level": 1,
            "policy_precision_retry_count": 0,
            "frame_total_ms": 10.0,
        }],
    )


def _measurement_with_values(
    path: Path,
    *,
    final_z_m: float,
    runtime_ms: float,
    precision_status: str = "MET",
) -> None:
    _measurement(path, final_z_m=final_z_m)
    rows = _read(path)
    rows[0]["frame_total_ms"] = str(runtime_ms)
    rows[0]["precision_status"] = precision_status
    _write_csv(path, rows)


def _gt(path: Path) -> None:
    _write_csv(
        path,
        [{
            "frame": 0,
            "timestamp_s": 1.0,
            "point_id": "P1",
            "gt_x_mm": 0.0,
            "gt_y_mm": 0.0,
            "gt_z_mm": 2000.0,
        }],
    )


def _calibration(path: Path) -> None:
    StereoCalibration(
        "cal120",
        (8, 6),
        np.eye(3),
        np.eye(3),
        np.zeros(5),
        np.zeros(5),
        np.eye(3),
        np.array([-120.0, 0.0, 0.0]),
        "mm",
    ).to_json(path)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _e0_entry(sequence_id: str, split: str) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "split": split,
        "measurement_csv": "measurement.csv",
        "gt_csv": "gt.csv",
        "alignment_mode": "frame",
        "max_time_offset_ms": 20,
        "calibration_id": "cal120",
        "distance_m": 2.0,
        "baseline_mm": 120.0,
        "method": "M3",
        "repeat": 1,
        "notes": "test",
    }


def _e1_entry(sequence_id: str, split: str) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "split": split,
        "distance_m": 2.0,
        "baseline_mm": 120.0,
        "target_sigma_z_mm": 5.0,
        "method": "M3",
        "repeat": 1,
        "calibration_id": "cal120",
        "calibration": "calibration.json",
        "measurement": "measurement.csv",
        "gt": "gt.csv",
        "alignment_mode": "frame",
    }


def test_e0_formal_results_exclude_calibration_split(tmp_path: Path) -> None:
    _measurement(tmp_path / "measurement.csv")
    _gt(tmp_path / "gt.csv")
    manifest = tmp_path / "e0.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E0",
        "experiment_id": "mixed",
        "gt_unit": "mm",
        "data_source": "REAL_E0",
        "sequences": [
            _e0_entry("CAL", "calibration"),
            _e0_entry("EVAL", "evaluation"),
        ],
    }), encoding="utf-8")

    result = evaluate_e0_manifest(manifest, tmp_path / "out")

    assert result.status == "OK"
    rows = _read(tmp_path / "out" / "e0_per_sequence.csv")
    assert [(row["sequence_id"], row["split"]) for row in rows] == [("EVAL", "evaluation")]


def test_e1_formal_results_exclude_calibration_split(tmp_path: Path) -> None:
    _measurement(tmp_path / "measurement.csv")
    _gt(tmp_path / "gt.csv")
    _calibration(tmp_path / "calibration.json")
    manifest = tmp_path / "e1.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E1",
        "experiment_id": "mixed",
        "gt_unit": "mm",
        "data_source": "REAL_E1",
        "conditions": [
            _e1_entry("CAL", "calibration"),
            _e1_entry("EVAL", "evaluation"),
        ],
    }), encoding="utf-8")

    result = evaluate_e1_manifest(manifest, tmp_path / "out")

    assert result.status == "OK"
    rows = _read(tmp_path / "out" / "e1_per_condition.csv")
    assert [(row["sequence_id"], row["split"]) for row in rows] == [("EVAL", "evaluation")]


def test_calibration_only_manifest_requires_real_evaluation_data(tmp_path: Path) -> None:
    _measurement(tmp_path / "measurement.csv")
    _gt(tmp_path / "gt.csv")
    manifest = tmp_path / "e0.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E0",
        "experiment_id": "calibration-only",
        "gt_unit": "mm",
        "data_source": "REAL_E0",
        "sequences": [_e0_entry("CAL", "calibration")],
    }), encoding="utf-8")

    result = evaluate_e0_manifest(manifest, tmp_path / "out")

    assert result.status == REAL_EVALUATION_DATA_REQUIRED
    assert _read(tmp_path / "out" / "e0_per_sequence.csv") == []


def test_formal_e0_uses_final_xyz_and_independent_gt(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"
    _measurement(measurement, final_z_m=2.0)
    rows = _read(measurement)
    rows[0].update({"X_gt": "999", "Y_gt": "999", "Z_gt": "999"})
    _write_csv(measurement, rows)
    _gt(tmp_path / "gt.csv")

    result = evaluate_e0_static(measurement, tmp_path / "gt.csv", tmp_path / "out")

    assert result.status == "OK"
    records = _read(result.records_path)
    assert records[0]["evaluated_xyz_stage"] == "final"
    assert float(records[0]["z_error_mm"]) == 0.0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["evaluated_xyz_stage"] == "final"


def test_formal_e0_does_not_fall_back_when_final_xyz_columns_are_missing(
    tmp_path: Path,
) -> None:
    measurement = tmp_path / "measurement.csv"
    _measurement(measurement)
    rows = _read(measurement)
    for row in rows:
        for field in ("final_X_m", "final_Y_m", "final_Z_m"):
            row.pop(field)
    _write_csv(measurement, rows)
    _gt(tmp_path / "gt.csv")

    result = evaluate_e0_static(measurement, tmp_path / "gt.csv", tmp_path / "out")

    assert result.status == FINAL_XYZ_UNAVAILABLE
    assert "final_X_m" in result.message


def test_e0_summary_aggregates_repeats_and_paper_table_uses_summary(
    tmp_path: Path,
) -> None:
    _gt(tmp_path / "gt.csv")
    sequences = []
    for repeat, error_mm in enumerate((1.0, 2.0, 3.0), start=1):
        measurement = tmp_path / f"measurement_{repeat}.csv"
        _measurement(measurement, final_z_m=(2000.0 + error_mm) / 1000.0)
        entry = _e0_entry(f"EVAL_{repeat}", "evaluation")
        entry.update({"measurement_csv": measurement.name, "repeat": repeat})
        sequences.append(entry)
    sequences.insert(0, _e0_entry("CAL", "calibration"))
    _measurement(tmp_path / "measurement.csv", final_z_m=2.100)
    manifest = tmp_path / "e0.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E0",
        "experiment_id": "aggregate",
        "gt_unit": "mm",
        "data_source": "REAL_E0",
        "sequences": sequences,
    }), encoding="utf-8")

    evaluate_e0_manifest(manifest, tmp_path / "out")

    per_rows = _read(tmp_path / "out" / "e0_per_sequence.csv")
    summary = _read(tmp_path / "out" / "e0_summary.csv")
    paper = _read(tmp_path / "out" / "table_e0_static_accuracy.csv")
    metadata = json.loads((tmp_path / "out" / "evaluation_metadata.json").read_text(encoding="utf-8"))
    assert len(per_rows) == 3
    assert len(summary) == 1
    assert summary[0]["sequence_count"] == "3"
    assert summary[0]["sample_count"] == "3"
    assert np.isclose(float(summary[0]["Z_RMSE_mean"]), 2.0)
    assert np.isclose(
        float(summary[0]["Z_RMSE_std"]), np.std([1.0, 2.0, 3.0])
    )
    assert len(paper) == 1
    assert paper[0]["data_split"] == "evaluation"
    assert paper[0]["evaluated_xyz_stage"] == "final"
    assert np.isclose(float(paper[0]["Z_RMSE_mean"]), 2.0)
    assert metadata == {
        "data_source": "REAL_E0",
        "formal_results_split": "evaluation",
        "evaluated_xyz_stage": "final",
        "num_calibration_sequences": 1,
        "num_evaluation_sequences": 3,
    }


def test_e1_summary_aggregates_repeat_statistics_and_precision_rates(
    tmp_path: Path,
) -> None:
    _gt(tmp_path / "gt.csv")
    _calibration(tmp_path / "calibration.json")
    conditions = []
    for repeat, (error_mm, runtime_ms, status) in enumerate(
        ((1.0, 10.0, "MET"), (3.0, 30.0, "VALID_BUT_PRECISION_UNMET")),
        start=1,
    ):
        measurement = tmp_path / f"measurement_{repeat}.csv"
        _measurement_with_values(
            measurement,
            final_z_m=(2000.0 + error_mm) / 1000.0,
            runtime_ms=runtime_ms,
            precision_status=status,
        )
        entry = _e1_entry(f"EVAL_{repeat}", "evaluation")
        entry.update({"measurement": measurement.name, "repeat": repeat})
        conditions.append(entry)
    manifest = tmp_path / "e1.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E1",
        "experiment_id": "aggregate",
        "gt_unit": "mm",
        "data_source": "REAL_E1",
        "conditions": conditions,
    }), encoding="utf-8")

    evaluate_e1_manifest(manifest, tmp_path / "out")

    summary = _read(tmp_path / "out" / "e1_summary.csv")
    paper = _read(tmp_path / "out" / "table_e1_distance_baseline.csv")
    assert len(summary) == 1
    row = summary[0]
    assert row["repeat_count"] == "2"
    assert row["sample_count"] == "2"
    assert float(row["Z_RMSE_mean"]) == 2.0
    assert float(row["Z_RMSE_std"]) == 1.0
    assert float(row["runtime_mean"]) == 20.0
    assert float(row["runtime_std"]) == 10.0
    assert float(row["MET_rate"]) == 0.5
    assert float(row["VALID_BUT_PRECISION_UNMET_rate"]) == 0.5
    assert paper[0]["data_split"] == "evaluation"
    assert paper[0]["evaluated_xyz_stage"] == "final"


def test_synthetic_paper_table_remains_explicitly_synthetic(tmp_path: Path) -> None:
    _measurement(tmp_path / "measurement.csv")
    _gt(tmp_path / "gt.csv")
    manifest = tmp_path / "e0.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E0",
        "experiment_id": "synthetic",
        "gt_unit": "mm",
        "data_source": "SYNTHETIC_DRY_RUN",
        "sequences": [_e0_entry("EVAL", "evaluation")],
    }), encoding="utf-8")

    result = evaluate_e0_manifest(manifest, tmp_path / "out")

    assert result.status == "SYNTHETIC_DRY_RUN"
    paper = _read(tmp_path / "out" / "table_e0_static_accuracy.csv")
    assert {row["data_source"] for row in paper} == {"SYNTHETIC_DRY_RUN"}
    assert {row["data_split"] for row in paper} == {"evaluation"}


def test_e0_manifest_surfaces_missing_final_xyz_as_an_error(tmp_path: Path) -> None:
    _measurement(tmp_path / "measurement.csv")
    rows = _read(tmp_path / "measurement.csv")
    for field in ("final_X_m", "final_Y_m", "final_Z_m"):
        rows[0].pop(field)
    _write_csv(tmp_path / "measurement.csv", rows)
    _gt(tmp_path / "gt.csv")
    manifest = tmp_path / "e0.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E0",
        "experiment_id": "missing-final",
        "gt_unit": "mm",
        "data_source": "REAL_E0",
        "sequences": [_e0_entry("EVAL", "evaluation")],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="FINAL_XYZ_UNAVAILABLE"):
        evaluate_e0_manifest(manifest, tmp_path / "out")


def test_e1_calibration_only_manifest_requires_evaluation_data(tmp_path: Path) -> None:
    _measurement(tmp_path / "measurement.csv")
    _gt(tmp_path / "gt.csv")
    _calibration(tmp_path / "calibration.json")
    manifest = tmp_path / "e1.json"
    manifest.write_text(json.dumps({
        "experiment_type": "E1",
        "experiment_id": "calibration-only",
        "gt_unit": "mm",
        "data_source": "REAL_E1",
        "conditions": [_e1_entry("CAL", "calibration")],
    }), encoding="utf-8")

    result = evaluate_e1_manifest(manifest, tmp_path / "out")

    assert result.status == REAL_EVALUATION_DATA_REQUIRED
    assert _read(tmp_path / "out" / "e1_per_condition.csv") == []


def test_explicit_estimated_stage_remains_available_for_ablation(tmp_path: Path) -> None:
    _measurement(tmp_path / "measurement.csv", final_z_m=2.0)
    _gt(tmp_path / "gt.csv")

    result = evaluate_e0_static(
        tmp_path / "measurement.csv",
        tmp_path / "gt.csv",
        tmp_path / "out",
        xyz_stage="estimated",
    )

    records = _read(result.records_path)
    assert records[0]["evaluated_xyz_stage"] == "estimated"
    assert float(records[0]["z_error_mm"]) == pytest.approx(100.0)


def test_measurement_csv_contract_remains_199_columns() -> None:
    assert len(CSV_FIELDS) == 199
