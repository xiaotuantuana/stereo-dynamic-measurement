from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from stereo_research.calibration import StereoCalibration
from stereo_research.evaluation_runner import evaluate_e0_static, run_e1_manifest, run_uncertainty_experiment_matrix


def _calibration(path: Path, tx_mm: float) -> None:
    StereoCalibration("test", (8, 6), np.eye(3), np.eye(3), np.zeros(5), np.zeros(5), np.eye(3), np.array([tx_mm, 0.0, 0.0]), "mm").to_json(path)


def _measurement(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "point_id", "method", "estimated_disparity", "estimated_sigma_d_px", "estimated_X_m", "estimated_Y_m", "estimated_Z_m", "final_X_m", "final_Y_m", "final_Z_m", "estimated_sigma_z_mm"])
        writer.writeheader(); writer.writerow({"frame": 0, "point_id": "p0", "method": "M3", "estimated_disparity": 10, "estimated_sigma_d_px": .5, "estimated_X_m": 0, "estimated_Y_m": 0, "estimated_Z_m": 2, "final_X_m": 0, "final_Y_m": 0, "final_Z_m": 2, "estimated_sigma_z_mm": 5})


def test_e0_reads_separate_gt_and_reports_data_required_when_absent(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; _measurement(measurement)
    result = evaluate_e0_static(measurement, tmp_path / "missing_gt.csv", tmp_path / "out")
    assert result.status == "REAL_DATA_REQUIRED"
    gt = tmp_path / "gt.csv"
    gt.write_text("frame,point_id,gt_x_mm,gt_y_mm,gt_z_mm\n0,p0,0,0,2000\n", encoding="utf-8")
    result = evaluate_e0_static(measurement, gt, tmp_path / "out")
    assert result.status == "OK" and result.report_path.exists()


def test_e1_identity_guard_fails_for_mismatched_baseline_and_accepts_matching_one(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; _measurement(measurement)
    calibration = tmp_path / "calibration_100.json"; _calibration(calibration, -100.0)
    bad = tmp_path / "bad.json"
    bad.write_text('{"conditions":[{"distance_m":2,"baseline_mm":150,"target_sigma_z_mm":5,"method":"M3","repeat":1,"calibration_id":"b150","calibration":"calibration_100.json","measurement":"measurement.csv"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="baseline"):
        run_e1_manifest(bad, tmp_path / "out")
    good = tmp_path / "good.json"
    good.write_text('{"conditions":[{"distance_m":2,"baseline_mm":100,"target_sigma_z_mm":5,"method":"M3","repeat":1,"calibration_id":"b100","calibration":"calibration_100.json","measurement":"measurement.csv"}]}', encoding="utf-8")
    assert run_e1_manifest(good, tmp_path / "out").status == "REAL_DATA_REQUIRED"


def test_uncertainty_matrix_writes_independent_refinement_retry_and_quality_tables(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; _measurement(measurement)
    gt = tmp_path / "gt.csv"; gt.write_text("frame,point_id,gt_x_mm,gt_y_mm,gt_z_mm,gt_disparity_px\n0,p0,0,0,2000,10\n", encoding="utf-8")
    result = run_uncertainty_experiment_matrix(
        gt_csv=gt, output_dir=tmp_path / "out",
        refinement_csvs={0: measurement, 1: measurement, 2: measurement},
        retry_csvs={"NO_RETRY": measurement, "BOUNDED_RETRY": measurement},
        quality_csvs={"good": measurement, "medium": measurement, "poor": measurement},
    )
    assert result.status == "OK"
    assert (tmp_path / "out" / "refinement_ablation.csv").exists()
    assert (tmp_path / "out" / "retry_effectiveness.csv").exists()
    assert (tmp_path / "out" / "vision_quality_stratification.csv").exists()
