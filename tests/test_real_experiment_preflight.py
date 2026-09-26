from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from stereo_research.calibration import StereoCalibration
from stereo_research.real_experiment import (
    create_synthetic_demo,
    evaluate_e0_manifest,
    evaluate_e1_manifest,
    load_real_experiment_manifest,
    run_synthetic_dry_run,
    validate_real_experiment,
)
from stereo_research.runner import CSV_FIELDS, _run_method_once


def _calibration(path: Path, baseline_mm: float = 120.0) -> None:
    StereoCalibration(
        "dry-run", (8, 6), np.eye(3), np.eye(3), np.zeros(5), np.zeros(5),
        np.eye(3), np.array([-baseline_mm, 0.0, 0.0]), "mm",
    ).to_json(path)


def _measurement(path: Path, *, timestamp: float = 1.0) -> None:
    fields = [
        "frame", "frame_timestamp_s", "point_id", "method", "status",
        "estimated_disparity", "estimated_sigma_d_px", "required_sigma_d_px",
        "estimated_X_m", "estimated_Y_m", "estimated_Z_m", "estimated_sigma_z_mm",
        "precision_status", "policy_refinement_level", "policy_precision_retry_count",
        "frame_total_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow({"frame": 0, "frame_timestamp_s": timestamp, "point_id": "P1", "method": "M3", "status": "valid", "estimated_disparity": 48.1, "estimated_sigma_d_px": .1, "required_sigma_d_px": .08, "estimated_X_m": .01, "estimated_Y_m": .02, "estimated_Z_m": 2.001, "estimated_sigma_z_mm": 4.2, "precision_status": "MET", "policy_refinement_level": 1, "policy_precision_retry_count": 0, "frame_total_ms": 12.5})


def _gt(path: Path, *, duplicate: bool = False, invalid_xyz: bool = False, timestamp: float = 1.0) -> None:
    rows = ["0,1.0,P1,10,20,2000,0.1,0.1,0.2,SYNTHETIC_DRY_RUN"]
    if duplicate:
        rows.append(rows[0])
    if invalid_xyz:
        rows[0] = "0,1.0,P1,nan,20,2000,0.1,0.1,0.2,SYNTHETIC_DRY_RUN"
    path.write_text("frame,timestamp_s,point_id,gt_x_mm,gt_y_mm,gt_z_mm,gt_sigma_x_mm,gt_sigma_y_mm,gt_sigma_z_mm,data_source\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _e0_manifest(path: Path, measurement: Path, gt: Path | None, *, split: str = "evaluation", alignment: str = "frame") -> None:
    sequence = {"sequence_id": "S1", "split": split, "measurement_csv": measurement.name, "gt_csv": None if gt is None else gt.name, "alignment_mode": alignment, "max_time_offset_ms": 20, "calibration_id": "cal120", "distance_m": 2.0, "baseline_mm": 120.0, "method": "M3", "repeat": 1, "notes": "test"}
    path.write_text(json.dumps({"experiment_type": "E0", "experiment_id": "dry", "gt_unit": "mm", "data_source": "SYNTHETIC_DRY_RUN", "sequences": [sequence]}), encoding="utf-8")


def test_e0_and_e1_manifests_are_accepted(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; gt = tmp_path / "gt.csv"; calibration = tmp_path / "calibration.json"
    _measurement(measurement); _gt(gt); _calibration(calibration)
    e0 = tmp_path / "e0.json"; _e0_manifest(e0, measurement, gt)
    assert load_real_experiment_manifest(e0, "E0")["sequences"][0]["sequence_id"] == "S1"
    e1 = tmp_path / "e1.json"
    e1.write_text(json.dumps({"experiment_type":"E1","experiment_id":"dry","gt_unit":"mm","data_source":"SYNTHETIC_DRY_RUN","conditions":[{"sequence_id":"B1","split":"evaluation","distance_m":2,"baseline_mm":120,"target_sigma_z_mm":5,"method":"M3","repeat":1,"calibration_id":"cal120","calibration":"calibration.json","measurement":"measurement.csv","gt":"gt.csv","alignment_mode":"frame"}]}), encoding="utf-8")
    assert load_real_experiment_manifest(e1, "E1")["conditions"][0]["calibration_id"] == "cal120"


def test_calibration_and_evaluation_sequence_overlap_fails_fast(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; gt = tmp_path / "gt.csv"; _measurement(measurement); _gt(gt)
    manifest = tmp_path / "e0.json"; _e0_manifest(manifest, measurement, gt)
    payload = json.loads(manifest.read_text()); payload["sequences"].append({**payload["sequences"][0], "split": "calibration"})
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_real_experiment_manifest(manifest, "E0")


@pytest.mark.parametrize("duplicate,invalid", [(True, False), (False, True)])
def test_validator_rejects_duplicate_or_invalid_gt(tmp_path: Path, duplicate: bool, invalid: bool) -> None:
    measurement = tmp_path / "measurement.csv"; gt = tmp_path / "gt.csv"; _measurement(measurement); _gt(gt, duplicate=duplicate, invalid_xyz=invalid)
    manifest = tmp_path / "e0.json"; _e0_manifest(manifest, measurement, gt)
    result = validate_real_experiment(manifest)
    assert result.status == "FAIL"


@pytest.mark.parametrize("alignment", ["frame", "timestamp"])
def test_alignment_dry_run_succeeds(tmp_path: Path, alignment: str) -> None:
    measurement = tmp_path / "measurement.csv"; gt = tmp_path / "gt.csv"; _measurement(measurement); _gt(gt)
    manifest = tmp_path / "e0.json"; _e0_manifest(manifest, measurement, gt, alignment=alignment)
    result = validate_real_experiment(manifest)
    assert result.status == "PASS"
    assert result.alignment_success_rate == pytest.approx(1.0)


def test_e1_baseline_calibration_mismatch_fails_fast(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; gt = tmp_path / "gt.csv"; calibration = tmp_path / "calibration.json"
    _measurement(measurement); _gt(gt); _calibration(calibration, 100)
    manifest = tmp_path / "e1.json"
    manifest.write_text(json.dumps({"experiment_type":"E1","experiment_id":"dry","gt_unit":"mm","data_source":"SYNTHETIC_DRY_RUN","conditions":[{"sequence_id":"B1","split":"evaluation","distance_m":2,"baseline_mm":120,"target_sigma_z_mm":5,"method":"M3","repeat":1,"calibration_id":"cal100","calibration":"calibration.json","measurement":"measurement.csv","gt":"gt.csv"}]}), encoding="utf-8")
    assert validate_real_experiment(manifest).status == "FAIL"


def test_e1_identity_guard_still_runs_when_gt_is_missing(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; calibration = tmp_path / "calibration.json"
    _measurement(measurement); _calibration(calibration, 100)
    manifest = tmp_path / "e1.json"
    manifest.write_text(json.dumps({"experiment_type":"E1","experiment_id":"real","gt_unit":"mm","data_source":"REAL_E1","conditions":[{"sequence_id":"B1","split":"evaluation","distance_m":2,"baseline_mm":120,"target_sigma_z_mm":5,"method":"M3","repeat":1,"calibration_id":"cal100","calibration":"calibration.json","measurement":"measurement.csv","gt":None}]}), encoding="utf-8")
    result = validate_real_experiment(manifest)
    assert result.status == "FAIL"
    assert any(issue.code == "BASELINE_IDENTITY" for issue in result.issues)


def test_missing_real_gt_returns_real_data_required(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.csv"; _measurement(measurement)
    manifest = tmp_path / "e0.json"; _e0_manifest(manifest, measurement, None); payload=json.loads(manifest.read_text()); payload["data_source"]="REAL_E0"; manifest.write_text(json.dumps(payload),encoding="utf-8")
    result = evaluate_e0_manifest(manifest, tmp_path / "results")
    assert result.status == "REAL_DATA_REQUIRED"


def test_synthetic_demo_runs_end_to_end_and_labels_every_table(tmp_path: Path) -> None:
    demo = create_synthetic_demo(tmp_path / "synthetic_demo")
    result = run_synthetic_dry_run(demo)
    assert result.status == "SYNTHETIC_DRY_RUN"
    assert result.output_paths
    for path in result.output_paths:
        if path.suffix == ".csv":
            assert "SYNTHETIC_DRY_RUN" in path.read_text(encoding="utf-8-sig")
    for path in (demo / "results").rglob("*"):
        if path.suffix in {".csv", ".json"}:
            assert "SYNTHETIC_DRY_RUN" in path.read_text(encoding="utf-8-sig")


def test_e0_and_e1_one_command_workflows_generate_expected_tables(tmp_path: Path) -> None:
    demo = create_synthetic_demo(tmp_path / "synthetic_demo")
    e0 = evaluate_e0_manifest(demo / "e0_manifest.json", demo / "results" / "E0")
    e1 = evaluate_e1_manifest(demo / "e1_manifest.json", demo / "results" / "E1")
    assert e0.status == "SYNTHETIC_DRY_RUN"
    assert (demo / "results" / "E0" / "e0_summary.csv").exists()
    assert e1.status == "SYNTHETIC_DRY_RUN"
    assert (demo / "results" / "E1" / "e1_per_condition.csv").exists()


def test_runtime_pipeline_still_has_no_gt_input_and_csv_stays_199_columns() -> None:
    assert "ground_truth" not in inspect.signature(_run_method_once).parameters
    assert len(CSV_FIELDS) == 199
