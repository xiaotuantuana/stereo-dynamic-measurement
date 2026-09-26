from pathlib import Path

import pandas as pd

from experiment.validation.innovation3 import run_innovation3_controlled_validation
from stereo_dynamic_measurement.simulation.fault_injection import FaultInjector


def _source() -> pd.DataFrame:
    return pd.DataFrame({
        "frame": range(12), "point_id": ["P1"] * 12,
        "x_left": [10.0] * 12, "x_right": [5.0] * 12,
        "y_left": [10.0] * 12, "y_right": [10.0] * 12,
        "X_raw": [0.0] * 12, "Y_raw": [0.0] * 12, "Z_raw": [2000.0] * 12,
        "observed": [True] * 12,
    })


def test_fault_severity_changes_stereo_mismatch_strength() -> None:
    injector = FaultInjector(seed=3)
    mild = injector.inject(_source(), "stereo_mismatch", severity="mild")
    severe = injector.inject(_source(), "stereo_mismatch", severity="severe")
    assert abs(severe.x_right.iloc[0] - 5.0) > abs(mild.x_right.iloc[0] - 5.0)
    assert set(severe.injected_severity) == {"severe"}


def test_innovation3_validation_reports_confusion_and_measurement_safety(tmp_path: Path) -> None:
    summary = run_innovation3_controlled_validation(tmp_path, seeds=2)
    assert summary["fault_class_count"] >= 5
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
    assert 0.0 <= summary["f1"] <= 1.0
    assert 0.0 <= summary["false_positive_rate"] <= 1.0
    assert 0.0 <= summary["false_negative_rate"] <= 1.0
    assert "catastrophic_interception_rate" in summary
    assert summary["ground_truth_online_access"] is False
    assert (tmp_path / "confusion_matrix.csv").is_file()
    assert (tmp_path / "severity_metrics.csv").is_file()
    assert (tmp_path / "false_diagnosis_cases.csv").is_file()
    assert (tmp_path / "run_metadata.json").is_file()
    assert (tmp_path / "diagnosis_success_cases.csv").is_file()
    assert (tmp_path / "false_positive_cases.csv").is_file()
    assert (tmp_path / "false_negative_cases.csv").is_file()
