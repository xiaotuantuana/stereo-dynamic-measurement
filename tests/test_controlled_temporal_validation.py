from pathlib import Path

import pandas as pd

from experiment.simulation.temporal_scenarios import build_temporal_scenarios
from experiment.validation.innovation2 import run_innovation2_controlled_validation


def test_temporal_scenario_matrix_covers_required_controlled_cases() -> None:
    scenarios = build_temporal_scenarios(frame_count=48, seed=11)
    assert set(scenarios) == {
        "stable", "gaussian_noise", "single_jump", "continuous_outlier",
        "slow_drift", "fast_legitimate_motion", "frame_loss", "measurement_outlier",
    }
    assert all(scenario.provenance == "CONTROLLED" for scenario in scenarios.values())


def test_innovation2_validation_reports_correction_and_safety_metrics(tmp_path: Path) -> None:
    summary = run_innovation2_controlled_validation(tmp_path, frame_count=48, seed=11)
    assert summary["scenario_count"] == 8
    assert 0.0 <= summary["aggregate_false_correction_rate"] <= 1.0
    assert (tmp_path / "scenario_metrics.csv").is_file()
    assert (tmp_path / "false_correction_cases.csv").is_file()
    assert (tmp_path / "run_metadata.json").is_file()
    assert summary["ground_truth_online_access"] is False
    assert "phase" in summary["unavailable_runtime_evidence"]
    assert "coherence" in summary["unavailable_runtime_evidence"]
    metrics = pd.read_csv(tmp_path / "scenario_metrics.csv").set_index("scenario")
    assert metrics.loc["single_jump", "corrected_rmse_mm"] < metrics.loc["single_jump", "raw_rmse_mm"]
    assert metrics.loc["fast_legitimate_motion", "false_correction_rate"] == 0.0
    assert "recovery_frames" in metrics.columns
    assert "raw_jump_count" in metrics.columns
    assert "corrected_jump_count" in metrics.columns
    assert (tmp_path / "successful_corrections.csv").is_file()
    assert (tmp_path / "failed_corrections.csv").is_file()
