from __future__ import annotations

import pandas as pd

from experiment.validation.phase4 import run_phase4_innovation2_validation
from experiment.validation.phase4_full import run_phase4_full_smoke


def test_phase4_innovation2_core_validation_exports_six_scenarios_and_trace(tmp_path) -> None:
    summary = run_phase4_innovation2_validation(tmp_path, frame_count=48, seed=20260827)

    core = pd.read_csv(tmp_path / "innovation2_core_validation.csv")
    trace = pd.read_csv(tmp_path / "innovation2_frame_trace.csv")

    assert summary["scenario_count"] == 6
    assert set(core["scenario"]) == {
        "stable", "single_jump", "continuous_outlier", "slow_drift",
        "fast_legitimate_motion", "noise",
    }
    assert list(core.columns) == [
        "scenario", "raw_rmse", "corrected_rmse", "raw_jump_count",
        "corrected_jump_count", "jump_suppression", "false_correction", "recovery_frames",
    ]
    assert (core["corrected_jump_count"] <= core["raw_jump_count"]).all()
    assert {"ground_truth_y_mm", "raw_y_mm", "prediction_y_mm", "corrected_y_mm",
            "trusted_state_y_mm", "decision", "C_phy", "correction_trigger", "state"} <= set(trace.columns)


def test_phase4_full_smoke_exports_comparison_and_arbitration_audit(tmp_path) -> None:
    summary = run_phase4_full_smoke(tmp_path, frame_count=4)

    comparison = pd.read_csv(tmp_path / "full_enhanced_comparison.csv")
    audit = pd.read_csv(tmp_path / "final_arbitration_audit.csv")

    assert summary["stateful_protocol"] is True
    assert list(comparison["mode"]) == ["M0", "M1", "M2", "M3"]
    assert list(comparison.columns) == [
        "mode", "coverage", "mae", "rmse", "cer10", "jitter", "jump_count", "false_correction", "runtime",
    ]
    assert {"frame", "I1_status", "I2_status", "I3_status", "decision", "result_source", "final_valid"} <= set(audit.columns)
