from __future__ import annotations

import pandas as pd

from stereo_dynamic_measurement.innovation2.physics_validation import PhysicsValidationConfig, run_physics_validation


def test_p3_jump_lowers_confidence_but_missing_visual_evidence_blocks_oracle_correction(tmp_path) -> None:
    result = run_physics_validation(PhysicsValidationConfig(duration_s=3.0, fs_hz=100.0, jump_frame=150), tmp_path)
    trajectories = pd.read_csv(result.raw_vs_corrected_csv)
    confidence = pd.read_csv(result.physics_confidence_csv)
    p3_jump = trajectories[(trajectories.point_id == "P3") & (trajectories.frame == 150)].iloc[0]
    other = confidence[(confidence.point_id != "P3") & (confidence.frame == 150)].C_phy.mean()
    p3 = confidence[(confidence.point_id == "P3") & (confidence.frame == 150)].C_phy.item()

    assert set(result.paths) == {"raw_vs_corrected_csv", "physics_confidence_csv", "trajectory_comparison_png", "psd_comparison_png"}
    assert p3 < other - 0.2
    assert p3_jump.corrected_Y_mm == p3_jump.raw_Y_mm
    assert not bool(p3_jump.correction_applied)
    assert p3_jump.correction_reason == "insufficient_evidence_for_safe_correction"
    assert result.trajectory_comparison_png.exists() and result.psd_comparison_png.exists()
