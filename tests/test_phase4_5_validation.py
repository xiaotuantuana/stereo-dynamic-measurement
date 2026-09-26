from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment.validation.phase4_5 import (
    _i2_candidate_xyz_m,
    build_phase4_5_case_windows,
    run_phase4_5_validation,
)
from stereo_research.models import FramePointResult


_MANIFEST = Path("results/phase3/stateful_innovation1_500/M3/sequence_manifest.json")


def test_phase4_5_small_window_runs_m1_m2_m3_and_exports_required_audit(tmp_path: Path) -> None:
    summary = run_phase4_5_validation(tmp_path, manifest_path=_MANIFEST, case_windows=build_phase4_5_case_windows(_MANIFEST)[:1])

    assert summary["allow_2000"] == "NO"
    for name in (
        "case_manifest.csv", "frame_level_results.csv", "arbitration_audit.csv",
        "scenario_summary.csv", "gate_summary.csv",
    ):
        assert (tmp_path / name).is_file()
    manifest = pd.read_csv(tmp_path / "case_manifest.csv")
    frame = pd.read_csv(tmp_path / "frame_level_results.csv")
    audit = pd.read_csv(tmp_path / "arbitration_audit.csv")
    assert set(manifest["arm"]) == {"clean", "injected"}
    assert {"M1", "M2", "M3"} == set(frame["mode"])
    assert {
        "proposed_decision", "committed_decision", "write_committed", "final_owner",
        "shadow_candidate_xyz_m", "i2_candidate_xyz_m",
        "i2_candidate_error_mm", "candidate_error_improvement_mm",
        "runtime_i1_confidence", "runtime_measurement_quality_score",
        "runtime_texture_std", "runtime_uniqueness_margin",
        "runtime_lr_error_px", "runtime_flow_fb_error_px",
        "runtime_prediction_residual_px", "runtime_c_phy",
    } <= set(audit.columns)
    assert "candidate_xyz_m" not in audit.columns
    assert not audit.loc[audit["mode"].isin(["M1", "M2"]), "write_committed"].astype(bool).any()

    keys = ["case_id", "arm", "frame_index", "point_id"]
    parity_fields = ["i1_status", "i2_state", "candidate_safe", "proposed_decision", "i3_fault"]
    m2 = audit.loc[audit["mode"] == "M2", keys + parity_fields].sort_values(keys).reset_index(drop=True)
    m3 = audit.loc[audit["mode"] == "M3", keys + parity_fields].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(m2, m3, check_dtype=False)


def test_phase4_5_false_correction_counts_only_written_use_corrected() -> None:
    from experiment.validation.phase4_5 import count_false_final_corrections

    audit = pd.DataFrame([
        {"arm": "clean", "mode": "M3", "committed_decision": "REJECT", "write_committed": True},
        {"arm": "clean", "mode": "M3", "committed_decision": "ACCEPT_WITH_WARNING", "write_committed": False},
        {"arm": "clean", "mode": "M3", "committed_decision": "USE_CORRECTED", "write_committed": True},
    ])

    assert count_false_final_corrections(audit) == 1


def test_phase4_5_audit_distinguishes_legacy_shadow_candidate_from_actual_i2_candidate() -> None:
    result = FramePointResult(
        method="research_full",
        frame=1,
        point_id="P1",
        status="valid",
        candidate_corrected_x_m=1.0,
        candidate_corrected_y_m=2.0,
        candidate_corrected_z_m=3.0,
        i2_prediction_x_m=4.0,
        i2_prediction_y_m=5.0,
        i2_prediction_z_m=6.0,
        i2_correction_applied=True,
    )

    assert result.candidate_xyz_m == (1.0, 2.0, 3.0)
    assert _i2_candidate_xyz_m(result) == (4.0, 5.0, 6.0)


def test_phase4_5_audit_does_not_call_a_prediction_an_i2_candidate_when_i2_abstained() -> None:
    result = FramePointResult(
        method="research_full",
        frame=1,
        point_id="P1",
        status="valid",
        i2_prediction_x_m=4.0,
        i2_prediction_y_m=5.0,
        i2_prediction_z_m=6.0,
        i2_correction_applied=False,
    )

    assert _i2_candidate_xyz_m(result) is None
