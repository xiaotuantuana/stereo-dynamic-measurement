from __future__ import annotations

import numpy as np

from stereo_research.final_arbitration import (
    CandidateSafety,
    ExperimentAuthority,
    FinalArbitrator,
    I1BaselineView,
    I3Recommendation,
    I3Risk,
)
from stereo_research.models import FramePointResult
from stereo_research.pipeline import TemporalStereoPipeline


def _pipeline() -> TemporalStereoPipeline:
    return TemporalStereoPipeline(
        "M3",
        np.array([[1, 0, 0, -120], [0, 1, 0, -70], [0, 0, 0, 100], [0, 0, 10, 0]], dtype=float),
        "m",
        experiment_authority=ExperimentAuthority.full_experiment(write_enabled=True),
    )


def _result(**overrides: object) -> FramePointResult:
    values: dict[str, object] = {
        "method": "research_full",
        "frame": 3,
        "point_id": "P1",
        "status": "valid",
        "final_x_m": 1.0,
        "final_y_m": 2.0,
        "final_z_m": 2.0,
        "distance_m": 3.0,
    }
    values.update(overrides)
    return FramePointResult(**values)


def _safe_candidate() -> CandidateSafety:
    return CandidateSafety.evaluate(
        candidate_xyz_m=(1.5, 2.5, 2.5),
        geometry_valid=True,
        correction_authorized=True,
        evidence_sufficient=True,
        post_correction_safe=True,
        i2_state="QUARANTINED",
    )


def _enhanced_result(frame: int, y_m: float, *, anomaly: bool = False) -> FramePointResult:
    return FramePointResult(
        method="research_full",
        frame=frame,
        point_id="P1",
        status="valid",
        final_x_m=0.0,
        final_y_m=y_m,
        final_z_m=2.0,
        distance_m=float(np.sqrt(y_m ** 2 + 4.0)),
        c_phy=0.1 if anomaly else 0.95,
        c_phy_valid=True,
        fault_class="STEREO_MISMATCH" if anomaly else "NORMAL",
        fault_confidence=0.2 if anomaly else 0.0,
    )


def test_apply_final_decision_commits_corrected_values_and_audit_atomically() -> None:
    pipeline = _pipeline()
    result = _result()
    decision = FinalArbitrator().decide(
        baseline=I1BaselineView.from_result(result),
        candidate_safety=_safe_candidate(),
        diagnosis=I3Recommendation.normal(),
        authority=ExperimentAuthority.full_experiment(write_enabled=True),
    )

    committed = pipeline._apply_final_decision(result, decision)

    assert committed.final_xyz_m == (1.5, 2.5, 2.5)
    assert committed.final_distance_m == np.sqrt(14.75)
    assert committed.final_valid is True
    assert committed.result_source == "I2_CORRECTED"
    assert committed.committed_decision == "USE_CORRECTED"
    assert committed.write_committed is True
    assert committed.status == "valid"


def test_reject_clears_final_measurement_without_mutating_i1_baseline_fields() -> None:
    pipeline = _pipeline()
    result = _result()
    decision = FinalArbitrator().decide(
        baseline=I1BaselineView.from_result(result),
        candidate_safety=CandidateSafety.unavailable("unsafe"),
        diagnosis=I3Recommendation(risk=I3Risk.BLOCKING, reason="unrecoverable fault"),
        authority=ExperimentAuthority.full_experiment(write_enabled=True),
    )

    committed = pipeline._apply_final_decision(result, decision)

    assert committed.final_xyz_m is None
    assert committed.final_distance_m is None
    assert committed.final_valid is False
    assert committed.status == "rejected"
    assert committed.distance_m == result.distance_m
    assert committed.x_m == result.x_m


def test_experimental_audit_fields_do_not_expand_the_legacy_csv() -> None:
    pipeline = _pipeline()
    result = _result()
    decision = FinalArbitrator().decide(
        baseline=I1BaselineView.from_result(result),
        candidate_safety=_safe_candidate(),
        diagnosis=I3Recommendation.normal(),
        authority=ExperimentAuthority.full_experiment(write_enabled=True),
    )

    row = pipeline._apply_final_decision(result, decision).as_csv_row()

    assert "proposed_decision" not in row
    assert "committed_decision" not in row
    assert "write_committed" not in row
    assert "result_source" not in row


def test_full_enhanced_finalizer_uses_trusted_prediction_for_safe_candidate() -> None:
    pipeline = _pipeline()
    pipeline._process_enhanced_results([_enhanced_result(0, 0.0)])
    pipeline._process_enhanced_results([_enhanced_result(1, 0.0)])

    corrected = pipeline._process_enhanced_results([_enhanced_result(2, 0.080, anomaly=True)])[0]

    assert corrected.final_y_m == 0.0
    assert corrected.result_source == "I2_CORRECTED"
    assert corrected.proposed_decision == "USE_CORRECTED"
    assert corrected.committed_decision == "USE_CORRECTED"
    assert corrected.write_committed is True


def test_shadow_authority_keeps_baseline_final_while_recording_proposal() -> None:
    pipeline = TemporalStereoPipeline(
        "M3",
        np.array([[1, 0, 0, -120], [0, 1, 0, -70], [0, 0, 0, 100], [0, 0, 10, 0]], dtype=float),
        "m",
        experiment_authority=ExperimentAuthority.full_experiment(write_enabled=False),
    )
    pipeline._process_enhanced_results([_enhanced_result(0, 0.0)])
    pipeline._process_enhanced_results([_enhanced_result(1, 0.0)])

    shadow = pipeline._process_enhanced_results([_enhanced_result(2, 0.080, anomaly=True)])[0]

    assert shadow.final_y_m == 0.080
    assert shadow.status == "valid"
    assert shadow.proposed_decision == "USE_CORRECTED"
    assert shadow.committed_decision == "ACCEPT_WITH_WARNING"
    assert shadow.write_committed is False
