from __future__ import annotations

from stereo_research.final_arbitration import (
    CandidateSafety,
    ExperimentAuthority,
    FinalArbitrator,
    FinalDecision,
    I1BaselineView,
    I3Recommendation,
    I3Risk,
    ResultSource,
)
from stereo_research.models import FramePointResult, SystemMode


def _baseline_result(**overrides: object) -> FramePointResult:
    values: dict[str, object] = {
        "method": "research_full",
        "frame": 7,
        "point_id": "P1",
        "status": "valid",
        "final_x_m": 1.0,
        "final_y_m": 2.0,
        "final_z_m": 2.0,
        "distance_m": 3.0,
        "confidence": 0.9,
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


def test_accept_preserves_immutable_i1_baseline_values() -> None:
    result = _baseline_result()
    baseline = I1BaselineView.from_result(result)

    decision = FinalArbitrator().decide(
        baseline=baseline,
        candidate_safety=CandidateSafety.unavailable("no correction proposed"),
        diagnosis=I3Recommendation.normal(),
        authority=ExperimentAuthority.shadow(SystemMode.ENHANCED_SHADOW),
    )

    assert baseline.xyz_m == (1.0, 2.0, 2.0)
    assert decision.proposed_decision is FinalDecision.ACCEPT
    assert decision.committed_decision is FinalDecision.ACCEPT
    assert decision.result_source is ResultSource.I1_BASELINE
    assert decision.final_xyz_m == baseline.xyz_m
    assert not decision.write_committed


def test_full_experiment_commits_only_a_complete_safe_candidate() -> None:
    decision = FinalArbitrator().decide(
        baseline=I1BaselineView.from_result(_baseline_result()),
        candidate_safety=_safe_candidate(),
        diagnosis=I3Recommendation.normal(),
        authority=ExperimentAuthority.full_experiment(write_enabled=True),
    )

    assert decision.proposed_decision is FinalDecision.USE_CORRECTED
    assert decision.committed_decision is FinalDecision.USE_CORRECTED
    assert decision.write_committed
    assert decision.result_source is ResultSource.I2_CORRECTED
    assert decision.final_xyz_m == (1.5, 2.5, 2.5)


def test_full_enhanced_without_write_flag_is_shadow_equivalent() -> None:
    baseline = I1BaselineView.from_result(_baseline_result())
    decision = FinalArbitrator().decide(
        baseline=baseline,
        candidate_safety=_safe_candidate(),
        diagnosis=I3Recommendation.normal(),
        authority=ExperimentAuthority.full_experiment(write_enabled=False),
    )

    assert decision.proposed_decision is FinalDecision.USE_CORRECTED
    assert decision.committed_decision is FinalDecision.ACCEPT_WITH_WARNING
    assert decision.result_source is ResultSource.I1_BASELINE
    assert decision.final_xyz_m == baseline.xyz_m
    assert not decision.write_committed
    assert decision.authority.effective_mode is SystemMode.ENHANCED_SHADOW


def test_write_flag_outside_full_enhanced_has_no_final_authority() -> None:
    decision = FinalArbitrator().decide(
        baseline=I1BaselineView.from_result(_baseline_result()),
        candidate_safety=_safe_candidate(),
        diagnosis=I3Recommendation.normal(),
        authority=ExperimentAuthority(
            requested_mode=SystemMode.INNOVATION1,
            write_enabled=True,
            scope="experiment",
        ),
    )

    assert decision.proposed_decision is FinalDecision.USE_CORRECTED
    assert not decision.write_committed
    assert decision.committed_decision is FinalDecision.ACCEPT_WITH_WARNING


def test_structured_blocking_i3_policy_rejects_when_no_safe_candidate_exists() -> None:
    decision = FinalArbitrator().decide(
        baseline=I1BaselineView.from_result(_baseline_result()),
        candidate_safety=CandidateSafety.unavailable("candidate failed safety gate"),
        diagnosis=I3Recommendation(risk=I3Risk.BLOCKING, reason="stereo mismatch"),
        authority=ExperimentAuthority.full_experiment(write_enabled=True),
    )

    assert decision.proposed_decision is FinalDecision.REJECT
    assert decision.committed_decision is FinalDecision.REJECT
    assert decision.result_source is ResultSource.REJECTED
    assert decision.final_xyz_m is None
    assert not decision.final_valid


def test_candidate_safety_is_derived_and_audits_failed_gates() -> None:
    safety = CandidateSafety.evaluate(
        candidate_xyz_m=(float("nan"), 1.0, 2.0),
        geometry_valid=False,
        correction_authorized=False,
        evidence_sufficient=False,
        post_correction_safe=False,
        i2_state="NORMAL",
    )

    assert not safety.safe
    assert "candidate_not_finite" in safety.failed_reasons
    assert "geometry_invalid" in safety.failed_reasons
    assert "correction_unauthorized" in safety.failed_reasons
