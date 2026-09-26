from __future__ import annotations

import numpy as np
import cv2

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


def _textured_stereo(disparity: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4242)
    left = cv2.GaussianBlur(rng.integers(0, 256, (140, 240), dtype=np.uint8), (3, 3), 0.5)
    transform = np.array([[1.0, 0.0, -disparity], [0.0, 1.0, 0.0]], dtype=np.float32)
    right = cv2.warpAffine(left, transform, (240, 140), borderMode=cv2.BORDER_REFLECT101)
    return left, right


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


def test_post_correction_safety_requires_an_accepted_i1_measurement_not_a_constant() -> None:
    pipeline = _pipeline()
    pipeline._process_enhanced_results([_enhanced_result(0, 0.0)])
    pipeline._process_enhanced_results([_enhanced_result(1, 0.0)])

    unaccepted = pipeline._process_enhanced_results([
        FramePointResult(
            **{
                **_enhanced_result(2, 0.080, anomaly=True).__dict__,
                "measurement_accepted_for_state": False,
            }
        )
    ])[0]

    assert unaccepted.candidate_safe is False
    assert "post_correction_unsafe" in unaccepted.candidate_safety_reasons
    assert unaccepted.committed_decision == "ACCEPT_WITH_WARNING"
    assert unaccepted.write_committed is False


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


def test_i3_structured_recommendation_does_not_mutate_measurement_result() -> None:
    result = _enhanced_result(2, 0.080, anomaly=True)
    before = result.as_csv_row()

    recommendation = _pipeline()._structured_i3_recommendation(result, hard_failure=False)

    assert recommendation.risk is I3Risk.WARNING
    assert result.as_csv_row() == before


def test_structured_occlusion_policy_keeps_baseline_without_explicit_permission() -> None:
    result = FramePointResult(**{
        **_enhanced_result(2, 0.080, anomaly=True).__dict__,
        "fault_class": "OCCLUSION",
        "fault_confidence": 0.65,
    })

    recommendation = TemporalStereoPipeline._structured_i3_recommendation(result, hard_failure=False)

    assert recommendation.risk is I3Risk.WARNING
    assert recommendation.action.value == "WARN"


def test_blocking_i3_recommendation_rejects_even_with_safe_i2_candidate() -> None:
    pipeline = _pipeline()
    pipeline._process_enhanced_results([_enhanced_result(0, 0.0)])
    pipeline._process_enhanced_results([_enhanced_result(1, 0.0)])

    blocked = pipeline._process_enhanced_results([
        FramePointResult(
            **{
                **_enhanced_result(2, 0.080, anomaly=True).__dict__,
                "fault_confidence": 0.95,
            }
        )
    ])[0]

    assert blocked.proposed_decision == "REJECT"
    assert blocked.committed_decision == "REJECT"
    assert blocked.write_committed is True
    assert blocked.final_valid is False
    assert blocked.status == "rejected"


def test_enhanced_exception_fallback_preserves_original_baseline_validity() -> None:
    pipeline = _pipeline()
    valid = _enhanced_result(0, 0.02)
    invalid = _result(status="lost", final_x_m=None, final_y_m=None, final_z_m=None)

    valid_fallback = pipeline._enhanced_fallback(valid, RuntimeError("boom"))
    invalid_fallback = pipeline._enhanced_fallback(invalid, RuntimeError("boom"))

    assert valid_fallback.final_xyz_m == valid.final_xyz_m
    assert valid_fallback.final_valid is True
    assert valid_fallback.committed_decision == "ACCEPT_WITH_WARNING"
    assert invalid_fallback.final_xyz_m is None
    assert invalid_fallback.final_valid is False
    assert invalid_fallback.status == "lost"


def test_full_enhanced_without_write_flag_matches_shadow_final_and_csv() -> None:
    left, right = _textured_stereo()
    q = np.array([[1, 0, 0, -120], [0, 1, 0, -70], [0, 0, 0, 100], [0, 0, 10, 0]], dtype=float)
    shadow = TemporalStereoPipeline("M3", q, "m")
    full_no_write = TemporalStereoPipeline(
        "M3", q, "m", experiment_authority=ExperimentAuthority.full_experiment(write_enabled=False)
    )

    # Keep the same one-point I1 input; no GUI or production caller receives authority.
    from stereo_research.models import PointSpec
    shadow_result = shadow.initialize(left, right, (PointSpec("P1", (150.0, 70.0)),), frame=0)[0]
    enhanced_result = full_no_write.initialize(left, right, (PointSpec("P1", (150.0, 70.0)),), frame=0)[0]

    assert enhanced_result.status == shadow_result.status
    assert enhanced_result.final_xyz_m == shadow_result.final_xyz_m
    assert set(enhanced_result.as_csv_row()) == set(shadow_result.as_csv_row())
    runtime_fields = {"flow_ms", "matching_ms", "total_ms"}
    for field, value in shadow_result.as_csv_row().items():
        if field not in runtime_fields:
            assert enhanced_result.as_csv_row()[field] == value
    assert enhanced_result.write_committed is False


def test_full_enhanced_smoke_produces_i1_i2_i3_and_final_diagnostics_per_frame() -> None:
    left, right = _textured_stereo()
    q = np.array([[1, 0, 0, -120], [0, 1, 0, -70], [0, 0, 0, 100], [0, 0, 10, 0]], dtype=float)
    pipeline = TemporalStereoPipeline(
        "M3", q, "m", experiment_authority=ExperimentAuthority.full_experiment(write_enabled=True)
    )
    from stereo_research.models import PointSpec
    points = (PointSpec("P1", (150.0, 70.0)),)

    first = pipeline.initialize(left, right, points, frame=0)[0]
    second = pipeline.step(left, right, frame=1)[0]

    for result in (first, second):
        assert result.i2_state
        assert result.i2_trusted_committed is not None
        assert result.i2_reason
        assert result.proposed_decision
        assert result.committed_decision
        assert result.result_source
        assert result.final_valid is not None
