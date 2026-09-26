from __future__ import annotations

from stereo_research.enhanced_processor import (
    EnhancedPointProcessor,
    I2Evidence,
    I2State,
    TimedObservation,
)


def _observation(
    frame: int,
    y_mm: float,
    *,
    confirmed_anomaly: bool = False,
    legitimate_motion: bool = False,
    hard_failure: bool = False,
    geometry_valid: bool | None = None,
) -> TimedObservation:
    return TimedObservation(
        frame=frame,
        timestamp_s=float(frame) / 10.0,
        xyz_mm=(0.0, y_mm, 2000.0),
        evidence=I2Evidence(
            confirmed_anomaly=confirmed_anomaly,
            legitimate_motion=legitimate_motion,
            hard_failure=hard_failure,
            geometry_valid=(not hard_failure) if geometry_valid is None else geometry_valid,
            evidence_sufficient=not hard_failure,
            post_correction_safe=not hard_failure,
        ),
    )


def test_no_correction_feedback_loop() -> None:
    processor = EnhancedPointProcessor(max_authorized_correction_mm=100.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    outlier = processor.process(_observation(2, 80.0, confirmed_anomaly=True))
    first_clean = processor.process(_observation(3, 0.0))
    second_clean = processor.process(_observation(4, 0.0))

    assert outlier.correction_applied
    assert outlier.corrected_xyz_mm == (0.0, 0.0, 2000.0)
    assert first_clean.correction_applied is False
    assert first_clean.state is I2State.RECOVERY
    assert second_clean.correction_applied is False
    assert second_clean.state is I2State.NORMAL
    assert [item.frame for item in processor.trusted_history] == [0, 1, 3, 4]


def test_return_to_trusted_prediction_recovers_even_if_legacy_evidence_still_flags_anomaly() -> None:
    processor = EnhancedPointProcessor(max_authorized_correction_mm=100.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    processor.process(_observation(2, 80.0, confirmed_anomaly=True))

    return_frame = processor.process(_observation(3, 0.0, confirmed_anomaly=True))

    assert return_frame.state is I2State.RECOVERY
    assert return_frame.correction_applied is False
    assert return_frame.corrected_xyz_mm is None


def test_continuous_outlier_stays_quarantined_when_legacy_evidence_temporarily_weakens() -> None:
    processor = EnhancedPointProcessor(max_authorized_correction_mm=100.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    processor.process(_observation(2, 60.0, confirmed_anomaly=True))

    unresolved = processor.process(_observation(3, 60.0, confirmed_anomaly=False))

    assert unresolved.state is I2State.QUARANTINED
    assert unresolved.trusted_committed is False
    assert [item.frame for item in processor.trusted_history] == [0, 1]


def test_raw_and_corrected_histories_never_drive_prediction() -> None:
    processor = EnhancedPointProcessor(max_authorized_correction_mm=100.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    processor.process(_observation(2, 80.0, confirmed_anomaly=True))

    outcome = processor.process(_observation(3, 0.0))

    assert processor.raw_history[-1].xyz_mm == (0.0, 0.0, 2000.0)
    assert processor.corrected_history[-2] == (0.0, 0.0, 2000.0)
    assert outcome.prediction_xyz_mm == (0.0, 0.0, 2000.0)
    assert [item.frame for item in processor.trusted_history] == [0, 1]


def test_recovery_keeps_original_timestamps_and_never_rewrites_prior_outputs() -> None:
    processor = EnhancedPointProcessor(max_authorized_correction_mm=100.0)
    historical = [processor.process(_observation(frame, 0.0)) for frame in (0, 1)]
    processor.process(_observation(2, 80.0, confirmed_anomaly=True))
    processor.process(_observation(3, 0.0))
    processor.process(_observation(4, 0.0))

    assert [(item.frame, item.timestamp_s) for item in processor.trusted_history] == [
        (0, 0.0), (1, 0.1), (3, 0.3), (4, 0.4)
    ]
    assert historical[0].state is I2State.NORMAL
    assert historical[1].trusted_committed


def test_fast_legitimate_motion_is_trusted_instead_of_smoothed() -> None:
    processor = EnhancedPointProcessor()

    outcomes = [
        processor.process(_observation(frame, y, legitimate_motion=True))
        for frame, y in enumerate((0.0, 10.0, 25.0, 45.0))
    ]

    assert all(outcome.state is I2State.NORMAL for outcome in outcomes)
    assert all(outcome.correction_applied is False for outcome in outcomes)
    assert [item.xyz_mm[1] for item in processor.trusted_history] == [0.0, 10.0, 25.0, 45.0]


def test_hard_failure_quarantines_without_secondary_evidence() -> None:
    processor = EnhancedPointProcessor()
    processor.process(_observation(0, 0.0))

    outcome = processor.process(_observation(1, float("nan"), hard_failure=True))

    assert outcome.state is I2State.QUARANTINED
    assert outcome.corrected_xyz_mm is None
    assert outcome.candidate_safety.safe is False
    assert "hard_failure" in outcome.candidate_safety.failed_reasons
    assert [item.frame for item in processor.trusted_history] == [0]


def test_large_correction_is_complete_or_abstains_never_partial() -> None:
    processor = EnhancedPointProcessor(max_authorized_correction_mm=25.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))

    outcome = processor.process(_observation(2, 80.0, confirmed_anomaly=True))

    assert outcome.state is I2State.QUARANTINED
    assert outcome.correction_applied is False
    assert outcome.corrected_xyz_mm is None
    assert "correction_unauthorized" in outcome.candidate_safety.failed_reasons


def test_recovery_consistency_keeps_the_existing_10_mm_inclusive_boundary() -> None:
    at_boundary = EnhancedPointProcessor(recovery_consistency_mm=10.0)
    at_boundary.process(_observation(0, 0.0))
    at_boundary.process(_observation(1, 0.0))
    at_boundary.process(_observation(2, 80.0, confirmed_anomaly=True))

    boundary = at_boundary.process(_observation(3, 10.0))

    beyond_boundary = EnhancedPointProcessor(recovery_consistency_mm=10.0)
    beyond_boundary.process(_observation(0, 0.0))
    beyond_boundary.process(_observation(1, 0.0))
    beyond_boundary.process(_observation(2, 80.0, confirmed_anomaly=True))
    beyond = beyond_boundary.process(_observation(3, 10.001))

    assert boundary.state is I2State.RECOVERY
    assert beyond.state is I2State.QUARANTINED


def test_stale_predictor_reacquires_after_two_stable_valid_observations() -> None:
    processor = EnhancedPointProcessor(recovery_consistency_mm=10.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    anomalous = processor.process(_observation(2, 80.0, confirmed_anomaly=True))
    first = processor.process(_observation(3, 30.0))
    reacquired = processor.process(_observation(4, 34.0))

    assert anomalous.corrected_xyz_mm == (0.0, 0.0, 2000.0)
    assert first.state is I2State.QUARANTINED
    assert first.reason == "reacquisition_first_clean_observation"
    assert reacquired.state is I2State.NORMAL
    assert reacquired.reason == "reacquisition_confirmed"
    assert [item.frame for item in processor.trusted_history] == [3, 4]


def test_stale_predictor_never_reacquires_from_one_valid_observation() -> None:
    processor = EnhancedPointProcessor(recovery_consistency_mm=10.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    processor.process(_observation(2, 80.0, confirmed_anomaly=True))

    first = processor.process(_observation(3, 30.0))

    assert first.state is I2State.QUARANTINED
    assert [item.frame for item in processor.trusted_history] == [0, 1]


def test_stale_predictor_never_reacquires_from_an_inconsistent_pair() -> None:
    processor = EnhancedPointProcessor(recovery_consistency_mm=10.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    processor.process(_observation(2, 80.0, confirmed_anomaly=True))
    processor.process(_observation(3, 30.0))

    rejected = processor.process(_observation(4, 45.0))

    assert rejected.state is I2State.QUARANTINED
    assert rejected.reason == "reacquisition_pair_inconsistent"
    assert [item.frame for item in processor.trusted_history] == [0, 1]


def test_stale_predictor_never_reacquires_a_confirmed_anomaly_or_hard_failure() -> None:
    processor = EnhancedPointProcessor(recovery_consistency_mm=10.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    processor.process(_observation(2, 80.0, confirmed_anomaly=True))
    anomaly = processor.process(_observation(3, 30.0, confirmed_anomaly=True))
    hard_failure = processor.process(_observation(4, float("nan"), hard_failure=True))

    assert anomaly.state is I2State.QUARANTINED
    assert hard_failure.state is I2State.QUARANTINED
    assert [item.frame for item in processor.trusted_history] == [0, 1]


def test_stale_predictor_never_reacquires_an_invalid_geometry_observation() -> None:
    processor = EnhancedPointProcessor(recovery_consistency_mm=10.0)
    processor.process(_observation(0, 0.0))
    processor.process(_observation(1, 0.0))
    processor.process(_observation(2, 80.0, confirmed_anomaly=True))

    invalid = processor.process(_observation(3, 30.0, geometry_valid=False))

    assert invalid.state is I2State.QUARANTINED
    assert invalid.reason == "quarantine_unresolved"
    assert [item.frame for item in processor.trusted_history] == [0, 1]
