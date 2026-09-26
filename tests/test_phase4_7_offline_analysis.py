from __future__ import annotations

import pandas as pd

from experiment.validation.phase4_7 import (
    bounded_candidate_xyz_mm,
    classify_candidate_outcome,
    classify_stale_prediction_recovery,
)


def test_c1_moves_toward_prediction_by_no_more_than_the_fixed_25_mm_cap() -> None:
    candidate, magnitude = bounded_candidate_xyz_mm(
        raw_xyz_mm=(0.0, 0.0, 0.0),
        prediction_xyz_mm=(60.0, 0.0, 0.0),
        cap_mm=25.0,
    )

    assert candidate == (25.0, 0.0, 0.0)
    assert magnitude == 25.0


def test_c1_uses_prediction_when_the_raw_to_prediction_distance_is_inside_cap() -> None:
    candidate, magnitude = bounded_candidate_xyz_mm(
        raw_xyz_mm=(0.0, 0.0, 0.0),
        prediction_xyz_mm=(10.0, 0.0, 0.0),
        cap_mm=25.0,
    )

    assert candidate == (10.0, 0.0, 0.0)
    assert magnitude == 10.0


def test_candidate_outcome_is_evaluation_only_and_uses_raw_baseline_error() -> None:
    assert classify_candidate_outcome(raw_error_mm=20.0, candidate_error_mm=10.0) == "BENEFICIAL"
    assert classify_candidate_outcome(raw_error_mm=20.0, candidate_error_mm=30.0) == "HARMFUL"
    assert classify_candidate_outcome(raw_error_mm=20.0, candidate_error_mm=20.0) == "NEUTRAL"


def test_stale_prediction_requires_two_clean_consistent_observations_far_from_old_prediction() -> None:
    trace = pd.DataFrame([
        {"case_id": "P45-01", "point_id": "P1", "frame_index": 3, "i1_valid": True,
         "hard_failure": False, "confirmed_anomaly": False, "raw_xyz_mm": "[30.0, 0.0, 0.0]",
         "prediction_xyz_mm": "[0.0, 0.0, 0.0]"},
        {"case_id": "P45-01", "point_id": "P1", "frame_index": 4, "i1_valid": True,
         "hard_failure": False, "confirmed_anomaly": False, "raw_xyz_mm": "[35.0, 0.0, 0.0]",
         "prediction_xyz_mm": "[0.0, 0.0, 0.0]"},
    ])

    record = classify_stale_prediction_recovery(trace, consistency_mm=10.0).iloc[0]

    assert record["recovery_pattern"] == "STALE_PREDICTION"


def test_stale_prediction_rejects_an_anomalous_or_unstable_pair() -> None:
    trace = pd.DataFrame([
        {"case_id": "P45-01", "point_id": "P1", "frame_index": 3, "i1_valid": True,
         "hard_failure": False, "confirmed_anomaly": False, "raw_xyz_mm": "[30.0, 0.0, 0.0]",
         "prediction_xyz_mm": "[0.0, 0.0, 0.0]"},
        {"case_id": "P45-01", "point_id": "P1", "frame_index": 4, "i1_valid": True,
         "hard_failure": False, "confirmed_anomaly": True, "raw_xyz_mm": "[35.0, 0.0, 0.0]",
         "prediction_xyz_mm": "[0.0, 0.0, 0.0]"},
    ])

    record = classify_stale_prediction_recovery(trace, consistency_mm=10.0).iloc[0]

    assert record["recovery_pattern"] == "INSUFFICIENT_CLEAN_OBSERVATIONS"


def test_stale_prediction_offline_label_excludes_an_exposure_frame_from_clean_pair() -> None:
    trace = pd.DataFrame([
        {"case_id": "P45-01", "point_id": "P1", "frame_index": 3, "frame_phase": "exposure",
         "i1_valid": True, "hard_failure": False, "confirmed_anomaly": False,
         "raw_xyz_mm": "[30.0, 0.0, 0.0]", "prediction_xyz_mm": "[0.0, 0.0, 0.0]"},
        {"case_id": "P45-01", "point_id": "P1", "frame_index": 4, "frame_phase": "recovery",
         "i1_valid": True, "hard_failure": False, "confirmed_anomaly": False,
         "raw_xyz_mm": "[35.0, 0.0, 0.0]", "prediction_xyz_mm": "[0.0, 0.0, 0.0]"},
    ])

    record = classify_stale_prediction_recovery(trace, consistency_mm=10.0).iloc[0]

    assert record["recovery_pattern"] == "INSUFFICIENT_CLEAN_OBSERVATIONS"
