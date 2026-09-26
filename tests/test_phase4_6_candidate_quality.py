from __future__ import annotations

import pandas as pd

from experiment.validation.phase4_6 import classify_i2_recovery, phase4_6_subset


def _row(
    frame_index: int,
    *,
    phase: str,
    i1_valid: bool,
    i2_state: str,
) -> dict[str, object]:
    return {
        "case_id": "P45-01",
        "point_id": "P1",
        "frame_index": frame_index,
        "frame_phase": phase,
        "i1_valid": i1_valid,
        "i2_state": i2_state,
    }


def test_phase4_6_split_is_deterministic_and_keeps_a_pair_in_one_subset() -> None:
    first = phase4_6_subset("P45-20")

    assert first in {"development", "holdout"}
    assert phase4_6_subset("P45-20") == first


def test_phase4_6_frozen_split_has_13_development_and_19_holdout_cases() -> None:
    subsets = [phase4_6_subset(f"P45-{index:02d}") for index in range(1, 33)]

    assert subsets.count("development") == 13
    assert subsets.count("holdout") == 19


def test_recovery_classifies_invalid_i1_after_exposure_separately() -> None:
    audit = pd.DataFrame([
        _row(0, phase="exposure", i1_valid=True, i2_state="QUARANTINED"),
        _row(1, phase="recovery", i1_valid=False, i2_state="QUARANTINED"),
        _row(2, phase="recovery", i1_valid=False, i2_state="QUARANTINED"),
    ])

    record = classify_i2_recovery(audit).iloc[0]

    assert record["recovery_class"] == "I1_BASELINE_NOT_RECOVERED"
    assert bool(record["i2_recovery_evaluable"]) is False


def test_recovery_requires_exactly_two_valid_clean_observations_before_normal() -> None:
    audit = pd.DataFrame([
        _row(0, phase="exposure", i1_valid=True, i2_state="QUARANTINED"),
        _row(1, phase="recovery", i1_valid=True, i2_state="RECOVERY"),
        _row(2, phase="recovery", i1_valid=True, i2_state="NORMAL"),
    ])

    record = classify_i2_recovery(audit).iloc[0]

    assert record["recovery_class"] == "I2_RECOVERY_CONFIRMED"
    assert bool(record["i2_recovery_evaluable"]) is True
    assert record["valid_clean_observations_to_normal"] == 2


def test_recovery_failure_is_only_recorded_after_two_valid_clean_observations() -> None:
    audit = pd.DataFrame([
        _row(0, phase="exposure", i1_valid=True, i2_state="QUARANTINED"),
        _row(1, phase="recovery", i1_valid=True, i2_state="RECOVERY"),
        _row(2, phase="recovery", i1_valid=True, i2_state="QUARANTINED"),
    ])

    record = classify_i2_recovery(audit).iloc[0]

    assert record["recovery_class"] == "I2_RECOVERY_FAILURE"
    assert bool(record["i2_recovery_evaluable"]) is True
