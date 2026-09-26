from __future__ import annotations

import pandas as pd

from experiment.validation.phase4_8 import (
    evaluate_persistence_acceptance,
    persistence_prediction_xyz_mm,
)


def test_persistence_prediction_is_exactly_the_latest_trusted_coordinate() -> None:
    assert persistence_prediction_xyz_mm([
        (0.0, 0.0, 2000.0),
        (1.0, 2.0, 2001.0),
    ]) == (1.0, 2.0, 2001.0)


def test_persistence_acceptance_requires_all_four_frozen_conditions() -> None:
    trace = pd.DataFrame([
        {"arm": "injected", "p0_outcome": "HARMFUL", "p1_outcome": "HARMFUL", "p0_improvement_mm": -10.0, "p1_improvement_mm": -5.0},
        {"arm": "injected", "p0_outcome": "HARMFUL", "p1_outcome": "BENEFICIAL", "p0_improvement_mm": -10.0, "p1_improvement_mm": 2.0},
        {"arm": "injected", "p0_outcome": "HARMFUL", "p1_outcome": "BENEFICIAL", "p0_improvement_mm": -10.0, "p1_improvement_mm": 3.0},
        {"arm": "injected", "p0_outcome": "BENEFICIAL", "p1_outcome": "BENEFICIAL", "p0_improvement_mm": 1.0, "p1_improvement_mm": 4.0},
        {"arm": "clean", "p0_outcome": "HARMFUL", "p1_outcome": "HARMFUL", "p0_improvement_mm": -2.0, "p1_improvement_mm": -1.0},
    ])

    result = evaluate_persistence_acceptance(trace)

    assert result["accepted"] is True
    assert result["injected_harmful_reduction"] == 2


def test_persistence_is_rejected_when_clean_harmful_increases() -> None:
    trace = pd.DataFrame([
        {"arm": "injected", "p0_outcome": "HARMFUL", "p1_outcome": "BENEFICIAL", "p0_improvement_mm": -4.0, "p1_improvement_mm": 4.0},
        {"arm": "injected", "p0_outcome": "HARMFUL", "p1_outcome": "BENEFICIAL", "p0_improvement_mm": -4.0, "p1_improvement_mm": 4.0},
        {"arm": "clean", "p0_outcome": "HARMFUL", "p1_outcome": "HARMFUL", "p0_improvement_mm": -1.0, "p1_improvement_mm": -1.0},
        {"arm": "clean", "p0_outcome": "NEUTRAL", "p1_outcome": "HARMFUL", "p0_improvement_mm": 0.0, "p1_improvement_mm": -1.0},
    ])

    result = evaluate_persistence_acceptance(trace)

    assert result["accepted"] is False
    assert result["clean_harmful_not_increased"] is False
