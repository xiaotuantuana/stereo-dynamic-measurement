from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment.analysis.failure_analysis import (
    CostCurveDiagnostics,
    calibrated_initial_confidence,
    classify_rejection,
    summarize_cost_curve,
)


def test_cost_curve_reports_non_adjacent_uniqueness_without_using_gt() -> None:
    diagnostics = summarize_cost_curve(
        disparities=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        costs=np.array([0.40, 0.20, 0.19, 0.31, 0.25]),
        predicted_disparity=2.8,
    )

    assert diagnostics.best_disparity == 3.0
    assert diagnostics.second_best_disparity == 5.0
    assert diagnostics.uniqueness_margin == 0.06
    assert diagnostics.candidate_count == 5
    assert diagnostics.predicted_cost == 0.19


def test_initial_confidence_is_low_for_ambiguous_candidate_even_with_good_lr() -> None:
    diagnostics = CostCurveDiagnostics(
        best_disparity=77.0,
        best_cost=0.18,
        second_best_disparity=79.0,
        second_best_cost=0.209,
        uniqueness_margin=0.029,
        predicted_cost=0.18,
        predicted_best_disagreement=0.1,
        candidate_count=127,
    )

    score = calibrated_initial_confidence(
        texture_std=38.0,
        lr_error_px=0.1,
        diagnostics=diagnostics,
        texture_reference=10.0,
        margin_reference=0.05,
        lr_reference=1.0,
        max_photo_cost=0.45,
    )

    assert 0.0 <= score < 0.75


def test_rejection_quality_uses_gt_only_after_runtime_decision() -> None:
    assert classify_rejection(forced_error=12.0) == "Good Reject"
    assert classify_rejection(forced_error=0.5) == "Potential Over-Reject"
    assert classify_rejection(forced_error=None) == "Unassessable"

