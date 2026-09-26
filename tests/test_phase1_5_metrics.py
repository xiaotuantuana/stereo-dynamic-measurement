from __future__ import annotations

from experiment.analysis.thresholds import confidence_threshold_curve
from experiment.benchmark.metrics import summarize_rows


def test_summary_reports_coverage_valid_accuracy_and_catastrophic_rates() -> None:
    summary = summarize_rows([
        {"valid": True, "disparity_error": 1.0, "runtime_ms": 5.0},
        {"valid": True, "disparity_error": 12.0, "runtime_ms": 5.0},
        {"valid": False, "disparity_error": None, "runtime_ms": 5.0},
    ])

    assert summary["coverage"] == 2 / 3
    assert summary["valid_mae"] == 6.5
    assert summary["cer_3"] == 0.5
    assert summary["cer_5"] == 0.5
    assert summary["cer_10"] == 0.5


def test_confidence_curve_never_counts_rejected_rows_as_accurate() -> None:
    curve = confidence_threshold_curve([
        {"calibrated_confidence": 0.9, "potential_error": 0.5, "runtime_ms": 5.0},
        {"calibrated_confidence": 0.6, "potential_error": 15.0, "runtime_ms": 5.0},
        {"calibrated_confidence": 0.2, "potential_error": 0.2, "runtime_ms": 5.0},
    ], thresholds=[0.0, 0.7])

    assert curve[0]["valid_rate"] == 1.0
    assert curve[0]["cer_10"] == 1 / 3
    assert curve[1]["valid_rate"] == 1 / 3
    assert curve[1]["cer_10"] == 0.0

