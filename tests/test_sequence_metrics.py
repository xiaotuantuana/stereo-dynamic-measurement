from experiment.sequence.metrics import summarize_sequence_rows


def test_sequence_metrics_report_coverage_errors_jitter_and_tracking_loss() -> None:
    rows = [
        {"valid": True, "evaluable": True, "disparity_error": 0.5, "final_disparity": 8.0, "runtime_ms": 2.0},
        {"valid": True, "evaluable": True, "disparity_error": 4.0, "final_disparity": 9.0, "runtime_ms": 3.0},
        {"valid": False, "evaluable": False, "status": "lost", "runtime_ms": 1.0},
    ]
    summary = summarize_sequence_rows(rows)
    assert summary["coverage"] == 2 / 3
    assert summary["cer_3"] == 0.5
    assert summary["tracking_loss_rate"] == 1 / 3
    assert summary["frame_to_frame_jitter_px"] == 1.0
