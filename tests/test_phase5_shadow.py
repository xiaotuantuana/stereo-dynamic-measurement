from __future__ import annotations

import pandas as pd

from experiment.validation.phase5_shadow import _mode_summary


def test_mode_summary_reports_runtime_and_jump_metrics() -> None:
    frame = pd.DataFrame([
        {"mode": "M1", "final_valid": True, "final_error_mm": 5.0, "jump_mm": None},
        {"mode": "M1", "final_valid": True, "final_error_mm": 15.0, "jump_mm": 30.0},
    ])

    summary = _mode_summary(frame, {"M1": 2.0}).iloc[0]

    assert summary["runtime_s"] == 2.0
    assert summary["throughput_results_per_s"] == 1.0
    assert summary["jump_count"] == 1
    assert summary["large_jump_count"] == 1
