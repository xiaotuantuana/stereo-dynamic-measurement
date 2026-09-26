from __future__ import annotations

import pandas as pd

from experiment.validation.phase5b_reporting import static_mode_summary


def test_static_mode_summary_reports_disparity_only_and_no_fake_3d() -> None:
    rows = pd.DataFrame([
        {"method": "M0", "valid": True, "disparity_error": 2.0, "runtime_ms": 10.0},
        {"method": "M0", "valid": False, "disparity_error": None, "runtime_ms": 10.0},
        {"method": "M1", "valid": True, "disparity_error": 1.0, "runtime_ms": 20.0},
        {"method": "M1", "valid": True, "disparity_error": 3.0, "runtime_ms": 20.0},
    ])

    summary = static_mode_summary(rows).set_index("method")

    assert summary.loc["M0", "coverage"] == 0.5
    assert summary.loc["M1", "disparity_rmse"] == (5.0 ** 0.5)
    assert summary.loc["M1", "depth_xyz_3d"] == "N/A:no_calibration_or_depth_gt"
