from __future__ import annotations

import numpy as np
import pytest

from stereo_dynamic_measurement.innovation1.precision_planner import required_disparity_precision


def test_depth_disparity_round_trip_uses_pixel_focal_length_and_mm_baseline() -> None:
    """The pinhole depth equation must preserve the declared mm/pixel units."""
    depth_mm = 2_000.0
    baseline_mm = 180.0
    focal_length_px = 1_200.0

    disparity_px = focal_length_px * baseline_mm / depth_mm
    reconstructed_depth_mm = focal_length_px * baseline_mm / disparity_px

    assert disparity_px == pytest.approx(108.0)
    assert reconstructed_depth_mm == pytest.approx(depth_mm, abs=1e-12)


def test_required_disparity_precision_matches_first_order_depth_propagation() -> None:
    sigma_d = required_disparity_precision(
        distance_mm=2_000.0,
        baseline_mm=180.0,
        focal_length_px=1_200.0,
        target_depth_error_mm=1.0,
    )

    expected = 1.0 * 1_200.0 * 180.0 / (2_000.0**2)
    assert np.isfinite(sigma_d)
    assert sigma_d == pytest.approx(expected)
