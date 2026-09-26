from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo_dynamic_measurement.innovation1.adaptive_search import AdaptiveSearchConfig, compute_search_radius
from stereo_dynamic_measurement.innovation1.stereo_matcher import LocalStereoMatcher
from stereo_dynamic_measurement.innovation1.subpixel_refiner import refine_parabolic
from stereo_dynamic_measurement.innovation1.confidence import neighbor_disparity_residual


def test_adaptive_search_radius_responds_to_evidence_and_precision_target() -> None:
    config = AdaptiveSearchConfig(min_radius_px=2, max_radius_px=32)
    stable = compute_search_radius(config, gradient_quality=0.95, fb_error_px=0.05, prediction_residual_px=0.05, previous_confidence=0.95, required_disparity_precision_px=0.15)
    degraded = compute_search_radius(config, gradient_quality=0.15, fb_error_px=1.5, prediction_residual_px=2.0, previous_confidence=0.2, required_disparity_precision_px=0.01)

    assert config.min_radius_px <= stable.radius_px < degraded.radius_px <= config.max_radius_px
    assert "low_texture" in degraded.reasons and "target_precision" in degraded.reasons


def test_local_zncc_gradient_matcher_recovers_local_disparity() -> None:
    rng = np.random.default_rng(12)
    left = cv2.GaussianBlur(rng.integers(0, 256, (100, 160), dtype=np.uint8), (3, 3), 0.7)
    right = cv2.warpAffine(left, np.array([[1, 0, -12], [0, 1, 0]], np.float32), (160, 100), borderMode=cv2.BORDER_REFLECT101)
    matcher = LocalStereoMatcher(patch_size=15)

    result = matcher.match(left, right, left_point=(90.0, 50.0), predicted_disparity_px=10.0, search_radius_px=5)

    assert result.valid
    assert result.integer_disparity_px == pytest.approx(12.0)
    assert np.isfinite(result.match_cost)
    assert result.cost_curve[12.0] <= result.cost_curve[11.0]


def test_subpixel_parabola_is_finite_and_handles_flat_or_boundary_costs() -> None:
    refined = refine_parabolic(12.0, cost_minus=1.2, cost_center=1.0, cost_plus=1.1, at_boundary=False)
    flat = refine_parabolic(12.0, cost_minus=1.0, cost_center=1.0, cost_plus=1.0, at_boundary=False)
    boundary = refine_parabolic(12.0, cost_minus=1.2, cost_center=1.0, cost_plus=1.1, at_boundary=True)

    assert refined.valid and 12.0 < refined.disparity_px < 12.5
    assert flat.disparity_px == 12.0 and not flat.valid
    assert boundary.disparity_px == 12.0 and not boundary.valid


def test_neighbor_residual_uses_median_and_is_finite() -> None:
    residual = neighbor_disparity_residual(12.0, [11.9, 12.1, 12.0, 30.0])

    assert residual == pytest.approx(0.05)
