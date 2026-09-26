from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo_dynamic_measurement.innovation1.gradient_quality import assess_image_quality
from stereo_dynamic_measurement.innovation1.motion_predictor import MotionPredictor, PointMeasurementState
from stereo_dynamic_measurement.innovation1.optical_flow import track_forward_backward
from stereo_dynamic_measurement.innovation1.precision_planner import recommend_baseline, required_disparity_precision


def test_image_quality_normalizes_texture_and_blur_evidence() -> None:
    textured = ((np.indices((80, 80)).sum(axis=0) // 8) % 2 * 255).astype(np.uint8)
    flat = np.full((80, 80), 128, dtype=np.uint8)

    textured_quality = assess_image_quality(textured)
    flat_quality = assess_image_quality(flat)

    assert all(0.0 <= value <= 1.0 for value in textured_quality.as_dict().values())
    assert textured_quality.quality_score > flat_quality.quality_score
    assert textured_quality.gradient_mean > flat_quality.gradient_mean


def test_forward_backward_lk_rejects_unreliable_flow_and_returns_prediction() -> None:
    rng = np.random.default_rng(8)
    previous = cv2.GaussianBlur(rng.integers(0, 255, (100, 120), dtype=np.uint8), (5, 5), 0.8)
    current = cv2.warpAffine(previous, np.array([[1, 0, 3], [0, 1, -2]], np.float32), (120, 100))

    result = track_forward_backward(previous, current, point=(60.0, 50.0), initial_velocity=(3.0, -2.0), fb_threshold_px=0.3)

    assert result.valid
    assert result.point == pytest.approx((63.0, 48.0), abs=0.2)
    assert result.predicted_point == (63.0, 48.0)
    assert result.fb_error_px is not None and result.fb_error_px < 0.3


@pytest.mark.parametrize("motion_model", ["constant_velocity", "kalman"])
def test_motion_predictor_tracks_xyz_disparity_velocity_and_confidence(motion_model: str) -> None:
    predictor = MotionPredictor(motion_model=motion_model, dt_s=0.1)
    state = PointMeasurementState(point_id="P1", xy_px=(100.0, 60.0), disparity_px=30.0, xyz_mm=(0.0, 0.0, 2000.0), velocity_xyz_mm_s=(10.0, 0.0, 0.0), confidence=0.9)
    predictor.update(state)

    prediction = predictor.predict("P1")

    assert prediction.xy_px == pytest.approx((101.0, 60.0))
    assert prediction.xyz_mm[0] > 0.0
    assert prediction.predicted_disparity_px == pytest.approx(30.0)


def test_precision_planner_computes_required_disparity_and_smallest_feasible_baseline() -> None:
    sigma_d = required_disparity_precision(distance_mm=3000.0, baseline_mm=120.0, focal_length_px=800.0, target_depth_error_mm=3.0)
    decision = recommend_baseline(distance_mm=3000.0, target_depth_error_mm=3.0, available_baselines_mm=[60, 100, 140, 180, 300], focal_length_px=800.0, assumed_disparity_noise_px=0.05)

    assert sigma_d == pytest.approx(0.032)
    assert decision.recommended_baseline_mm == 300.0
    assert decision.feasible
