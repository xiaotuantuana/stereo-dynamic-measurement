from __future__ import annotations

import cv2
import numpy as np

from stereo_dynamic_measurement.calibration.camera_model import StereoCameraModel
from stereo_dynamic_measurement.innovation1.measurement_pipeline import AdaptiveStereoMeasurementPipeline, Innovation1Config
from stereo_dynamic_measurement.innovation1.motion_predictor import PointMeasurementState


def _camera() -> StereoCameraModel:
    return StereoCameraModel.from_mapping({"image_size": [160, 100], "K_left": [[300, 0, 80], [0, 300, 50], [0, 0, 1]], "D_left": [0] * 5, "K_right": [[300, 0, 80], [0, 300, 50], [0, 0, 1]], "D_right": [0] * 5, "R": np.eye(3).tolist(), "T": [-100, 0, 0], "unit": "mm"})


def test_pipeline_outputs_mm_measurement_and_multisource_confidence() -> None:
    rng = np.random.default_rng(14)
    left = cv2.GaussianBlur(rng.integers(0, 256, (100, 160), dtype=np.uint8), (3, 3), 0.7)
    right = cv2.warpAffine(left, np.array([[1, 0, -12], [0, 1, 0]], np.float32), (160, 100), borderMode=cv2.BORDER_REFLECT101)
    state = PointMeasurementState("P1", (90.0, 50.0), 12.0, (80.0, 0.0, 2500.0), (0.0, 0.0, 0.0), 0.9)
    pipeline = AdaptiveStereoMeasurementPipeline(_camera(), Innovation1Config(motion_model="constant_velocity"))
    pipeline.initialize(state)

    result = pipeline.process(left, left, right, point_id="P1", target_depth_error_mm=10.0)

    assert result.disparity_px is not None
    assert result.xyz_mm is not None and np.isfinite(result.xyz_mm).all()
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_state in {"HIGH", "MEDIUM", "LOW"}
    assert result.search_radius_px >= 2
    assert set(result.confidence_components) >= {"gradient", "matching", "left_right", "flow", "temporal", "neighbor"}


def test_low_confidence_path_uses_global_sgbm_fallback() -> None:
    flat = np.full((100, 160), 128, dtype=np.uint8)
    state = PointMeasurementState("P1", (90.0, 50.0), 12.0, (80.0, 0.0, 2500.0), (0.0, 0.0, 0.0), 0.05)
    pipeline = AdaptiveStereoMeasurementPipeline(_camera(), Innovation1Config(motion_model="kalman"))
    pipeline.initialize(state)

    result = pipeline.process(flat, flat, flat, point_id="P1", target_depth_error_mm=1.0)

    assert result.fallback_used
    assert result.action in {"fallback", "predict_only", "reinitialize"}
