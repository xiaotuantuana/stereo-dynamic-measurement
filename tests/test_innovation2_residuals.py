from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stereo_dynamic_measurement.calibration.camera_model import StereoCameraModel
from stereo_dynamic_measurement.innovation2.flow_3d_consistency import reprojection_flow_residual
from stereo_dynamic_measurement.innovation2.temporal_consistency import temporal_prediction_residual
from stereo_dynamic_measurement.innovation2.spatial_consistency import spatial_consistency_residuals


def _camera() -> StereoCameraModel:
    return StereoCameraModel.from_mapping({"image_size": [640, 480], "K_left": [[800, 0, 320], [0, 800, 240], [0, 0, 1]], "D_left": [0] * 5, "K_right": [[800, 0, 320], [0, 800, 240], [0, 0, 1]], "D_right": [0] * 5, "R": np.eye(3).tolist(), "T": [-120, 0, 0], "unit": "mm"})


def test_2d_3d_reprojection_residual_is_zero_for_consistent_flow() -> None:
    camera = _camera()
    previous_xyz = np.array([0.0, 0.0, 2000.0])
    current_xyz = np.array([10.0, 5.0, 2000.0])
    uv_previous, uv_current = camera.project(np.array([previous_xyz, current_xyz]), camera="left")
    observed_uv = uv_current - uv_previous

    result = reprojection_flow_residual(previous_xyz, current_xyz, observed_uv, camera)

    assert result.residual_px == pytest.approx(0.0, abs=1e-10)
    assert result.predicted_uv_px == pytest.approx(tuple(observed_uv))


def test_temporal_residual_keeps_high_acceleration_when_transient_protection_enabled() -> None:
    history = np.array([[0.0, 0.0, 2000.0], [1.0, 0.0, 2000.0]])
    impact = np.array([20.0, 0.0, 2000.0])

    result = temporal_prediction_residual(history, impact, transient_acceleration_mm=5.0)

    assert result.residual_mm > 10.0
    assert result.transient_protected
    assert result.confidence_penalty < 1.0


def test_spatial_residual_flags_only_the_jump_point_against_reference_shape() -> None:
    baseline = pd.DataFrame({"point_id": ["P1", "P2", "P3", "P4"], "X_mm": [0, 100, 200, 300], "Y_mm": [0, 0, 0, 0], "Z_mm": [2000, 2000, 2000, 2000]})
    current = baseline.copy()
    current.loc[current.point_id == "P3", "Y_mm"] = 50.0

    residuals = spatial_consistency_residuals(baseline, current)

    assert residuals.loc[residuals.point_id == "P3", "spatial_residual_mm"].item() > 30.0
    assert residuals.loc[residuals.point_id != "P3", "spatial_residual_mm"].max() < 1e-9
