from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.camera_model import StereoCameraModel


@dataclass(frozen=True)
class Flow3DConsistency:
    predicted_uv_px: tuple[float, float]
    observed_uv_px: tuple[float, float]
    residual_px: float


def reprojection_flow_residual(previous_xyz_mm: np.ndarray, current_xyz_mm: np.ndarray, observed_uv_px: np.ndarray, camera: StereoCameraModel) -> Flow3DConsistency:
    """Compare LK flow with image motion predicted by successive 3D observations."""
    points = np.asarray([previous_xyz_mm, current_xyz_mm], dtype=np.float64)
    if points.shape != (2, 3):
        raise ValueError("previous_xyz_mm and current_xyz_mm must be three-vectors")
    observed = np.asarray(observed_uv_px, dtype=np.float64)
    if observed.shape != (2,) or not np.isfinite(observed).all():
        raise ValueError("observed_uv_px must be a finite two-vector")
    projected = camera.project(points, camera="left")
    predicted = projected[1] - projected[0]
    return Flow3DConsistency(tuple(float(value) for value in predicted), tuple(float(value) for value in observed), float(np.linalg.norm(predicted - observed)))
