from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointMeasurementState:
    point_id: str
    xy_px: tuple[float, float]
    disparity_px: float
    xyz_mm: tuple[float, float, float]
    velocity_xyz_mm_s: tuple[float, float, float]
    confidence: float
    velocity_xy_px_s: tuple[float, float] | None = None
    disparity_velocity_px_s: float = 0.0


@dataclass(frozen=True)
class MotionPrediction:
    xy_px: tuple[float, float]
    xyz_mm: tuple[float, float, float]
    predicted_disparity_px: float
    confidence: float


class MotionPredictor:
    """Per-point constant-velocity or lightweight Kalman state prediction."""

    def __init__(self, motion_model: str = "kalman", dt_s: float = 1.0 / 30.0, process_noise: float = 1e-2) -> None:
        if motion_model not in {"kalman", "constant_velocity"} or dt_s <= 0:
            raise ValueError("motion_model must be kalman/constant_velocity and dt_s positive")
        self.motion_model, self.dt_s, self.process_noise = motion_model, dt_s, process_noise
        self._states: dict[str, PointMeasurementState] = {}
        self._covariances: dict[str, np.ndarray] = {}

    def update(self, state: PointMeasurementState) -> None:
        if not 0.0 <= state.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self._states[state.point_id] = state
        self._covariances.setdefault(state.point_id, np.eye(10, dtype=np.float64))

    def predict(self, point_id: str) -> MotionPrediction:
        state = self._states[point_id]
        dt = self.dt_s
        velocity_xy = state.velocity_xy_px_s or state.velocity_xyz_mm_s[:2]
        xy = tuple(float(state.xy_px[i] + velocity_xy[i] * dt) for i in range(2))
        xyz = tuple(float(state.xyz_mm[i] + state.velocity_xyz_mm_s[i] * dt) for i in range(3))
        disparity = float(state.disparity_px + state.disparity_velocity_px_s * dt)
        if self.motion_model == "kalman":
            covariance = self._covariances[point_id]
            self._covariances[point_id] = covariance + np.eye(covariance.shape[0]) * self.process_noise
        return MotionPrediction(xy, xyz, disparity, state.confidence)
