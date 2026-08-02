from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import MatcherConfig


@dataclass(frozen=True)
class KalmanUpdateResult:
    status: str
    estimated_left_xy: tuple[float, float]
    estimated_disparity: float
    innovation: tuple[float, float, float]
    innovation_norm: float
    kalman_gain_disparity: float
    nis: float


class AdaptivePointKalman:
    def __init__(self, config: MatcherConfig):
        self.config = config
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.predict_only_frames = 0

    def initialize(self, left_xy: tuple[float, float], disparity: float) -> None:
        measurement = np.asarray([left_xy[0], left_xy[1], disparity], dtype=np.float64)
        if not np.isfinite(measurement).all():
            raise ValueError("Kalman initialization must be finite")
        self.state = np.zeros(6, dtype=np.float64)
        self.state[:3] = measurement
        self.covariance = np.diag(
            [self.config.kalman_initial_position_variance] * 3
            + [self.config.kalman_initial_velocity_variance] * 3
        ).astype(np.float64)
        self.predict_only_frames = 0

    def predict(self, dt: float = 1.0) -> np.ndarray:
        if self.state is None or self.covariance is None:
            raise RuntimeError("Kalman filter must be initialized before prediction")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3, dtype=np.float64) * dt
        process = np.diag(
            [self.config.kalman_process_position_variance * dt * dt] * 3
            + [self.config.kalman_process_velocity_variance * dt] * 3
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        return self.state.copy()

    def update(
        self,
        left_xy: tuple[float, float],
        disparity: float,
        left_variance_px2: float,
        disparity_variance_px2: float,
    ) -> KalmanUpdateResult:
        if self.state is None or self.covariance is None:
            raise RuntimeError("Kalman filter must be initialized before update")
        measurement = np.asarray([left_xy[0], left_xy[1], disparity], dtype=np.float64)
        if not np.isfinite(measurement).all():
            raise ValueError("Kalman measurement must be finite")
        if left_variance_px2 <= 0 or disparity_variance_px2 <= 0:
            raise ValueError("Measurement variances must be positive")
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3, dtype=np.float64)
        measurement_covariance = np.diag(
            [left_variance_px2, left_variance_px2, disparity_variance_px2]
        ).astype(np.float64)
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + measurement_covariance
        nis = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
        status = "updated"
        if nis > self.config.kalman_nis_hard_threshold:
            self.predict_only_frames += 1
            return self._result(
                "predict_only_outlier",
                innovation,
                nis,
                kalman_gain_disparity=0.0,
            )
        if nis > self.config.kalman_nis_soft_threshold:
            measurement_covariance *= self.config.kalman_nis_variance_scale
            innovation_covariance = (
                observation @ self.covariance @ observation.T + measurement_covariance
            )
            nis = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
            status = "variance_inflated"
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        identity = np.eye(6, dtype=np.float64)
        residual_transform = identity - gain @ observation
        self.covariance = (
            residual_transform @ self.covariance @ residual_transform.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.predict_only_frames = 0
        return self._result(status, innovation, nis, float(gain[2, 2]))

    def current_measurement_state(self) -> tuple[tuple[float, float], float]:
        if self.state is None:
            raise RuntimeError("Kalman filter has not been initialized")
        return (float(self.state[0]), float(self.state[1])), float(self.state[2])

    def _result(
        self,
        status: str,
        innovation: np.ndarray,
        nis: float,
        kalman_gain_disparity: float,
    ) -> KalmanUpdateResult:
        assert self.state is not None
        return KalmanUpdateResult(
            status=status,
            estimated_left_xy=(float(self.state[0]), float(self.state[1])),
            estimated_disparity=float(self.state[2]),
            innovation=tuple(float(value) for value in innovation),
            innovation_norm=float(np.linalg.norm(innovation)),
            kalman_gain_disparity=kalman_gain_disparity,
            nis=nis,
        )
