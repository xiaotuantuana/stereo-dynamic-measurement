from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.camera_model import StereoCameraModel


@dataclass(frozen=True)
class StereoProjection:
    left_ideal_px: np.ndarray
    right_ideal_px: np.ndarray
    left_points_px: np.ndarray
    right_points_px: np.ndarray


@dataclass
class StereoScene:
    """Point-only virtual stereo rig with controllable image/localization noise."""

    camera: StereoCameraModel
    image_noise_std_px: float = 0.0
    localization_noise_std_px: float = 0.0
    random_seed: int | None = 7

    def __post_init__(self) -> None:
        if self.image_noise_std_px < 0 or self.localization_noise_std_px < 0:
            raise ValueError("Noise standard deviations must be non-negative pixels")
        self._rng = np.random.default_rng(self.random_seed)

    def project(self, xyz_mm: np.ndarray) -> StereoProjection:
        left_ideal = self.camera.project(xyz_mm, camera="left")
        right_ideal = self.camera.project(xyz_mm, camera="right")
        total_std = float(np.hypot(self.image_noise_std_px, self.localization_noise_std_px))
        left = left_ideal + self._rng.normal(0.0, total_std, left_ideal.shape)
        right = right_ideal + self._rng.normal(0.0, total_std, right_ideal.shape)
        return StereoProjection(left_ideal, right_ideal, left, right)
