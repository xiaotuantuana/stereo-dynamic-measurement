from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _xyz(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("XYZ must be a finite three-vector in mm")
    return result


def _unit_interval(name: str, value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return numeric


@dataclass(frozen=True)
class MeasurementResult:
    frame_id: int
    timestamp: float
    point_id: str
    left_xy: tuple[float, float]
    right_xy: tuple[float, float]
    disparity_raw: float
    disparity_subpixel: float
    xyz_raw: np.ndarray
    gradient_score: float = 0.0
    texture_score: float = 0.0
    blur_score: float = 0.0
    flow_u: float = 0.0
    flow_v: float = 0.0
    flow_fb_error: float = 0.0
    lr_residual: float = 0.0
    epipolar_residual: float = 0.0
    matching_cost: float = 0.0
    neighbor_residual: float = 0.0
    temporal_residual: float = 0.0
    tracking_loss_residual: float = 0.0
    measurement_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.frame_id < 0 or not self.point_id or not np.isfinite(self.timestamp):
            raise ValueError("frame_id, timestamp and point_id are invalid")
        object.__setattr__(self, "xyz_raw", _xyz(self.xyz_raw))
        for name in ("gradient_score", "texture_score", "blur_score", "measurement_confidence"):
            _unit_interval(name, getattr(self, name))
        if not np.isfinite(self.tracking_loss_residual) or self.tracking_loss_residual < 0:
            raise ValueError("tracking_loss_residual must be finite and non-negative")


@dataclass(frozen=True)
class PhysicsValidationResult:
    frame_id: int
    timestamp: float
    point_id: str
    xyz_raw: np.ndarray
    xyz_corrected: np.ndarray
    flow_3d_score: float = 0.0
    temporal_score: float = 0.0
    spatial_score: float = 0.0
    spectral_score: float = 0.0
    phase_score: float = 0.0
    coherence_score: float = 0.0
    physics_confidence: float = 0.0
    correction_applied: bool = False
    correction_method: str = "none"
    correction_magnitude_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.frame_id < 0 or not self.point_id or not np.isfinite(self.timestamp):
            raise ValueError("frame_id, timestamp and point_id are invalid")
        object.__setattr__(self, "xyz_raw", _xyz(self.xyz_raw))
        object.__setattr__(self, "xyz_corrected", _xyz(self.xyz_corrected))
        for name in ("flow_3d_score", "temporal_score", "spatial_score", "spectral_score", "phase_score", "coherence_score", "physics_confidence"):
            _unit_interval(name, getattr(self, name))
        if not np.isfinite(self.correction_magnitude_mm) or self.correction_magnitude_mm < 0:
            raise ValueError("correction_magnitude_mm must be finite and non-negative")


@dataclass(frozen=True)
class RecoveryAction:
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosisResult:
    frame_id: int
    timestamp: float
    point_id: str
    fault_type: str
    fault_score: float
    recovery_action: RecoveryAction
    recovery_success: bool
    c_phy_before: float
    c_phy_after: float
    residuals_before: dict[str, float] = field(default_factory=dict)
    residuals_after: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("fault_score", "c_phy_before", "c_phy_after"):
            _unit_interval(name, getattr(self, name))
