from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaultFingerprint:
    gradient_quality: float | None = None
    blur_score: float | None = None
    lr_residual: float | None = None
    epipolar_residual: float | None = None
    matching_cost_residual: float | None = None
    neighbor_residual: float | None = None
    flow_residual: float | None = None
    fb_error: float | None = None
    temporal_residual: float | None = None
    reference_motion_residual: float | None = None
    geometry_health_residual: float | None = None
    physics_residual: float | None = None
    common_target_motion_ratio: float | None = None
    tracking_loss_residual: float | None = None

    def __post_init__(self) -> None:
        supplied = [value for value in self.__dict__.values() if value is not None]
        if not all(np.isfinite(value) for value in supplied):
            raise ValueError("supplied FaultFingerprint fields must be finite")
        for name in ("gradient_quality", "blur_score", "physics_residual", "common_target_motion_ratio"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1] when available")
