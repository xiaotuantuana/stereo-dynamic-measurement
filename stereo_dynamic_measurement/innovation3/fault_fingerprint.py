from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaultFingerprint:
    gradient_quality: float = 1.0; blur_score: float = 1.0
    lr_residual: float = 0.0; epipolar_residual: float = 0.0; matching_cost_residual: float = 0.0; neighbor_residual: float = 0.0
    flow_residual: float = 0.0; fb_error: float = 0.0; temporal_residual: float = 0.0
    reference_motion_residual: float = 0.0; geometry_health_residual: float = 0.0; physics_residual: float = 0.0
    common_target_motion_ratio: float = 0.0; tracking_loss_residual: float = 0.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in self.__dict__.values()):
            raise ValueError("FaultFingerprint fields must be finite")
        if not 0 <= self.gradient_quality <= 1 or not 0 <= self.blur_score <= 1 or not 0 <= self.physics_residual <= 1:
            raise ValueError("quality/blur/physics residual must lie in [0,1]")
