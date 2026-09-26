from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GeometryHealth:
    median_vertical_disparity_px: float; p95_vertical_disparity_px: float; epipolar_rms_px: float; health_score: float

def geometry_health(left_y: np.ndarray, right_y: np.ndarray, scale_px: float = 1.0) -> GeometryHealth:
    left, right = np.asarray(left_y, float), np.asarray(right_y, float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0: raise ValueError("Reference y arrays must be equal nonempty vectors")
    residual = left - right
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return GeometryHealth(float(np.median(abs(residual))), float(np.percentile(abs(residual), 95)), rms, float(np.exp(-rms / max(scale_px, 1e-9))))
