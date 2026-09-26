from __future__ import annotations

import numpy as np


def neighbor_disparity_residual(disparity_px: float, neighbor_disparities_px: list[float]) -> float:
    """Absolute deviation from the finite neighbor median."""
    neighbors = np.asarray(neighbor_disparities_px, dtype=np.float64)
    finite = neighbors[np.isfinite(neighbors)]
    if finite.size == 0 or not np.isfinite(disparity_px):
        return float("inf")
    return float(abs(disparity_px - np.median(finite)))
