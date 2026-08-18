from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectionConfig:
    threshold: float = 0.55
    prediction_weight: float = 0.65
    neighbor_weight: float = 0.35

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1 or self.prediction_weight < 0 or self.neighbor_weight < 0 or self.prediction_weight + self.neighbor_weight <= 0:
            raise ValueError("Invalid correction configuration")


@dataclass(frozen=True)
class CorrectionResult:
    raw_xyz_mm: np.ndarray
    corrected_xyz_mm: np.ndarray
    correction_applied: bool
    correction_method: str
    correction_magnitude_mm: float


def correct_trajectory_point(raw_xyz_mm: np.ndarray, predicted_xyz_mm: np.ndarray | None, neighbor_xyz_mm: np.ndarray, *, c_phy: float, config: CorrectionConfig) -> CorrectionResult:
    """Generate an auditable correction, never modifying the supplied raw coordinates."""
    raw = np.asarray(raw_xyz_mm, dtype=np.float64).copy()
    prediction = None if predicted_xyz_mm is None else np.asarray(predicted_xyz_mm, dtype=np.float64)
    neighbors = np.asarray(neighbor_xyz_mm, dtype=np.float64)
    if raw.shape != (3,) or (prediction is not None and prediction.shape != (3,)):
        raise ValueError("raw and predicted XYZ must be three-vectors")
    if not 0 <= c_phy <= 1:
        raise ValueError("c_phy must be in [0,1]")
    if c_phy >= config.threshold:
        return CorrectionResult(raw, raw.copy(), False, "none", 0.0)
    usable_neighbors = neighbors.reshape(-1, 3) if neighbors.size else np.empty((0, 3))
    usable_neighbors = usable_neighbors[np.isfinite(usable_neighbors).all(axis=1)]
    if prediction is not None and np.isfinite(prediction).all() and len(usable_neighbors):
        neighbor = np.median(usable_neighbors, axis=0)
        corrected = (config.prediction_weight * prediction + config.neighbor_weight * neighbor) / (config.prediction_weight + config.neighbor_weight)
        method = "history_neighbor_weighted"
    elif prediction is not None and np.isfinite(prediction).all():
        corrected, method = prediction.copy(), "history_prediction"
    elif len(usable_neighbors):
        corrected, method = np.median(usable_neighbors, axis=0), "neighbor_interpolation"
    else:
        return CorrectionResult(raw, raw.copy(), False, "insufficient_evidence", 0.0)
    magnitude = float(np.linalg.norm(corrected - raw))
    return CorrectionResult(raw, np.asarray(corrected, dtype=np.float64), True, method, magnitude)
