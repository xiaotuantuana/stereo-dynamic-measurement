from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalConsistency:
    predicted_xyz_mm: tuple[float, float, float]
    residual_mm: float
    transient_protected: bool
    confidence_penalty: float


def temporal_prediction_residual(history_xyz_mm: np.ndarray, current_xyz_mm: np.ndarray, *, transient_acceleration_mm: float = 20.0, residual_scale_mm: float = 10.0) -> TemporalConsistency:
    """Second-order temporal residual without filtering away impact/tail motion."""
    history = np.asarray(history_xyz_mm, dtype=np.float64)
    current = np.asarray(current_xyz_mm, dtype=np.float64)
    if history.shape != (2, 3) or current.shape != (3,):
        raise ValueError("history must be (2,3) and current must be (3,)")
    predicted = 2.0 * history[1] - history[0]
    innovation = current - predicted
    residual = float(np.linalg.norm(innovation))
    acceleration = float(np.linalg.norm(current - 2.0 * history[1] + history[0]))
    protected = acceleration >= transient_acceleration_mm
    penalty = float(np.clip(residual / max(residual_scale_mm, 1e-9), 0.0, 1.0))
    if protected:
        penalty *= 0.25
    return TemporalConsistency(tuple(float(value) for value in predicted), residual, protected, penalty)
