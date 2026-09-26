from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostCurveDiagnostics:
    best_disparity: float
    best_cost: float
    second_best_disparity: float | None
    second_best_cost: float | None
    uniqueness_margin: float | None
    predicted_cost: float | None
    predicted_best_disagreement: float
    candidate_count: int


def summarize_cost_curve(
    disparities: np.ndarray,
    costs: np.ndarray,
    predicted_disparity: float,
) -> CostCurveDiagnostics:
    disparity_values = np.asarray(disparities, dtype=np.float64).reshape(-1)
    cost_values = np.asarray(costs, dtype=np.float64).reshape(-1)
    valid = np.isfinite(disparity_values) & np.isfinite(cost_values)
    disparity_values, cost_values = disparity_values[valid], cost_values[valid]
    if not disparity_values.size:
        raise ValueError("Cost curve has no finite candidates")
    order = np.argsort(cost_values)
    best_index = int(order[0])
    best_disparity = float(disparity_values[best_index])
    best_cost = float(cost_values[best_index])
    second_index = next(
        (int(index) for index in order[1:] if abs(float(disparity_values[index]) - best_disparity) > 1.0),
        None,
    )
    predicted_index = int(np.argmin(np.abs(disparity_values - float(predicted_disparity))))
    second_cost = None if second_index is None else float(cost_values[second_index])
    return CostCurveDiagnostics(
        best_disparity=best_disparity,
        best_cost=best_cost,
        second_best_disparity=None if second_index is None else float(disparity_values[second_index]),
        second_best_cost=second_cost,
        uniqueness_margin=None if second_cost is None else max(0.0, second_cost - best_cost),
        predicted_cost=float(cost_values[predicted_index]),
        predicted_best_disagreement=abs(float(predicted_disparity) - best_disparity),
        candidate_count=int(disparity_values.size),
    )


def calibrated_initial_confidence(
    *,
    texture_std: float | None,
    lr_error_px: float | None,
    diagnostics: CostCurveDiagnostics,
    texture_reference: float,
    margin_reference: float,
    lr_reference: float,
    max_photo_cost: float,
) -> float:
    """Conservative runtime-only confidence; Ground Truth is intentionally absent."""

    if texture_std is None or diagnostics.predicted_cost is None or diagnostics.uniqueness_margin is None:
        return 0.0
    texture = float(np.clip(texture_std / max(texture_reference, 1e-9), 0.0, 1.0))
    photo = float(np.clip(1.0 - diagnostics.predicted_cost / max(max_photo_cost, 1e-9), 0.0, 1.0))
    uniqueness = float(np.clip(diagnostics.uniqueness_margin / max(margin_reference, 1e-9), 0.0, 1.0))
    lr = 0.0 if lr_error_px is None else float(np.clip(1.0 - lr_error_px / max(lr_reference, 1e-9), 0.0, 1.0))
    agreement = float(np.exp(-diagnostics.predicted_best_disagreement / 2.0))
    return float(np.clip(min(texture, photo, uniqueness, lr, agreement), 0.0, 1.0))


def classify_rejection(forced_error: float | None, *, good_error_px: float = 3.0) -> str:
    if forced_error is None or not np.isfinite(forced_error):
        return "Unassessable"
    return "Good Reject" if forced_error > good_error_px else "Potential Over-Reject"

