from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubpixelResult:
    disparity_px: float
    offset_px: float
    valid: bool
    status: str


def refine_parabolic(integer_disparity_px: float, *, cost_minus: float, cost_center: float, cost_plus: float, at_boundary: bool, denominator_epsilon: float = 1e-9) -> SubpixelResult:
    """Refine a discrete minimum using three costs without emitting NaN/Inf."""
    values = (cost_minus, cost_center, cost_plus)
    if at_boundary or not all(np.isfinite(value) for value in values):
        return SubpixelResult(float(integer_disparity_px), 0.0, False, "boundary_or_nonfinite")
    denominator = cost_minus - 2.0 * cost_center + cost_plus
    if denominator <= denominator_epsilon:
        return SubpixelResult(float(integer_disparity_px), 0.0, False, "flat_or_nonconvex")
    offset = 0.5 * (cost_minus - cost_plus) / denominator
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return SubpixelResult(float(integer_disparity_px), 0.0, False, "unstable_offset")
    return SubpixelResult(float(integer_disparity_px + offset), float(offset), True, "valid")
