from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import MatcherConfig


@dataclass(frozen=True)
class CycleCheckResult:
    status: str
    error_px: float
    cost: float
    confidence_scale: float
    should_recover: bool = False


def evaluate_cycle_consistency(
    stereo_right_xy: tuple[float, float],
    temporal_right_xy: tuple[float, float],
    config: MatcherConfig,
    recovered_right_xy: tuple[float, float] | None = None,
) -> CycleCheckResult:
    stereo = np.asarray(stereo_right_xy, dtype=np.float64)
    temporal = np.asarray(temporal_right_xy, dtype=np.float64)
    if stereo.shape != (2,) or temporal.shape != (2,) or not np.isfinite(stereo).all() or not np.isfinite(temporal).all():
        raise ValueError("Cycle coordinates must be finite two-dimensional points")
    error = float(np.linalg.norm(stereo - temporal))
    if error <= config.cycle_soft_threshold_px:
        return _result("cycle_valid", error, config, 1.0)
    if error <= config.cycle_hard_threshold_px:
        span = config.cycle_hard_threshold_px - config.cycle_soft_threshold_px
        scale = 1.0 - 0.5 * (error - config.cycle_soft_threshold_px) / span
        return _result("cycle_soft", error, config, scale)
    if error > config.cycle_recovery_threshold_px:
        return _result("cycle_failed", error, config, 0.0)
    if recovered_right_xy is None:
        return _result("cycle_recovery_required", error, config, 0.5, should_recover=True)

    recovered = np.asarray(recovered_right_xy, dtype=np.float64)
    if recovered.shape != (2,) or not np.isfinite(recovered).all():
        return _result("cycle_failed", error, config, 0.0)
    recovered_error = float(np.linalg.norm(recovered - temporal))
    if recovered_error <= config.cycle_hard_threshold_px:
        scale = 1.0 if recovered_error <= config.cycle_soft_threshold_px else 0.6
        return _result("cycle_recovered", recovered_error, config, scale)
    return _result("cycle_failed", recovered_error, config, 0.0)


def _result(
    status: str,
    error: float,
    config: MatcherConfig,
    confidence_scale: float,
    should_recover: bool = False,
) -> CycleCheckResult:
    return CycleCheckResult(
        status=status,
        error_px=error,
        cost=float(min(error / config.cycle_hard_threshold_px, 1.0)),
        confidence_scale=float(np.clip(confidence_scale, 0.0, 1.0)),
        should_recover=should_recover,
    )
