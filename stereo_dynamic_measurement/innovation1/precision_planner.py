from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class TargetAccuracySpec:
    """Requested one-standard-deviation uncertainty in millimetres."""

    metric: Literal["sigma"] = "sigma"
    target_sigma_x_mm: float | None = None
    target_sigma_y_mm: float | None = None
    target_sigma_z_mm: float | None = None

    def __post_init__(self) -> None:
        if self.metric != "sigma":
            raise ValueError("only the sigma target metric is supported")
        targets = (
            self.target_sigma_x_mm,
            self.target_sigma_y_mm,
            self.target_sigma_z_mm,
        )
        supplied = [value for value in targets if value is not None]
        if not supplied or any(not np.isfinite(value) or value <= 0 for value in supplied):
            raise ValueError("at least one positive target sigma is required")


@dataclass(frozen=True)
class PrecisionPlan:
    target_spec: TargetAccuracySpec
    current_depth_m: float
    required_sigma_disparity_px: float | None
    estimated_sigma_disparity_px: float | None
    estimated_sigma_x_mm: float | None
    estimated_sigma_y_mm: float | None
    estimated_sigma_z_mm: float | None
    precision_ratio: float | None
    feasible: bool
    feasibility_reason: str
    limiting_axis: str


def build_precision_plan(
    target_spec: TargetAccuracySpec,
    *,
    current_depth_m: float,
    focal_length_px: float,
    baseline_mm: float,
    estimated_sigma_disparity_px: float | None,
    minimum_achievable_sigma_disparity_px: float | None = None,
) -> PrecisionPlan:
    """Build a Z-sigma plan from causal geometry and estimated match uncertainty.

    X/Y targets remain visible in ``target_spec`` but are not controlling until a
    calibrated full reprojection covariance model is available.
    """

    if not np.isfinite(current_depth_m) or current_depth_m <= 0:
        raise ValueError("current_depth_m must be positive and finite")
    if not np.isfinite(focal_length_px) or focal_length_px <= 0:
        raise ValueError("focal_length_px must be positive and finite")
    if not np.isfinite(baseline_mm) or baseline_mm <= 0:
        raise ValueError("baseline_mm must be positive and finite")
    for name, value in (
        ("estimated_sigma_disparity_px", estimated_sigma_disparity_px),
        ("minimum_achievable_sigma_disparity_px", minimum_achievable_sigma_disparity_px),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite when supplied")

    target_z = target_spec.target_sigma_z_mm
    if target_z is None:
        return PrecisionPlan(
            target_spec=target_spec,
            current_depth_m=float(current_depth_m),
            required_sigma_disparity_px=None,
            estimated_sigma_disparity_px=estimated_sigma_disparity_px,
            estimated_sigma_x_mm=None,
            estimated_sigma_y_mm=None,
            estimated_sigma_z_mm=None,
            precision_ratio=None,
            feasible=False,
            feasibility_reason="z_target_unavailable_x_y_partial_not_controlling",
            limiting_axis="x_y_partial",
        )

    depth_mm = float(current_depth_m) * 1000.0
    required = required_disparity_precision(
        depth_mm,
        float(baseline_mm),
        float(focal_length_px),
        float(target_z),
    )
    estimated_z = (
        None
        if estimated_sigma_disparity_px is None
        else depth_mm**2 * float(estimated_sigma_disparity_px)
        / (float(focal_length_px) * float(baseline_mm))
    )
    ratio = (
        None
        if estimated_sigma_disparity_px is None
        else float(estimated_sigma_disparity_px) / required
    )
    floor = (
        minimum_achievable_sigma_disparity_px
        if minimum_achievable_sigma_disparity_px is not None
        else estimated_sigma_disparity_px
    )
    feasible = floor is not None and float(floor) <= required
    reason = (
        "uncertainty_floor_within_target"
        if feasible
        else "required_sigma_d_below_uncertainty_floor"
        if floor is not None
        else "estimated_uncertainty_unavailable"
    )
    return PrecisionPlan(
        target_spec=target_spec,
        current_depth_m=float(current_depth_m),
        required_sigma_disparity_px=required,
        estimated_sigma_disparity_px=estimated_sigma_disparity_px,
        estimated_sigma_x_mm=None,
        estimated_sigma_y_mm=None,
        estimated_sigma_z_mm=estimated_z,
        precision_ratio=ratio,
        feasible=feasible,
        feasibility_reason=reason,
        limiting_axis="z",
    )


@dataclass(frozen=True)
class BaselineRecommendation:
    recommended_baseline_mm: float
    required_disparity_precision_px: float
    achievable_depth_error_mm: float
    feasible: bool


def required_disparity_precision(distance_mm: float, baseline_mm: float, focal_length_px: float, target_depth_error_mm: float) -> float:
    if min(distance_mm, baseline_mm, focal_length_px, target_depth_error_mm) <= 0:
        raise ValueError("distance, baseline, focal length and target error must be positive")
    return target_depth_error_mm * focal_length_px * baseline_mm / (distance_mm ** 2)


def recommend_baseline(distance_mm: float, target_depth_error_mm: float, available_baselines_mm: list[float], focal_length_px: float, assumed_disparity_noise_px: float) -> BaselineRecommendation:
    if not available_baselines_mm or assumed_disparity_noise_px <= 0:
        raise ValueError("At least one baseline and a positive disparity noise are required")
    candidates = sorted(float(value) for value in available_baselines_mm if value > 0)
    if not candidates:
        raise ValueError("Baselines must be positive")
    required = distance_mm ** 2 * assumed_disparity_noise_px / (focal_length_px * target_depth_error_mm)
    selected = next((baseline for baseline in candidates if baseline >= required), candidates[-1])
    achievable = distance_mm ** 2 * assumed_disparity_noise_px / (focal_length_px * selected)
    return BaselineRecommendation(selected, required_disparity_precision(distance_mm, selected, focal_length_px, target_depth_error_mm), achievable, selected >= required)
