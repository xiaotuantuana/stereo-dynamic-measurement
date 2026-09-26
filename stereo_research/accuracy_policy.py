from __future__ import annotations

from dataclasses import dataclass

from stereo_dynamic_measurement.innovation1.precision_planner import (
    PrecisionPlan,
    TargetAccuracySpec,
)


@dataclass(frozen=True)
class MeasurementPolicyDecision:
    enabled: bool
    precision_policy_warmup: bool
    target_spec: TargetAccuracySpec
    required_sigma_d_px: float | None
    estimated_sigma_d_px: float | None
    estimated_sigma_x_mm: float | None
    estimated_sigma_y_mm: float | None
    estimated_sigma_z_mm: float | None
    precision_ratio: float | None
    precision_feasible: bool
    precision_status: str
    limiting_axis: str
    base_search_radius_px: int
    final_search_radius_px: int
    subpixel_policy: str
    refinement_level: int
    retry_budget: int
    precision_retry_count: int
    acceptance_reason: str
    policy_reason: str


def decide_measurement_policy(
    plan: PrecisionPlan,
    *,
    base_search_radius_px: int,
    vision_state: str,
    max_precision_retry: int,
    max_refinement_level: int,
    precision_retry_count: int = 0,
    warmup: bool = False,
) -> MeasurementPolicyDecision:
    """Fuse target strictness with vision state without changing motion radius."""

    if base_search_radius_px < 1:
        raise ValueError("base_search_radius_px must be positive")
    if max_precision_retry < 0 or max_refinement_level < 0:
        raise ValueError("precision retry and refinement bounds must be non-negative")
    if not 0 <= precision_retry_count <= max_precision_retry:
        raise ValueError("precision_retry_count is outside the configured bound")
    if vision_state not in {"HIGH", "MEDIUM", "LOW", "LOST"}:
        raise ValueError("vision_state is invalid")

    ratio = plan.precision_ratio
    if ratio is None or ratio <= 1.0:
        requested_level = 0
    elif ratio <= 2.0:
        requested_level = 1
    else:
        requested_level = 2
    refinement_level = min(requested_level, max_refinement_level)

    if warmup:
        status = "WARMUP"
        retry_budget = 0
        acceptance = "legacy_warmup_measurement"
        reason = "no_prior_causal_depth"
    elif not plan.feasible:
        status = "INFEASIBLE"
        retry_budget = 0
        acceptance = "valid_measurement_precision_infeasible"
        reason = plan.feasibility_reason
    elif ratio is None:
        status = "UNAVAILABLE"
        retry_budget = 0
        acceptance = "valid_stereo_precision_unavailable"
        reason = "post_match_uncertainty_unavailable"
    elif ratio <= 1.0:
        status = "MET"
        retry_budget = 0
        acceptance = "valid_stereo_precision_met"
        reason = "estimated_sigma_within_target"
    else:
        vision_allows_retry = vision_state in {"HIGH", "MEDIUM"}
        retry_budget = max_precision_retry if vision_allows_retry else 0
        if vision_allows_retry and precision_retry_count < max_precision_retry:
            status = "RETRY_STRONGER"
            acceptance = "valid_stereo_retry_for_precision"
            reason = "precision_unmet_visual_evidence_retryable"
        else:
            status = "VALID_BUT_PRECISION_UNMET"
            acceptance = (
                "valid_stereo_precision_unmet_retry_exhausted"
                if precision_retry_count >= max_precision_retry
                else "valid_stereo_precision_unmet_vision_limited"
            )
            reason = "precision_unmet_but_geometry_valid"

    return MeasurementPolicyDecision(
        enabled=True,
        precision_policy_warmup=warmup,
        target_spec=plan.target_spec,
        required_sigma_d_px=plan.required_sigma_disparity_px,
        estimated_sigma_d_px=plan.estimated_sigma_disparity_px,
        estimated_sigma_x_mm=plan.estimated_sigma_x_mm,
        estimated_sigma_y_mm=plan.estimated_sigma_y_mm,
        estimated_sigma_z_mm=plan.estimated_sigma_z_mm,
        precision_ratio=ratio,
        precision_feasible=plan.feasible,
        precision_status=status,
        limiting_axis=plan.limiting_axis,
        base_search_radius_px=base_search_radius_px,
        final_search_radius_px=base_search_radius_px,
        subpixel_policy=("legacy" if refinement_level == 0 else f"stronger_level_{refinement_level}"),
        refinement_level=refinement_level,
        retry_budget=retry_budget,
        precision_retry_count=precision_retry_count,
        acceptance_reason=acceptance,
        policy_reason=reason,
    )
