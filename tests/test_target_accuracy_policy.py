from __future__ import annotations

from stereo_dynamic_measurement.innovation1.precision_planner import (
    TargetAccuracySpec,
    build_precision_plan,
)
from stereo_research.accuracy_policy import decide_measurement_policy


def _plan(target_z_mm: float, estimated_sigma_d_px: float = 0.05):
    return build_precision_plan(
        TargetAccuracySpec(target_sigma_z_mm=target_z_mm),
        current_depth_m=2.0,
        focal_length_px=800.0,
        baseline_mm=120.0,
        estimated_sigma_disparity_px=estimated_sigma_d_px,
        minimum_achievable_sigma_disparity_px=0.01,
    )


def test_stricter_target_never_requests_less_precision_effort() -> None:
    loose = decide_measurement_policy(
        _plan(5.0), base_search_radius_px=4, vision_state="HIGH",
        max_precision_retry=2, max_refinement_level=2,
    )
    medium = decide_measurement_policy(
        _plan(1.0), base_search_radius_px=4, vision_state="HIGH",
        max_precision_retry=2, max_refinement_level=2,
    )
    strict = decide_measurement_policy(
        _plan(0.5), base_search_radius_px=4, vision_state="HIGH",
        max_precision_retry=2, max_refinement_level=2,
    )

    assert loose.required_sigma_d_px > medium.required_sigma_d_px > strict.required_sigma_d_px
    assert loose.refinement_level <= medium.refinement_level <= strict.refinement_level
    assert loose.retry_budget <= medium.retry_budget <= strict.retry_budget
    assert {loose.precision_status, medium.precision_status, strict.precision_status} >= {"MET", "RETRY_STRONGER"}


def test_vision_state_controls_motion_radius_without_being_overridden_by_target() -> None:
    good_loose = decide_measurement_policy(
        _plan(5.0), base_search_radius_px=4, vision_state="HIGH",
        max_precision_retry=1, max_refinement_level=2,
    )
    good_strict = decide_measurement_policy(
        _plan(0.5), base_search_radius_px=4, vision_state="HIGH",
        max_precision_retry=1, max_refinement_level=2,
    )
    poor_loose = decide_measurement_policy(
        _plan(5.0), base_search_radius_px=16, vision_state="LOW",
        max_precision_retry=1, max_refinement_level=2,
    )
    poor_strict = decide_measurement_policy(
        _plan(0.5), base_search_radius_px=16, vision_state="LOW",
        max_precision_retry=1, max_refinement_level=2,
    )

    assert good_loose.final_search_radius_px == good_strict.final_search_radius_px == 4
    assert poor_loose.final_search_radius_px == poor_strict.final_search_radius_px == 16
    assert good_strict.refinement_level > good_loose.refinement_level
    assert poor_strict.refinement_level > poor_loose.refinement_level
    assert good_strict.retry_budget == 1
    assert poor_strict.retry_budget == 0


def test_infeasible_target_is_honest_and_never_retries() -> None:
    plan = build_precision_plan(
        TargetAccuracySpec(target_sigma_z_mm=0.001),
        current_depth_m=5.0,
        focal_length_px=800.0,
        baseline_mm=120.0,
        estimated_sigma_disparity_px=0.1,
        minimum_achievable_sigma_disparity_px=0.05,
    )

    decision = decide_measurement_policy(
        plan, base_search_radius_px=8, vision_state="HIGH",
        max_precision_retry=3, max_refinement_level=2,
    )

    assert decision.precision_status == "INFEASIBLE"
    assert not decision.precision_feasible
    assert decision.retry_budget == 0
    assert decision.precision_retry_count == 0
    assert decision.acceptance_reason == "valid_measurement_precision_infeasible"


def test_retry_is_bounded_and_unmet_stays_distinct_from_stereo_validity() -> None:
    plan = _plan(0.5)
    first = decide_measurement_policy(
        plan, base_search_radius_px=8, vision_state="MEDIUM",
        max_precision_retry=1, max_refinement_level=2, precision_retry_count=0,
    )
    exhausted = decide_measurement_policy(
        plan, base_search_radius_px=8, vision_state="MEDIUM",
        max_precision_retry=1, max_refinement_level=2, precision_retry_count=1,
    )

    assert first.precision_status == "RETRY_STRONGER"
    assert exhausted.precision_status == "VALID_BUT_PRECISION_UNMET"
    assert exhausted.precision_retry_count == 1
    assert exhausted.acceptance_reason == "valid_stereo_precision_unmet_retry_exhausted"
