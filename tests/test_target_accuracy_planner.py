from __future__ import annotations

import inspect

import pytest

from stereo_dynamic_measurement.innovation1.precision_planner import (
    TargetAccuracySpec,
    build_precision_plan,
)


def test_target_metric_is_explicitly_standard_uncertainty_only() -> None:
    spec = TargetAccuracySpec(metric="sigma", target_sigma_z_mm=3.0)

    assert spec.metric == "sigma"
    with pytest.raises(ValueError, match="sigma"):
        TargetAccuracySpec(metric="rmse", target_sigma_z_mm=3.0)
    with pytest.raises(ValueError, match="positive target"):
        TargetAccuracySpec(metric="sigma", target_sigma_z_mm=0.0)


def test_precision_plan_matches_worked_depth_sigma_example() -> None:
    plan = build_precision_plan(
        TargetAccuracySpec(metric="sigma", target_sigma_z_mm=3.0),
        current_depth_m=3.0,
        focal_length_px=800.0,
        baseline_mm=120.0,
        estimated_sigma_disparity_px=0.05,
        minimum_achievable_sigma_disparity_px=0.02,
    )

    assert plan.required_sigma_disparity_px == pytest.approx(0.032)
    assert plan.estimated_sigma_z_mm == pytest.approx(4.6875)
    assert plan.precision_ratio == pytest.approx(1.5625)
    assert plan.feasible
    assert plan.limiting_axis == "z"
    assert plan.estimated_sigma_x_mm is None
    assert plan.estimated_sigma_y_mm is None


def test_required_disparity_sigma_has_target_distance_and_geometry_monotonicity() -> None:
    def required(target_z_mm: float, depth_m: float, focal_px: float, baseline_mm: float) -> float:
        return build_precision_plan(
            TargetAccuracySpec(target_sigma_z_mm=target_z_mm),
            current_depth_m=depth_m,
            focal_length_px=focal_px,
            baseline_mm=baseline_mm,
            estimated_sigma_disparity_px=0.05,
            minimum_achievable_sigma_disparity_px=0.01,
        ).required_sigma_disparity_px

    assert required(1.0, 3.0, 800.0, 120.0) < required(3.0, 3.0, 800.0, 120.0) < required(10.0, 3.0, 800.0, 120.0)
    assert required(3.0, 5.0, 800.0, 120.0) < required(3.0, 3.0, 800.0, 120.0)
    assert required(3.0, 3.0, 1200.0, 120.0) > required(3.0, 3.0, 800.0, 120.0)
    assert required(3.0, 3.0, 800.0, 180.0) > required(3.0, 3.0, 800.0, 120.0)


def test_impossible_target_is_reported_without_claiming_attainment() -> None:
    plan = build_precision_plan(
        TargetAccuracySpec(target_sigma_z_mm=0.01),
        current_depth_m=5.0,
        focal_length_px=800.0,
        baseline_mm=120.0,
        estimated_sigma_disparity_px=0.1,
        minimum_achievable_sigma_disparity_px=0.05,
    )

    assert not plan.feasible
    assert plan.precision_ratio > 1.0
    assert "floor" in plan.feasibility_reason


def test_runtime_precision_planner_has_no_ground_truth_inputs() -> None:
    names = inspect.signature(build_precision_plan).parameters

    forbidden = ("gt_", "ground_truth", "truth", "label", "actual_error")
    assert all(not any(token in name.lower() for token in forbidden) for name in names)
