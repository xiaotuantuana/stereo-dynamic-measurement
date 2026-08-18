from __future__ import annotations

from dataclasses import dataclass


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
