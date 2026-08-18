from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    min_radius_px: int = 2
    max_radius_px: int = 32
    nominal_fb_error_px: float = 0.5
    nominal_prediction_residual_px: float = 1.0
    nominal_precision_px: float = 0.05

    def __post_init__(self) -> None:
        if self.min_radius_px < 1 or self.max_radius_px < self.min_radius_px:
            raise ValueError("Search radii must be positive and ordered")


@dataclass(frozen=True)
class SearchRadiusDecision:
    radius_px: int
    risk_score: float
    reasons: tuple[str, ...]


def compute_search_radius(config: AdaptiveSearchConfig, *, gradient_quality: float, fb_error_px: float | None, prediction_residual_px: float | None, previous_confidence: float, required_disparity_precision_px: float) -> SearchRadiusDecision:
    """Derive an actual local disparity radius from quality, state and accuracy demand."""
    quality = float(np.clip(gradient_quality, 0.0, 1.0))
    confidence = float(np.clip(previous_confidence, 0.0, 1.0))
    fb_risk = 1.0 if fb_error_px is None else float(np.clip(fb_error_px / config.nominal_fb_error_px, 0.0, 2.0)) / 2.0
    temporal_risk = 1.0 if prediction_residual_px is None else float(np.clip(prediction_residual_px / config.nominal_prediction_residual_px, 0.0, 2.0)) / 2.0
    precision_risk = float(np.clip(config.nominal_precision_px / max(required_disparity_precision_px, 1e-6), 0.0, 3.0)) / 3.0
    risk = float(np.clip(0.25 * (1.0 - quality) + 0.20 * fb_risk + 0.25 * temporal_risk + 0.20 * (1.0 - confidence) + 0.10 * precision_risk, 0.0, 1.0))
    radius = int(round(config.min_radius_px + risk * (config.max_radius_px - config.min_radius_px)))
    reasons: list[str] = []
    if quality < 0.4: reasons.append("low_texture")
    if fb_error_px is None or fb_error_px > config.nominal_fb_error_px: reasons.append("flow_uncertain")
    if prediction_residual_px is None or prediction_residual_px > config.nominal_prediction_residual_px: reasons.append("prediction_error")
    if confidence < 0.5: reasons.append("low_confidence")
    if required_disparity_precision_px < config.nominal_precision_px: reasons.append("target_precision")
    return SearchRadiusDecision(int(np.clip(radius, config.min_radius_px, config.max_radius_px)), risk, tuple(reasons))
