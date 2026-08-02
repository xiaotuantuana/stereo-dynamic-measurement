from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import MatcherConfig


@dataclass(frozen=True)
class MatchUncertainty:
    disparity_variance_px2: float
    left_position_variance_px2: float
    quality_score: float
    components: dict[str, float]


def estimate_match_uncertainty(
    texture_std: float | None,
    photo_cost: float | None,
    uniqueness_margin: float | None,
    cost_curvature: float | None,
    flow_fb_error_px: float | None,
    right_flow_fb_error_px: float | None,
    lr_error_px: float | None,
    cycle_error_px: float | None,
    icgn_residual: float | None,
    icgn_hessian: float | None,
    config: MatcherConfig,
) -> MatchUncertainty:
    maximum = config.uncertainty_max_component
    epsilon = 1e-9

    def ratio(value: float | None, denominator: float) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        return float(np.clip(abs(value) / max(denominator, epsilon), 0.0, maximum))

    texture_risk = (
        0.0
        if texture_std is None or not np.isfinite(texture_std)
        else float(
            np.clip(
                config.uncertainty_texture_reference / max(texture_std, epsilon),
                0.0,
                maximum,
            )
        )
    )
    photo_risk = ratio(photo_cost, 1.0)
    margin_risk = (
        0.0
        if uniqueness_margin is None or not np.isfinite(uniqueness_margin)
        else float(
            np.clip(
                config.uncertainty_margin_reference / max(uniqueness_margin, epsilon),
                0.0,
                maximum,
            )
        )
    )
    flow_values = [
        ratio(flow_fb_error_px, config.flow_fb_threshold),
        ratio(right_flow_fb_error_px, config.right_flow_fb_threshold),
    ]
    present_flow_count = sum(value is not None for value in (flow_fb_error_px, right_flow_fb_error_px))
    flow_risk = sum(flow_values) / max(present_flow_count, 1)
    lr_risk = ratio(lr_error_px, config.lr_threshold)
    cycle_risk = ratio(cycle_error_px, config.cycle_hard_threshold_px)
    icgn_parts: list[float] = []
    if icgn_residual is not None:
        icgn_parts.append(ratio(icgn_residual, config.icgn_max_residual))
    if icgn_hessian is not None and np.isfinite(icgn_hessian):
        icgn_parts.append(
            float(np.clip(1.0 / max(icgn_hessian, epsilon), 0.0, maximum))
        )
    if cost_curvature is not None and np.isfinite(cost_curvature):
        icgn_parts.append(
            float(np.clip(0.1 / max(cost_curvature, epsilon), 0.0, maximum))
        )
    icgn_risk = float(np.mean(icgn_parts)) if icgn_parts else 0.0

    components = {
        "texture": texture_risk,
        "photo": photo_risk,
        "margin": margin_risk,
        "flow": flow_risk,
        "lr": lr_risk,
        "cycle": cycle_risk,
        "icgn": icgn_risk,
    }
    weighted_risk = (
        config.uncertainty_weight_texture * texture_risk
        + config.uncertainty_weight_photo * photo_risk
        + config.uncertainty_weight_margin * margin_risk
        + config.uncertainty_weight_flow * flow_risk
        + config.uncertainty_weight_lr * lr_risk
        + config.uncertainty_weight_cycle * cycle_risk
        + config.uncertainty_weight_icgn * icgn_risk
    )
    disparity_variance = float(
        np.clip(
            config.uncertainty_base_disparity_variance_px2 * (1.0 + weighted_risk),
            config.uncertainty_min_disparity_variance_px2,
            config.uncertainty_max_disparity_variance_px2,
        )
    )
    left_risk = (
        config.uncertainty_weight_texture * texture_risk
        + config.uncertainty_weight_flow * flow_risk
        + 0.5 * config.uncertainty_weight_cycle * cycle_risk
    )
    left_variance = float(
        np.clip(
            config.uncertainty_base_left_variance_px2 * (1.0 + left_risk),
            config.uncertainty_min_left_variance_px2,
            config.uncertainty_max_left_variance_px2,
        )
    )
    return MatchUncertainty(
        disparity_variance_px2=disparity_variance,
        left_position_variance_px2=left_variance,
        quality_score=float(1.0 / (1.0 + weighted_risk)),
        components=components,
    )
