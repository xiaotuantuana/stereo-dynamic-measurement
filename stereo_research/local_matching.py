from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .icgn import ICGNResult, refine_disparity_icgn
from .models import MatcherConfig, MethodProfile


def sanitize_cost_curvature(
    curvature: float,
    config: MatcherConfig,
) -> float | None:
    if not np.isfinite(curvature):
        return None
    if curvature <= 0:
        return float(config.curvature_min_value)
    return float(
        np.clip(curvature, config.curvature_min_value, config.curvature_max_value)
    )


@dataclass(frozen=True)
class LocalMatchResult:
    status: str
    right_xy: tuple[float, float] | None = None
    disparity: float | None = None
    cost: float | None = None
    confidence: float = 0.0
    lr_error_px: float | None = None
    used_search_radius: int = 0
    neighbor_disparity: float | None = None
    neighbor_disparity_mad: float | None = None
    raw_disparity: float | None = None
    integer_disparity: float | None = None
    measured_disparity: float | None = None
    estimated_disparity: float | None = None
    estimated_right_xy: tuple[float, float] | None = None
    subpixel_offset: float | None = None
    quality_stage: str = "primary"
    second_best_cost: float | None = None
    uniqueness_margin_value: float | None = None
    texture_std: float | None = None
    cost_curvature: float | None = None
    temporal_right_xy: tuple[float, float] | None = None
    cycle_error_px: float | None = None
    right_flow_fb_error_px: float | None = None
    cycle_cost: float | None = None
    icgn_status: str = "not_attempted"
    icgn_converged: bool = False
    icgn_iterations: int = 0
    icgn_residual: float | None = None
    icgn_hessian: float | None = None
    icgn_cost_curvature: float | None = None
    icgn_hessian_density: float | None = None
    icgn_termination_reason: str = ""
    icgn_fallback_used: bool = False
    icgn_fallback_method: str = "none"
    icgn_iterative_disparity: float | None = None
    icgn_final_increment_px: float | None = None
    subpixel_final_status: str = "not_attempted"
    zncc_cost_curvature: float | None = None
    curvature_sample_step_px: float | None = None


@dataclass(frozen=True)
class InitialMatchDiagnostics:
    status: str
    predicted_disparity: float
    predicted_cost: float | None = None
    best_disparity: float | None = None
    best_cost: float | None = None
    second_best_disparity: float | None = None
    second_best_cost: float | None = None
    uniqueness_margin: float | None = None
    texture_std: float | None = None
    candidate_count: int = 0
    predicted_best_disagreement: float | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class _Candidate:
    disparity: int
    vertical_offset: int
    photo_cost: float
    total_cost: float
    cycle_cost: float = 0.0


def predict_disparity(
    latest: float,
    previous: float | None,
    max_velocity: float,
) -> float:
    if previous is None:
        return float(latest)
    velocity = float(np.clip(latest - previous, -max_velocity, max_velocity))
    return float(latest + velocity)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("values and weights must be non-empty one-dimensional arrays of equal length")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("values and weights must be finite, with non-negative weights")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, total * 0.5, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def normalized_constraint_cost(terms: list[tuple[float, float]]) -> float:
    """Combine only active constraints without changing the cost scale."""
    active = [(float(weight), float(cost)) for weight, cost in terms if weight > 0]
    denominator = sum(weight for weight, _ in active)
    if denominator <= 0:
        raise ValueError("At least one positive-weight cost term is required")
    return float(sum(weight * cost for weight, cost in active) / denominator)


class LocalMatcher:
    def __init__(self, config: MatcherConfig):
        self.config = config

    def diagnose_initial(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        predicted_disparity: float,
        *,
        max_disparity: int,
        lr_error_px: float | None,
    ) -> InitialMatchDiagnostics:
        """Score an initialization using runtime evidence without changing its disparity."""

        self._validate_images(left_gray, right_gray)
        left_patch = self._extract_patch(left_gray, left_xy)
        if left_patch is None:
            return InitialMatchDiagnostics("out_of_bounds", predicted_disparity)
        texture_std = float(np.std(left_patch))
        if texture_std < self.config.min_texture_std:
            return InitialMatchDiagnostics(
                "low_texture", predicted_disparity, texture_std=texture_std
            )
        candidates: list[tuple[float, float]] = []
        maximum = min(
            int(max_disparity),
            int(left_xy[0] - self.config.patch_size // 2 - 1),
        )
        for disparity in range(1, maximum + 1):
            right_patch = self._extract_patch(
                right_gray, (left_xy[0] - disparity, left_xy[1])
            )
            if right_patch is not None:
                candidates.append((float(disparity), self._photo_cost(left_patch, right_patch)))
        if not candidates:
            return InitialMatchDiagnostics(
                "out_of_bounds", predicted_disparity, texture_std=texture_std
            )
        ordered = sorted(candidates, key=lambda item: item[1])
        best_disparity, best_cost = ordered[0]
        second = next(
            (item for item in ordered[1:] if abs(item[0] - best_disparity) > 1.0),
            None,
        )
        predicted_disparity_value, predicted_cost = min(
            candidates, key=lambda item: abs(item[0] - predicted_disparity)
        )
        second_disparity = None if second is None else second[0]
        second_cost = None if second is None else second[1]
        margin = None if second_cost is None else max(0.0, second_cost - best_cost)
        texture_score = float(np.clip(
            texture_std / self.config.uncertainty_texture_reference, 0.0, 1.0
        ))
        photo_score = float(np.clip(
            1.0 - predicted_cost / self.config.max_photo_cost, 0.0, 1.0
        ))
        margin_score = 0.0 if margin is None else float(np.clip(
            margin / self.config.uniqueness_margin, 0.0, 1.0
        ))
        lr_score = 0.0 if lr_error_px is None else float(np.clip(
            1.0 - lr_error_px / self.config.lr_threshold, 0.0, 1.0
        ))
        disagreement = abs(predicted_disparity - best_disparity)
        agreement_score = float(np.exp(-disagreement / 2.0))
        confidence = float(min(
            texture_score, photo_score, margin_score, lr_score, agreement_score
        ))
        return InitialMatchDiagnostics(
            status="valid",
            predicted_disparity=predicted_disparity_value,
            predicted_cost=float(predicted_cost),
            best_disparity=best_disparity,
            best_cost=float(best_cost),
            second_best_disparity=second_disparity,
            second_best_cost=None if second_cost is None else float(second_cost),
            uniqueness_margin=margin,
            texture_std=texture_std,
            candidate_count=len(candidates),
            predicted_best_disagreement=disagreement,
            confidence=confidence,
        )

    def match(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        predicted_disparity: float,
        profile: MethodProfile,
        latest_disparity: float | None = None,
        temporal_right_xy: tuple[float, float] | None = None,
        right_flow_fb_error_px: float | None = None,
        search_radius: int | None = None,
        refinement_level: int = 0,
    ) -> LocalMatchResult:
        if refinement_level < 0:
            raise ValueError("refinement_level must be non-negative")
        self._validate_images(left_gray, right_gray)
        left_patch = self._extract_patch(left_gray, left_xy)
        if left_patch is None:
            return LocalMatchResult(status="out_of_bounds")
        texture_std = float(np.std(left_patch))
        if texture_std < self.config.min_texture_std:
            return LocalMatchResult(status="low_texture")

        use_prediction = profile.use_prediction and self.config.enable_prediction
        use_epipolar = profile.use_epipolar and self.config.enable_epipolar
        use_neighborhood = profile.use_neighborhood and self.config.enable_neighborhood
        use_cycle = (
            profile.use_cycle_consistency
            and self.config.enable_cycle_consistency
            and temporal_right_xy is not None
            and right_flow_fb_error_px is not None
            and right_flow_fb_error_px <= self.config.right_flow_fb_threshold
        )
        neighbor_disparity, neighbor_disparity_mad = (
            self._neighborhood_disparity(left_gray, right_gray, left_xy, predicted_disparity)
            if use_neighborhood
            else (None, None)
        )

        primary_radius = self.config.search_radius if search_radius is None else search_radius
        # Fixed M1/M2 calls retain the configured radius. M3/research_full passes
        # its per-frame radius from the confidence-feedback controller.
        fallback_radius = (
            self.config.expanded_search_radius
            if search_radius is None
            else primary_radius
        )
        first = self._search(
            left_patch,
            right_gray,
            left_xy,
            predicted_disparity,
            primary_radius,
            use_prediction,
            use_epipolar,
            neighbor_disparity,
            temporal_right_xy if use_cycle else None,
        )
        if not first:
            first = self._search(
                left_patch,
                right_gray,
                left_xy,
                predicted_disparity,
                fallback_radius,
                use_prediction,
                use_epipolar,
                neighbor_disparity,
                temporal_right_xy if use_cycle else None,
            )
        elif self._best_on_horizontal_boundary(first, predicted_disparity, primary_radius):
            expanded = self._search(
                left_patch,
                right_gray,
                left_xy,
                predicted_disparity,
                fallback_radius,
                use_prediction,
                use_epipolar,
                neighbor_disparity,
                temporal_right_xy if use_cycle else None,
            )
            if expanded:
                first = expanded
        if not first:
            return LocalMatchResult(status="out_of_bounds")

        ordered = sorted(first, key=lambda item: item.total_cost)
        best = ordered[0]
        if best.photo_cost > self.config.max_photo_cost:
            return LocalMatchResult(
                status="ambiguous",
                cost=best.total_cost,
                used_search_radius=self._search_radius(first, predicted_disparity),
                neighbor_disparity=neighbor_disparity,
                texture_std=texture_std,
                temporal_right_xy=temporal_right_xy,
                right_flow_fb_error_px=right_flow_fb_error_px,
            )
        alternatives = [
            item
            for item in ordered[1:]
            if abs(item.disparity - best.disparity) > 1
            or item.vertical_offset != best.vertical_offset
        ]
        second_cost = alternatives[0].total_cost if alternatives else 1.0
        margin = max(0.0, float(second_cost - best.total_cost))
        if margin < self.config.uniqueness_margin:
            return LocalMatchResult(
                status="ambiguous",
                cost=best.total_cost,
                used_search_radius=self._search_radius(first, predicted_disparity),
                neighbor_disparity=neighbor_disparity,
                second_best_cost=second_cost,
                uniqueness_margin_value=margin,
                texture_std=texture_std,
                temporal_right_xy=temporal_right_xy,
                right_flow_fb_error_px=right_flow_fb_error_px,
            )

        disparity = float(best.disparity)
        icgn_result: ICGNResult | None = None
        icgn_output_status = "not_attempted"
        icgn_fallback_used = False
        icgn_fallback_method = "none"
        subpixel_final_status = "not_attempted"
        quality_stage = "primary"
        if profile.use_subpixel and self.config.enable_subpixel:
            subpixel_method = (
                self.config.research_subpixel_method
                if profile.use_icgn and self.config.enable_icgn
                else self.config.subpixel_method
            )
            if subpixel_method == "icgn":
                icgn_result = refine_disparity_icgn(
                    left_gray,
                    right_gray,
                    left_xy,
                    float(best.disparity),
                    float(best.vertical_offset),
                    self.config.icgn_patch_size,
                    self.config.icgn_max_iterations * (1 + refinement_level),
                    self.config.icgn_epsilon / (1 + refinement_level),
                    self.config.icgn_max_offset_px,
                    self.config.icgn_max_residual,
                )
                icgn_output_status = icgn_result.status
                if icgn_result.hessian is not None and icgn_result.hessian < self.config.icgn_min_hessian:
                    icgn_output_status = "icgn_low_hessian"
                if (
                    icgn_output_status == "icgn_converged"
                    and icgn_result.converged
                    and icgn_result.refined_disparity is not None
                ):
                    disparity = float(icgn_result.refined_disparity)
                    quality_stage += "_icgn"
                    subpixel_final_status = "icgn_converged"
                elif self.config.icgn_fallback_method == "line_search":
                    center = (
                        icgn_result.iterative_disparity
                        if icgn_result.iterative_disparity is not None
                        else float(best.disparity)
                    )
                    disparity = self._line_search_refine(
                        left_patch,
                        right_gray,
                        left_xy,
                        center,
                        best.vertical_offset,
                    )
                    icgn_fallback_used = True
                    icgn_fallback_method = "line_search"
                    subpixel_final_status = "line_search_refined"
                    quality_stage += "_icgn_fallback_line_search"
                elif self.config.icgn_fallback_method != "integer":
                    disparity += self._subpixel_offset(
                        left_patch,
                        right_gray,
                        left_xy,
                        best.disparity,
                        best.vertical_offset,
                        method=self.config.icgn_fallback_method,
                    )
                    icgn_fallback_used = True
                    icgn_fallback_method = self.config.icgn_fallback_method
                    subpixel_final_status = (
                        "parabolic_fallback"
                        if self.config.icgn_fallback_method == "parabolic"
                        else "line_search_refined"
                    )
                    quality_stage += f"_icgn_fallback_{self.config.icgn_fallback_method}"
                else:
                    icgn_fallback_used = True
                    icgn_fallback_method = "integer"
                    subpixel_final_status = "integer_fallback"
                    quality_stage += "_icgn_fallback_integer"
            else:
                disparity += self._subpixel_offset(
                    left_patch,
                    right_gray,
                    left_xy,
                    best.disparity,
                    best.vertical_offset,
                    method=subpixel_method,
                )
                subpixel_final_status = f"{subpixel_method}_refined"
        right_xy = (
            float(left_xy[0] - disparity),
            float(left_xy[1] + best.vertical_offset),
        )
        zncc_cost_curvature = (
            self._zncc_cost_curvature(
                left_patch,
                right_gray,
                left_xy,
                disparity,
                best.vertical_offset,
            )
            if profile.use_icgn
            else None
        )
        cycle_error = (
            float(np.linalg.norm(np.asarray(right_xy) - np.asarray(temporal_right_xy)))
            if use_cycle and temporal_right_xy is not None
            else None
        )
        cycle_cost = (
            float(min(cycle_error / self.config.cycle_hard_threshold_px, 1.0))
            if cycle_error is not None
            else None
        )

        lr_error: float | None = None
        if profile.use_lr_check and self.config.enable_lr_check:
            lr_error = self._left_right_error(
                left_gray,
                right_gray,
                left_xy,
                right_xy,
                disparity,
            )
            if lr_error is None or lr_error > self.config.lr_threshold:
                return LocalMatchResult(
                    status="lr_failed",
                    right_xy=right_xy,
                    disparity=disparity,
                    cost=best.total_cost,
                    lr_error_px=lr_error,
                    used_search_radius=self._search_radius(first, predicted_disparity),
                    neighbor_disparity=neighbor_disparity,
                    second_best_cost=second_cost,
                    uniqueness_margin_value=margin,
                    texture_std=texture_std,
                    icgn_status=icgn_output_status,
                    icgn_converged=(icgn_result.converged if icgn_result else False),
                    icgn_iterations=(icgn_result.iterations if icgn_result else 0),
                    icgn_residual=(icgn_result.residual_rms if icgn_result else None),
                    icgn_hessian=(icgn_result.hessian if icgn_result else None),
                    cost_curvature=zncc_cost_curvature,
                    icgn_cost_curvature=zncc_cost_curvature,
                    icgn_hessian_density=(icgn_result.hessian_density if icgn_result else None),
                    icgn_termination_reason=(icgn_result.termination_reason if icgn_result else ""),
                    icgn_fallback_used=icgn_fallback_used,
                    icgn_fallback_method=icgn_fallback_method,
                    icgn_iterative_disparity=(icgn_result.iterative_disparity if icgn_result else None),
                    icgn_final_increment_px=(icgn_result.final_increment_px if icgn_result else None),
                    subpixel_final_status=subpixel_final_status,
                    zncc_cost_curvature=zncc_cost_curvature,
                    curvature_sample_step_px=(
                        self.config.curvature_sample_step_px
                        if zncc_cost_curvature is not None
                        else None
                    ),
                    temporal_right_xy=temporal_right_xy,
                    cycle_error_px=cycle_error,
                    right_flow_fb_error_px=right_flow_fb_error_px,
                    cycle_cost=cycle_cost,
                )

        confidence = float(
            np.clip(
                0.65 * min(margin / 0.2, 1.0)
                + 0.35 * max(0.0, 1.0 - best.photo_cost),
                0.0,
                1.0,
            )
        )
        if use_neighborhood and neighbor_disparity is None:
            confidence *= 0.8
        return LocalMatchResult(
            status="valid",
            right_xy=right_xy,
            disparity=disparity,
            cost=best.total_cost,
            confidence=confidence,
            lr_error_px=lr_error,
            used_search_radius=self._search_radius(first, predicted_disparity),
            neighbor_disparity=neighbor_disparity,
            neighbor_disparity_mad=neighbor_disparity_mad,
            raw_disparity=disparity,
            integer_disparity=float(best.disparity),
            measured_disparity=disparity,
            estimated_disparity=disparity,
            estimated_right_xy=right_xy,
            subpixel_offset=float(disparity - best.disparity),
            quality_stage=quality_stage,
            second_best_cost=second_cost,
            uniqueness_margin_value=margin,
            texture_std=texture_std,
            cost_curvature=zncc_cost_curvature,
            icgn_status=icgn_output_status,
            icgn_converged=(icgn_result.converged if icgn_result else False),
            icgn_iterations=(icgn_result.iterations if icgn_result else 0),
            icgn_residual=(icgn_result.residual_rms if icgn_result else None),
            icgn_hessian=(icgn_result.hessian if icgn_result else None),
            icgn_cost_curvature=zncc_cost_curvature,
            icgn_hessian_density=(icgn_result.hessian_density if icgn_result else None),
            icgn_termination_reason=(icgn_result.termination_reason if icgn_result else ""),
            icgn_fallback_used=icgn_fallback_used,
            icgn_fallback_method=icgn_fallback_method,
            icgn_iterative_disparity=(icgn_result.iterative_disparity if icgn_result else None),
            icgn_final_increment_px=(icgn_result.final_increment_px if icgn_result else None),
            subpixel_final_status=subpixel_final_status,
            zncc_cost_curvature=zncc_cost_curvature,
            curvature_sample_step_px=(
                self.config.curvature_sample_step_px
                if zncc_cost_curvature is not None
                else None
            ),
            temporal_right_xy=temporal_right_xy,
            cycle_error_px=cycle_error,
            right_flow_fb_error_px=right_flow_fb_error_px,
            cycle_cost=cycle_cost,
        )

    def _search(
        self,
        left_patch: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        predicted_disparity: float,
        radius: int,
        use_prediction: bool,
        use_epipolar: bool,
        neighbor_disparity: float | None,
        temporal_right_xy: tuple[float, float] | None = None,
    ) -> list[_Candidate]:
        center = int(round(predicted_disparity))
        candidates: list[_Candidate] = []
        for disparity in range(max(1, center - radius), center + radius + 1):
            for vertical_offset in range(-self.config.vertical_radius, self.config.vertical_radius + 1):
                right_xy = (
                    float(left_xy[0] - disparity),
                    float(left_xy[1] + vertical_offset),
                )
                right_patch = self._extract_patch(right_gray, right_xy)
                if right_patch is None:
                    continue
                photo = self._photo_cost(left_patch, right_patch)
                if use_prediction:
                    temporal = min(abs(disparity - predicted_disparity) / max(radius, 1), 1.0)
                else:
                    temporal = 0.0
                if use_epipolar:
                    epipolar = abs(vertical_offset) / max(self.config.vertical_radius, 1)
                else:
                    epipolar = 0.0
                if neighbor_disparity is not None:
                    neighborhood = min(
                        abs(disparity - neighbor_disparity) / max(radius, 1),
                        1.0,
                    )
                else:
                    neighborhood = 0.0
                cycle_cost = (
                    min(
                        float(np.linalg.norm(np.asarray(right_xy) - np.asarray(temporal_right_xy)))
                        / self.config.cycle_hard_threshold_px,
                        1.0,
                    )
                    if temporal_right_xy is not None
                    else 0.0
                )
                terms = [(self.config.photo_weight, photo)]
                if use_prediction:
                    terms.append((self.config.prediction_weight, temporal))
                if use_epipolar:
                    terms.append((self.config.epipolar_weight, epipolar))
                if neighbor_disparity is not None:
                    terms.append((self.config.neighborhood_weight, neighborhood))
                if temporal_right_xy is not None:
                    terms.append((self.config.cycle_weight, cycle_cost))
                total = normalized_constraint_cost(terms)
                candidates.append(
                    _Candidate(
                        disparity=disparity,
                        vertical_offset=vertical_offset,
                        photo_cost=photo,
                        total_cost=float(total),
                        cycle_cost=float(cycle_cost),
                    )
                )
        return candidates

    def _neighborhood_disparity(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        predicted_disparity: float,
    ) -> tuple[float | None, float | None]:
        x, y = left_xy
        center_intensity = self._sample_intensity(left_gray, left_xy)
        support_points = [
            (x - self.config.support_offset, y),
            (x + self.config.support_offset, y),
            (x, y - self.config.support_offset),
            (x, y + self.config.support_offset),
        ]
        disparities: list[float] = []
        weights: list[float] = []
        for support_xy in support_points:
            patch = self._extract_patch(left_gray, support_xy)
            if patch is None or float(np.std(patch)) < self.config.min_texture_std:
                continue
            intensity_delta = abs(self._sample_intensity(left_gray, support_xy) - center_intensity)
            if intensity_delta > self.config.support_intensity_threshold:
                continue
            best = self._best_photo_disparity(
                patch,
                right_gray,
                support_xy,
                predicted_disparity,
                self.config.search_radius,
            )
            if best is None:
                continue
            disparities.append(float(best))
            weights.append(float(np.exp(-intensity_delta / max(self.config.support_intensity_threshold, 1e-6))))
        if len(disparities) < 3:
            return None, None
        values = np.asarray(disparities, dtype=np.float64)
        median = float(np.median(values))
        return weighted_median(values, np.asarray(weights)), float(np.median(np.abs(values - median)))

    def _best_photo_disparity(
        self,
        left_patch: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        predicted_disparity: float,
        radius: int,
    ) -> int | None:
        center = int(round(predicted_disparity))
        best_disparity: int | None = None
        best_cost = float("inf")
        for disparity in range(max(1, center - radius), center + radius + 1):
            right_patch = self._extract_patch(
                right_gray,
                (left_xy[0] - disparity, left_xy[1]),
            )
            if right_patch is None:
                continue
            cost = self._photo_cost(left_patch, right_patch)
            if cost < best_cost:
                best_cost = cost
                best_disparity = disparity
        return best_disparity

    def _subpixel_offset(
        self,
        left_patch: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        disparity: int,
        vertical_offset: int,
        method: str | None = None,
    ) -> float:
        selected_method = self.config.subpixel_method if method is None else method
        if selected_method == "continuous":
            return self._continuous_subpixel_offset(
                left_patch,
                right_gray,
                left_xy,
                disparity,
                vertical_offset,
            )
        if selected_method == "integer":
            return 0.0
        costs: list[float] = []
        for candidate in (disparity - 1, disparity, disparity + 1):
            patch = self._extract_patch(
                right_gray,
                (left_xy[0] - candidate, left_xy[1] + vertical_offset),
            )
            if patch is None:
                return 0.0
            costs.append(self._photo_cost(left_patch, patch))
        c_minus, c_zero, c_plus = costs
        denominator = c_minus - 2.0 * c_zero + c_plus
        if denominator <= 1e-9:
            return 0.0
        offset = 0.5 * (c_minus - c_plus) / denominator
        return float(np.clip(offset, -1.0, 1.0))

    def _continuous_subpixel_offset(
        self,
        left_patch: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        disparity: int,
        vertical_offset: int,
    ) -> float:
        offsets = np.arange(-1.0, 1.0 + self.config.subpixel_step * 0.5, self.config.subpixel_step)
        best_offset = 0.0
        best_cost = float("inf")
        for offset in offsets:
            patch = self._extract_patch(
                right_gray,
                (
                    left_xy[0] - (disparity + float(offset)),
                    left_xy[1] + vertical_offset,
                ),
            )
            if patch is None:
                continue
            cost = self._photo_cost(left_patch, patch)
            if cost < best_cost:
                best_cost = cost
                best_offset = float(offset)
        return float(np.clip(best_offset, -1.0, 1.0))

    def _line_search_refine(
        self,
        left_patch: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        center_disparity: float,
        vertical_offset: int,
    ) -> float:
        radius = self.config.icgn_fallback_search_radius_px
        values = np.linspace(
            center_disparity - radius,
            center_disparity + radius,
            self.config.icgn_fallback_sample_count,
        )
        costs = np.full(values.shape, np.inf, dtype=np.float64)
        for index, candidate in enumerate(values):
            patch = self._extract_patch(
                right_gray,
                (left_xy[0] - float(candidate), left_xy[1] + vertical_offset),
            )
            if patch is not None:
                costs[index] = self._zncc_cost(left_patch, patch)
        if not np.isfinite(costs).any():
            return float(center_disparity)
        best_index = int(np.nanargmin(costs))
        best_disparity = float(values[best_index])
        if best_index == 0 or best_index == len(values) - 1:
            return best_disparity
        c_minus, c_zero, c_plus = costs[best_index - 1:best_index + 2]
        denominator = c_minus - 2.0 * c_zero + c_plus
        if not np.isfinite(denominator) or denominator <= 1e-12:
            return best_disparity
        sample_step = float(values[1] - values[0])
        fractional = 0.5 * (c_minus - c_plus) / denominator
        return float(best_disparity + np.clip(fractional, -1.0, 1.0) * sample_step)

    def _zncc_cost_curvature(
        self,
        left_patch: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        disparity: float,
        vertical_offset: int,
    ) -> float | None:
        delta = self.config.curvature_sample_step_px
        costs: list[float] = []
        for candidate in (disparity - delta, disparity, disparity + delta):
            patch = self._extract_patch(
                right_gray,
                (left_xy[0] - candidate, left_xy[1] + vertical_offset),
            )
            if patch is None:
                return None
            costs.append(self._zncc_cost(left_patch, patch))
        c_minus, c_zero, c_plus = costs
        raw_curvature = (c_minus - 2.0 * c_zero + c_plus) / (delta * delta)
        return sanitize_cost_curvature(raw_curvature, self.config)

    @staticmethod
    def _zncc_cost(left_patch: np.ndarray, right_patch: np.ndarray) -> float:
        left_f = left_patch.astype(np.float64, copy=False)
        right_f = right_patch.astype(np.float64, copy=False)
        left_centered = left_f - float(left_f.mean())
        right_centered = right_f - float(right_f.mean())
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        if denominator <= 1e-12:
            return float("inf")
        zncc = float(np.sum(left_centered * right_centered) / denominator)
        return float(np.clip((1.0 - zncc) * 0.5, 0.0, 1.0))

    def _left_right_error(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        original_left_xy: tuple[float, float],
        right_xy: tuple[float, float],
        disparity: float,
    ) -> float | None:
        right_patch = self._extract_patch(right_gray, right_xy)
        if right_patch is None:
            return None
        center = int(round(disparity))
        best: tuple[float, float, float] | None = None
        for candidate in range(max(1, center - self.config.search_radius), center + self.config.search_radius + 1):
            for vertical_offset in range(-self.config.vertical_radius, self.config.vertical_radius + 1):
                left_xy = (
                    right_xy[0] + candidate,
                    right_xy[1] + vertical_offset,
                )
                left_patch = self._extract_patch(left_gray, left_xy)
                if left_patch is None:
                    continue
                photo_cost = self._photo_cost(right_patch, left_patch)
                epipolar_cost = abs(vertical_offset) / max(self.config.vertical_radius, 1)
                terms = [
                    (self.config.photo_weight, photo_cost),
                    (self.config.epipolar_weight, epipolar_cost),
                ]
                cost = normalized_constraint_cost(terms)
                if best is None or cost < best[0]:
                    best = (cost, left_xy[0], left_xy[1])
        if best is None:
            return None
        spatial_error = float(
            np.hypot(best[1] - original_left_xy[0], best[2] - original_left_xy[1])
        )
        reverse_disparity = best[1] - right_xy[0]
        return max(spatial_error, abs(reverse_disparity - disparity))

    def _extract_patch(
        self,
        image: np.ndarray,
        center: tuple[float, float],
    ) -> np.ndarray | None:
        radius = self.config.patch_size // 2
        x, y = center
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius >= image.shape[1]
            or y + radius >= image.shape[0]
        ):
            return None
        return cv2.getRectSubPix(
            image,
            (self.config.patch_size, self.config.patch_size),
            (float(x), float(y)),
        )

    @staticmethod
    def _sample_intensity(image: np.ndarray, xy: tuple[float, float]) -> float:
        x = int(np.clip(round(xy[0]), 0, image.shape[1] - 1))
        y = int(np.clip(round(xy[1]), 0, image.shape[0] - 1))
        return float(image[y, x])

    @staticmethod
    def _photo_cost(left_patch: np.ndarray, right_patch: np.ndarray) -> float:
        left_f = left_patch.astype(np.float32)
        right_f = right_patch.astype(np.float32)
        left_bits = left_f.reshape(-1) >= left_f[left_f.shape[0] // 2, left_f.shape[1] // 2]
        right_bits = right_f.reshape(-1) >= right_f[right_f.shape[0] // 2, right_f.shape[1] // 2]
        census = float(np.mean(left_bits != right_bits))
        left_centered = left_f - float(left_f.mean())
        right_centered = right_f - float(right_f.mean())
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        zncc = 0.0 if denominator <= 1e-9 else float(
            np.sum(left_centered * right_centered) / denominator
        )
        zncc_cost = float(np.clip((1.0 - zncc) * 0.5, 0.0, 1.0))
        return float(0.6 * census + 0.4 * zncc_cost)

    @staticmethod
    def _validate_images(left_gray: np.ndarray, right_gray: np.ndarray) -> None:
        if left_gray.ndim != 2 or right_gray.ndim != 2:
            raise ValueError("LocalMatcher expects grayscale images")
        if left_gray.shape != right_gray.shape:
            raise ValueError("Left and right images must have identical shapes")

    @staticmethod
    def _best_on_horizontal_boundary(
        candidates: list[_Candidate],
        predicted_disparity: float,
        radius: int,
    ) -> bool:
        best = min(candidates, key=lambda item: item.total_cost)
        center = int(round(predicted_disparity))
        return best.disparity in {max(1, center - radius), center + radius}

    @staticmethod
    def _search_radius(candidates: list[_Candidate], predicted_disparity: float) -> int:
        center = int(round(predicted_disparity))
        return max(abs(item.disparity - center) for item in candidates)


class QualityLocalMatcher:
    """Quality-first cascade that keeps the default matcher as its primary stage."""

    def __init__(self, config: MatcherConfig):
        self.config = config
        self.primary = LocalMatcher(config)
        context_patch_size = max(
            config.quality_context_patch_size,
            config.patch_size + 4,
        )
        self.context = LocalMatcher(
            replace(config, patch_size=context_patch_size)
        )

    def match(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        predicted_disparity: float,
        profile: MethodProfile,
        latest_disparity: float | None = None,
        temporal_right_xy: tuple[float, float] | None = None,
        right_flow_fb_error_px: float | None = None,
        search_radius: int | None = None,
        refinement_level: int = 0,
    ) -> LocalMatchResult:
        primary = self.primary.match(
            left_gray,
            right_gray,
            left_xy,
            predicted_disparity,
            profile,
            latest_disparity=latest_disparity,
            temporal_right_xy=temporal_right_xy,
            right_flow_fb_error_px=right_flow_fb_error_px,
            search_radius=search_radius,
            refinement_level=refinement_level,
        )
        if primary.status == "valid":
            result = replace(
                primary,
                quality_stage=(
                    primary.quality_stage
                    if primary.quality_stage != "primary"
                    else "primary"
                ),
            )
        else:
            context = self.context.match(
                left_gray,
                right_gray,
                left_xy,
                predicted_disparity,
                profile,
                latest_disparity=latest_disparity,
                temporal_right_xy=temporal_right_xy,
                right_flow_fb_error_px=right_flow_fb_error_px,
                search_radius=search_radius,
                refinement_level=refinement_level,
            )
            if context.status == "valid":
                result = replace(context, quality_stage="context_recovery")
            elif self.config.enable_pyramid:
                pyramid_disparity = self._pyramid_disparity(
                    left_gray,
                    right_gray,
                    left_xy,
                    predicted_disparity,
                )
                if pyramid_disparity is None:
                    return primary
                recovered = self.primary.match(
                    left_gray,
                    right_gray,
                    left_xy,
                    pyramid_disparity,
                    profile,
                    latest_disparity=latest_disparity,
                    temporal_right_xy=temporal_right_xy,
                    right_flow_fb_error_px=right_flow_fb_error_px,
                    search_radius=search_radius,
                    refinement_level=refinement_level,
                )
                if recovered.status != "valid":
                    return primary
                result = replace(recovered, quality_stage="pyramid_recovery")
            else:
                return primary

        raw_disparity = result.measured_disparity
        if raw_disparity is None:
            raw_disparity = result.disparity
        if raw_disparity is None or result.right_xy is None:
            return result
        raw_disparity = float(raw_disparity)
        if (
            latest_disparity is None
            or not self.config.enable_temporal_estimation
        ):
            return replace(
                result,
                disparity=raw_disparity,
                raw_disparity=raw_disparity,
                measured_disparity=raw_disparity,
                estimated_disparity=raw_disparity,
                estimated_right_xy=result.right_xy,
            )
        if (
            abs(raw_disparity - latest_disparity)
            > self.config.quality_max_smoothing_innovation_px
        ):
            if result.quality_stage in {"context_recovery", "pyramid_recovery"}:
                if self.config.enable_pyramid:
                    verifier_disparity = self._pyramid_disparity(
                        left_gray,
                        right_gray,
                        left_xy,
                        predicted_disparity,
                    )
                else:
                    verifier_disparity = (
                        result.neighbor_disparity
                        if result.neighbor_disparity is not None
                        else predicted_disparity
                    )
            else:
                verifier = self.context.match(
                    left_gray,
                    right_gray,
                    left_xy,
                    predicted_disparity,
                    profile,
                    latest_disparity=latest_disparity,
                    temporal_right_xy=temporal_right_xy,
                    right_flow_fb_error_px=right_flow_fb_error_px,
                )
                verifier_disparity = (
                    verifier.measured_disparity
                    if verifier.measured_disparity is not None
                    else verifier.disparity
                ) if verifier.status == "valid" else None
            if (
                verifier_disparity is None
                or abs(verifier_disparity - raw_disparity)
                > self.config.quality_motion_agreement_px
            ):
                return replace(
                    result,
                    status="ambiguous",
                    raw_disparity=raw_disparity,
                    measured_disparity=raw_disparity,
                    estimated_disparity=None,
                    estimated_right_xy=None,
                    quality_stage=(
                        "context_motion_rejected"
                        if result.quality_stage == "context_recovery"
                        else "motion_rejected"
                    ),
                )
            return replace(
                result,
                disparity=raw_disparity,
                raw_disparity=raw_disparity,
                measured_disparity=raw_disparity,
                estimated_disparity=raw_disparity,
                estimated_right_xy=result.right_xy,
                quality_stage=(
                    "context_motion_verified"
                    if result.quality_stage == "context_recovery"
                    else (
                        "pyramid_recovery"
                        if result.quality_stage == "pyramid_recovery"
                        else "motion_verified"
                    )
                ),
            )
        return replace(
            result,
            disparity=raw_disparity,
            raw_disparity=raw_disparity,
            measured_disparity=raw_disparity,
            estimated_disparity=raw_disparity,
            estimated_right_xy=result.right_xy,
        )

    def _pyramid_disparity(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_xy: tuple[float, float],
        predicted_disparity: float,
    ) -> float | None:
        levels = max(1, self.config.pyramid_levels)
        left_pyramid = [left_gray]
        right_pyramid = [right_gray]
        for _ in range(1, levels):
            left_pyramid.append(cv2.pyrDown(left_pyramid[-1]))
            right_pyramid.append(cv2.pyrDown(right_pyramid[-1]))
        estimate = float(predicted_disparity) / (2 ** (levels - 1))
        for level in range(levels - 1, -1, -1):
            scale = float(2**level)
            if level < levels - 1:
                estimate *= 2.0
            left_level = left_pyramid[level]
            right_level = right_pyramid[level]
            xy = (left_xy[0] / scale, left_xy[1] / scale)
            patch_size = min(self.config.patch_size, 7 if level > 0 else self.config.patch_size)
            if patch_size % 2 == 0:
                patch_size -= 1
            level_matcher = LocalMatcher(replace(self.config, patch_size=max(3, patch_size)))
            left_patch = level_matcher._extract_patch(left_level, xy)
            if left_patch is None or float(np.std(left_patch)) < self.config.min_texture_std:
                return None
            radius = (
                self.config.pyramid_search_radius
                if level == levels - 1
                else self.config.pyramid_refine_radius
            )
            best = level_matcher._best_photo_disparity(
                left_patch,
                right_level,
                xy,
                estimate,
                radius,
            )
            if best is None:
                return None
            estimate = float(best)
        return estimate
