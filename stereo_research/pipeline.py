from __future__ import annotations

import time
from dataclasses import replace
from typing import Iterable

import cv2
import numpy as np

from .camera_compensation import apply_rigid_transform, estimate_camera_compensation
from .cycle_consistency import evaluate_cycle_consistency
from .filtering import AdaptivePointKalman
from .geometry import reproject_point_m
from .global_matching import GlobalSample, GlobalStereoMatcher, GlobalStereoResult
from .local_matching import LocalMatcher, QualityLocalMatcher, predict_disparity
from .models import (
    FramePointResult,
    MatcherConfig,
    MethodName,
    PointSpec,
    PointState,
    method_profile,
)
from .tracking import track_point_lk, track_xy_lk
from .uncertainty import estimate_match_uncertainty


class TemporalStereoPipeline:
    def __init__(
        self,
        method: MethodName,
        q: np.ndarray,
        calibration_unit: str,
        config: MatcherConfig | None = None,
    ):
        self.method = method
        self.profile = method_profile(method)
        self.q = np.asarray(q, dtype=np.float64)
        self.calibration_unit = calibration_unit
        self.config = config or MatcherConfig()
        self.global_matcher = GlobalStereoMatcher(self.config)
        self.local_matcher = (
            QualityLocalMatcher(self.config)
            if method in {"full_quality", "research_full"}
            else LocalMatcher(self.config)
        )
        self.recovery_matcher = QualityLocalMatcher(
            replace(
                self.config,
                search_radius=self.config.expanded_search_radius,
                expanded_search_radius=self.config.recovery_search_radius,
                enable_pyramid=True,
            )
        )
        self.states: dict[str, PointState] = {}
        self.previous_left_gray: np.ndarray | None = None
        self.previous_right_gray: np.ndarray | None = None
        self.flow_reference_gray: dict[str, np.ndarray] = {}
        self.right_flow_reference_gray: dict[str, np.ndarray] = {}
        self.filters: dict[str, AdaptivePointKalman] = {}
        self.initialized = False

    def initialize(
        self,
        left: np.ndarray,
        right: np.ndarray,
        points: Iterable[PointSpec],
        frame: int,
    ) -> list[FramePointResult]:
        left_gray, right_gray = self._gray_pair(left, right)
        specs = tuple(points)
        if not specs:
            raise ValueError("At least one point is required")
        global_result = self.global_matcher.compute(left_gray, right_gray)
        initialization_matching_ms = global_result.elapsed_ms
        samples: dict[str, GlobalSample] = {}
        for point in specs:
            patch_status = self._initial_patch_status(left_gray, point.xy)
            samples[point.point_id] = (
                self.global_matcher.sample_initial(global_result, point.xy)
                if patch_status == "valid"
                else GlobalSample(status=patch_status)
            )
        if any(sample.status in {"saturated", "ambiguous", "lr_failed"} for sample in samples.values()):
            expanded = self.global_matcher.compute(
                left_gray,
                right_gray,
                self.config.expanded_num_disparities,
            )
            for point in specs:
                if samples[point.point_id].status in {"saturated", "ambiguous", "lr_failed"}:
                    samples[point.point_id] = self.global_matcher.sample_initial(expanded, point.xy)
            initialization_matching_ms += expanded.elapsed_ms

        results: list[FramePointResult] = []
        for point in specs:
            state = PointState(
                point_id=point.point_id,
                initial_left_xy=point.xy,
                left_xy=point.xy,
                point_role=point.role,
            )
            self.states[point.point_id] = state
            self.flow_reference_gray[point.point_id] = left_gray.copy()
            sample = samples[point.point_id]
            if sample.status != "valid" or sample.disparity is None or sample.right_xy is None:
                state.record_failure(sample.status, self.config.max_failures)
                results.append(
                    FramePointResult.invalid(
                        method=self.method,
                        frame=frame,
                        point_state=state,
                        status="lost" if state.status == "lost" else sample.status,
                        matching_ms=initialization_matching_ms,
                        total_ms=initialization_matching_ms,
                    )
                )
                continue
            result = self._valid_result(
                    frame,
                    state,
                    sample.right_xy,
                    sample.disparity,
                    match_cost=None,
                    lr_error=sample.lr_error_px,
                    confidence=1.0,
                    flow_ms=0.0,
                    matching_ms=initialization_matching_ms,
                    total_ms=initialization_matching_ms,
                )
            if result.status == "valid":
                result = self._finalize_right_flow_reference(result, frame, right_gray)
            results.append(result)
        self.previous_left_gray = left_gray.copy()
        self.previous_right_gray = right_gray.copy()
        self.initialized = True
        return self._apply_camera_compensation(results)

    def step(
        self,
        left: np.ndarray,
        right: np.ndarray,
        frame: int,
    ) -> list[FramePointResult]:
        if not self.initialized or self.previous_left_gray is None:
            raise RuntimeError("Pipeline must be initialized before step()")
        left_gray, right_gray = self._gray_pair(left, right)
        dense_result: GlobalStereoResult | None = None
        if self.profile.dense_sgbm_each_frame:
            dense_result = self.global_matcher.compute(left_gray, right_gray)
        shared_matching_ms = 0.0 if dense_result is None else dense_result.elapsed_ms
        results: list[FramePointResult] = []
        for state in self.states.values():
            point_start = time.perf_counter()
            recovery_attempt = False
            recovery_stage = ""
            if state.status == "lost":
                can_attempt = (
                    self.method in {"full_quality", "research_full"}
                    and self.config.enable_recovery
                    and frame % self.config.recovery_interval_frames == 0
                )
                if not can_attempt:
                    results.append(
                        FramePointResult.invalid(
                            self.method,
                            frame,
                            state,
                            "lost",
                            matching_ms=shared_matching_ms,
                            total_ms=shared_matching_ms
                            + (time.perf_counter() - point_start) * 1000.0,
                            recovery_stage="waiting",
                        )
                    )
                    continue
                recovery_attempt = True
                recovery_stage = "attempt"
                state.recovery_attempt_count += 1
                if state.recovery_started_frame is None:
                    state.recovery_started_frame = frame
            elif (
                state.status == "recovering"
                and self.method in {"full_quality", "research_full"}
                and self.config.enable_recovery
            ):
                recovery_attempt = True
                recovery_stage = "attempt"
                state.recovery_attempt_count += 1
                if state.recovery_started_frame is None:
                    state.recovery_started_frame = frame

            flow_ms = 0.0
            flow_error: float | None = None
            temporal_right_xy: tuple[float, float] | None = None
            right_flow_error: float | None = None
            cycle_status = "not_used"
            if self.profile.use_flow and self.config.enable_flow:
                flow_start = time.perf_counter()
                flow = track_point_lk(
                    self.flow_reference_gray[state.point_id],
                    left_gray,
                    state,
                    self.config,
                    recovery=recovery_attempt,
                )
                flow_ms = (time.perf_counter() - flow_start) * 1000.0
                flow_error = flow.fb_error_px
                if flow.status != "valid" or flow.left_xy is None:
                    state.record_failure(flow.status, self.config.max_failures)
                    results.append(
                        FramePointResult.invalid(
                            self.method,
                            frame,
                            state,
                            "lost" if state.status == "lost" else flow.status,
                            flow_ms=flow_ms,
                            matching_ms=shared_matching_ms,
                            total_ms=shared_matching_ms
                            + (time.perf_counter() - point_start) * 1000.0,
                            flow_fb_error_px=flow_error,
                            recovery_stage=recovery_stage,
                        )
                    )
                    continue
                state.record_flow(flow.left_xy)
                self.flow_reference_gray[state.point_id] = left_gray.copy()
            else:
                state.left_xy = state.initial_left_xy

            if self.profile.use_cycle_consistency and self.config.enable_cycle_consistency:
                reference_image, reference_xy, reference_status = self._right_flow_reference(
                    state.point_id,
                    current_frame=frame,
                )
                if reference_status == "invalid_right_reference":
                    cycle_status = reference_status
                elif reference_status != "valid":
                    cycle_status = "not_attempted"
            else:
                reference_image, reference_xy, reference_status = None, None, "not_enabled"
            if reference_status == "valid":
                assert reference_image is not None and reference_xy is not None
                right_flow_start = time.perf_counter()
                right_flow = track_xy_lk(
                    reference_image,
                    right_gray,
                    previous_xy=reference_xy,
                    initial_velocity=state.right_velocity,
                    config=self.config,
                    recovery=recovery_attempt,
                    fb_threshold=self.config.right_flow_fb_threshold,
                )
                flow_ms += (time.perf_counter() - right_flow_start) * 1000.0
                right_flow_error = right_flow.fb_error_px
                if right_flow.status == "valid" and right_flow.left_xy is not None:
                    temporal_right_xy = right_flow.left_xy
                else:
                    cycle_status = "right_flow_failed"

            if self.profile.dense_sgbm_each_frame:
                assert dense_result is not None
                sample = self.global_matcher.sample_dense(dense_result, state.left_xy)
                matching_ms = dense_result.elapsed_ms
                if sample.status != "valid" or sample.disparity is None or sample.right_xy is None:
                    state.record_failure(sample.status, self.config.max_failures)
                    results.append(
                        FramePointResult.invalid(
                            self.method,
                            frame,
                            state,
                            "lost" if state.status == "lost" else sample.status,
                            flow_ms=flow_ms,
                            matching_ms=matching_ms,
                            total_ms=shared_matching_ms
                            + (time.perf_counter() - point_start) * 1000.0,
                            flow_fb_error_px=flow_error,
                        )
                    )
                    continue
                results.append(
                    self._valid_result(
                        frame,
                        state,
                        sample.right_xy,
                        sample.disparity,
                        None,
                        sample.lr_error_px,
                        1.0,
                        flow_ms,
                        matching_ms,
                        shared_matching_ms
                        + (time.perf_counter() - point_start) * 1000.0,
                        flow_error,
                    )
                )
                continue

            prediction_history = (
                state.estimated_disparity_history
                if self.method == "research_full" and state.estimated_disparity_history
                else state.disparity_history
            )
            latest = prediction_history[-1] if prediction_history else None
            if latest is None:
                if recovery_attempt:
                    recovery_start = time.perf_counter()
                    recovery_dense = self.global_matcher.compute(left_gray, right_gray)
                    recovery_sample = self.global_matcher.sample_initial(
                        recovery_dense,
                        state.left_xy,
                    )
                    if recovery_sample.status != "valid":
                        recovery_dense = self.global_matcher.compute(
                            left_gray,
                            right_gray,
                            self.config.expanded_num_disparities,
                        )
                        recovery_sample = self.global_matcher.sample_initial(
                            recovery_dense,
                            state.left_xy,
                        )
                    recovery_matching_ms = (time.perf_counter() - recovery_start) * 1000.0
                    if (
                        recovery_sample.status == "valid"
                        and recovery_sample.disparity is not None
                        and recovery_sample.right_xy is not None
                    ):
                        recovered_result = self._valid_result(
                                frame,
                                state,
                                recovery_sample.right_xy,
                                recovery_sample.disparity,
                                None,
                                recovery_sample.lr_error_px,
                                1.0,
                                flow_ms,
                                recovery_matching_ms,
                                (time.perf_counter() - point_start) * 1000.0,
                                flow_error,
                                recovery_stage="sgbm_reinitialized",
                            )
                        if recovered_result.status == "valid":
                            recovered_result = self._finalize_right_flow_reference(
                                recovered_result,
                                frame,
                                right_gray,
                            )
                        results.append(recovered_result)
                        continue
                state.record_failure("ambiguous", self.config.max_failures)
                results.append(
                    FramePointResult.invalid(
                        self.method,
                        frame,
                        state,
                        "lost" if state.status == "lost" else "ambiguous",
                        flow_ms=flow_ms,
                        total_ms=(time.perf_counter() - point_start) * 1000.0,
                        flow_fb_error_px=flow_error,
                        recovery_stage=recovery_stage,
                    )
                )
                continue
            predicted = (
                predict_disparity(
                    latest,
                    prediction_history[-2] if len(prediction_history) >= 2 else None,
                    self.config.max_disparity_velocity,
                )
                if self.profile.use_prediction and self.config.enable_prediction
                else latest
            )
            if temporal_right_xy is not None:
                cycle_predicted = state.left_xy[0] - temporal_right_xy[0]
                blend = self.config.cycle_prediction_blend
                predicted = float((1.0 - blend) * predicted + blend * cycle_predicted)
            matching_start = time.perf_counter()
            matcher = self.recovery_matcher if recovery_attempt else self.local_matcher
            match = matcher.match(
                left_gray,
                right_gray,
                state.left_xy,
                predicted,
                self.profile,
                latest_disparity=latest,
                temporal_right_xy=temporal_right_xy,
                right_flow_fb_error_px=right_flow_error,
            )
            matching_ms = (time.perf_counter() - matching_start) * 1000.0
            if match.status != "valid" or match.disparity is None or match.right_xy is None:
                state.record_failure(match.status, self.config.max_failures)
                results.append(
                    FramePointResult.invalid(
                        self.method,
                        frame,
                        state,
                        "lost" if state.status == "lost" else match.status,
                        flow_ms=flow_ms,
                        matching_ms=matching_ms,
                        total_ms=(time.perf_counter() - point_start) * 1000.0,
                        flow_fb_error_px=flow_error,
                        recovery_stage=recovery_stage,
                    )
                )
                continue
            if temporal_right_xy is not None:
                cycle_check = evaluate_cycle_consistency(
                    match.right_xy,
                    temporal_right_xy,
                    self.config,
                )
                if cycle_check.should_recover:
                    recovery_match = self.recovery_matcher.match(
                        left_gray,
                        right_gray,
                        state.left_xy,
                        state.left_xy[0] - temporal_right_xy[0],
                        self.profile,
                        latest_disparity=latest,
                        temporal_right_xy=temporal_right_xy,
                        right_flow_fb_error_px=right_flow_error,
                    )
                    matching_ms = (time.perf_counter() - matching_start) * 1000.0
                    if recovery_match.status == "valid" and recovery_match.right_xy is not None:
                        cycle_check = evaluate_cycle_consistency(
                            match.right_xy,
                            temporal_right_xy,
                            self.config,
                            recovered_right_xy=recovery_match.right_xy,
                        )
                        if cycle_check.status == "cycle_recovered":
                            match = replace(
                                recovery_match,
                                cycle_error_px=cycle_check.error_px,
                                cycle_cost=cycle_check.cost,
                                quality_stage="cycle_recovered",
                            )
                cycle_status = cycle_check.status
                if cycle_status == "cycle_failed":
                    state.record_failure("cycle_failed", self.config.max_failures)
                    results.append(
                        FramePointResult.invalid(
                            self.method,
                            frame,
                            state,
                            "lost" if state.status == "lost" else "cycle_failed",
                            flow_ms=flow_ms,
                            matching_ms=matching_ms,
                            total_ms=(time.perf_counter() - point_start) * 1000.0,
                            flow_fb_error_px=flow_error,
                            recovery_stage=recovery_stage,
                            temporal_right_xy=temporal_right_xy,
                            right_flow_fb_error_px=right_flow_error,
                            cycle_error_px=cycle_check.error_px,
                            cycle_cost=cycle_check.cost,
                            cycle_status="cycle_failed",
                        )
                    )
                    continue
                match = replace(
                    match,
                    confidence=match.confidence * cycle_check.confidence_scale,
                    cycle_error_px=cycle_check.error_px,
                    cycle_cost=cycle_check.cost,
                )
            point_result = self._valid_result(
                    frame,
                    state,
                    match.right_xy,
                    match.disparity,
                    match.cost,
                    match.lr_error_px,
                    match.confidence,
                    flow_ms,
                    matching_ms,
                    (time.perf_counter() - point_start) * 1000.0,
                    flow_error,
                    raw_disparity=match.raw_disparity,
                    measured_disparity=match.measured_disparity,
                    estimated_disparity=match.estimated_disparity,
                    estimated_right_xy=match.estimated_right_xy,
                    integer_disparity=match.integer_disparity,
                    subpixel_offset=match.subpixel_offset,
                    neighbor_disparity=match.neighbor_disparity,
                    used_search_radius=match.used_search_radius,
                    quality_stage=match.quality_stage,
                    recovery_stage="recaptured" if recovery_attempt else recovery_stage,
                    temporal_right_xy=temporal_right_xy,
                    right_flow_fb_error_px=right_flow_error,
                    cycle_error_px=match.cycle_error_px,
                    cycle_cost=match.cycle_cost,
                    cycle_status=cycle_status,
                    texture_std=match.texture_std,
                    second_best_cost=match.second_best_cost,
                    uniqueness_margin_value=match.uniqueness_margin_value,
                    cost_curvature=match.cost_curvature,
                    icgn_status=match.icgn_status,
                    icgn_converged=match.icgn_converged,
                    icgn_iterations=match.icgn_iterations,
                    icgn_residual=match.icgn_residual,
                    icgn_hessian=match.icgn_hessian,
                    icgn_cost_curvature=match.icgn_cost_curvature,
                    icgn_hessian_density=match.icgn_hessian_density,
                    zncc_cost_curvature=match.zncc_cost_curvature,
                    curvature_sample_step_px=match.curvature_sample_step_px,
                    icgn_termination_reason=match.icgn_termination_reason,
                    icgn_fallback_used=match.icgn_fallback_used,
                    icgn_fallback_method=match.icgn_fallback_method,
                    icgn_iterative_disparity=match.icgn_iterative_disparity,
                    icgn_final_increment_px=match.icgn_final_increment_px,
                    subpixel_final_status=match.subpixel_final_status,
                )
            if point_result.status == "valid":
                point_result = self._finalize_right_flow_reference(
                    point_result,
                    frame,
                    right_gray,
                )
            results.append(point_result)
            if point_result.status == "valid" and temporal_right_xy is not None:
                state.last_cycle_error_px = match.cycle_error_px
                state.last_right_flow_fb_error_px = right_flow_error
        self.previous_left_gray = left_gray.copy()
        self.previous_right_gray = right_gray.copy()
        return self._apply_camera_compensation(results)

    def _update_right_flow_reference(
        self,
        point_id: str,
        frame: int,
        right_gray: np.ndarray,
        right_xy: tuple[float, float],
    ) -> None:
        """Atomically update all right temporal-reference components."""
        if point_id not in self.states:
            raise KeyError(f"Unknown point id: {point_id}")
        self.right_flow_reference_gray[point_id] = right_gray.copy()
        self.states[point_id].set_right_flow_reference(frame, right_xy)

    def _right_flow_reference(
        self,
        point_id: str,
        current_frame: int,
    ) -> tuple[np.ndarray | None, tuple[float, float] | None, str]:
        state = self.states[point_id]
        image = self.right_flow_reference_gray.get(point_id)
        xy = state.right_flow_reference_xy
        reference_frame = state.right_flow_reference_frame
        if image is None or xy is None or reference_frame is None:
            return None, None, "not_attempted"
        if reference_frame >= current_frame:
            return None, None, "invalid_right_reference"
        return image, xy, "valid"

    def _finalize_right_flow_reference(
        self,
        result: FramePointResult,
        frame: int,
        right_gray: np.ndarray,
    ) -> FramePointResult:
        if not (
            self.profile.use_cycle_consistency
            and self.config.enable_cycle_consistency
            and result.status == "valid"
        ):
            return result
        if result.measurement_accepted_for_state:
            xy = (result.measured_right_x, result.measured_right_y)
            source = "measured"
        else:
            xy = (result.estimated_right_x, result.estimated_right_y)
            source = "estimated"
        if any(value is None or not np.isfinite(value) for value in xy):
            reference_frame = self.states[result.point_id].right_flow_reference_frame
            return replace(
                result,
                right_flow_reference_frame=reference_frame,
                right_reference_source="retained",
                right_reference_age_frames=(
                    frame - reference_frame if reference_frame is not None else None
                ),
            )
        right_xy = (float(xy[0]), float(xy[1]))
        height, width = right_gray.shape[:2]
        if not (0.0 <= right_xy[0] < width and 0.0 <= right_xy[1] < height):
            reference_frame = self.states[result.point_id].right_flow_reference_frame
            return replace(
                result,
                right_flow_reference_frame=reference_frame,
                right_reference_source="retained",
                right_reference_age_frames=(
                    frame - reference_frame if reference_frame is not None else None
                ),
            )
        self._update_right_flow_reference(result.point_id, frame, right_gray, right_xy)
        return replace(
            result,
            right_flow_reference_frame=frame,
            right_reference_source=source,
            right_reference_age_frames=0,
        )

    def _apply_camera_compensation(
        self,
        results: list[FramePointResult],
    ) -> list[FramePointResult]:
        if not (
            self.profile.use_camera_compensation
            and self.config.enable_camera_compensation
        ):
            return results
        reference_results: list[FramePointResult] = []
        current_points: list[tuple[float, float, float]] = []
        initial_points: list[tuple[float, float, float]] = []
        weights: list[float] = []
        for result in results:
            state = self.states[result.point_id]
            current = (result.estimated_x_m, result.estimated_y_m, result.estimated_z_m)
            if (
                result.status != "valid"
                or state.point_role != "reference"
                or state.reference_estimated_xyz is None
                or any(value is None for value in current)
            ):
                continue
            current_array = np.asarray(current, dtype=np.float64)
            if not np.isfinite(current_array).all():
                continue
            reference_results.append(result)
            current_points.append(tuple(float(value) for value in current_array))
            initial_points.append(state.reference_estimated_xyz)
            variance = state.last_disparity_variance_px2 or 1.0
            weights.append(max(state.confidence, 1e-3) / max(variance, 1e-9))
        if current_points:
            compensation = estimate_camera_compensation(
                np.asarray(current_points, dtype=np.float64),
                np.asarray(initial_points, dtype=np.float64),
                tuple(result.point_id for result in reference_results),
                np.asarray(weights, dtype=np.float64),
                self.config,
            )
        else:
            compensation = estimate_camera_compensation(
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                (),
                None,
                self.config,
            )
        reference_count = len(reference_results)
        applied = (
            compensation.status == "valid"
            and compensation.rotation is not None
            and compensation.translation_m is not None
        )
        if applied:
            assert compensation.rotation is not None
            assert compensation.translation_m is not None
            rotation_angle = float(
                np.degrees(
                    np.arccos(
                        np.clip((np.trace(compensation.rotation) - 1.0) * 0.5, -1.0, 1.0)
                    )
                )
            )
            translation_mm = compensation.translation_m * 1000.0
        else:
            rotation_angle = None
            translation_mm = np.asarray([None, None, None], dtype=object)

        processed: list[FramePointResult] = []
        for result in results:
            state = self.states[result.point_id]
            estimated_values = (result.estimated_x_m, result.estimated_y_m, result.estimated_z_m)
            if result.status != "valid" or any(value is None for value in estimated_values):
                processed.append(
                    replace(
                        result,
                        compensation_status=compensation.status,
                        compensation_applied=False,
                        compensation_inlier_count=len(compensation.inlier_ids),
                        compensation_reference_count=reference_count,
                        compensation_rmse_mm=compensation.residual_rmse_mm,
                    )
                )
                continue
            estimated_xyz = np.asarray(estimated_values, dtype=np.float64)
            reference_xyz = np.asarray(
                state.reference_compensated_xyz
                or state.reference_estimated_xyz
                or tuple(estimated_xyz),
                dtype=np.float64,
            )
            raw_delta = (estimated_xyz - reference_xyz) * 1000.0
            final_xyz = estimated_xyz
            compensated_xyz: np.ndarray | None = None
            if applied:
                compensated_xyz = apply_rigid_transform(
                    estimated_xyz.reshape(1, 3),
                    compensation.rotation,
                    compensation.translation_m,
                )[0]
                final_xyz = compensated_xyz
            state.compensated_xyz = tuple(float(value) for value in final_xyz)
            if state.reference_compensated_xyz is None:
                state.reference_compensated_xyz = tuple(
                    state.reference_estimated_xyz
                    or tuple(float(value) for value in final_xyz)
                )
                reference_xyz = np.asarray(state.reference_compensated_xyz, dtype=np.float64)
            final_delta = (final_xyz - reference_xyz) * 1000.0
            processed.append(
                replace(
                    result,
                    compensation_status=compensation.status,
                    compensation_applied=applied,
                    compensation_inlier_count=len(compensation.inlier_ids),
                    compensation_reference_count=reference_count,
                    compensation_rmse_mm=compensation.residual_rmse_mm,
                    camera_tx_mm=translation_mm[0],
                    camera_ty_mm=translation_mm[1],
                    camera_tz_mm=translation_mm[2],
                    camera_rotation_angle_deg=rotation_angle,
                    compensated_x_m=(float(compensated_xyz[0]) if compensated_xyz is not None else None),
                    compensated_y_m=(float(compensated_xyz[1]) if compensated_xyz is not None else None),
                    compensated_z_m=(float(compensated_xyz[2]) if compensated_xyz is not None else None),
                    final_x_m=float(final_xyz[0]),
                    final_y_m=float(final_xyz[1]),
                    final_z_m=float(final_xyz[2]),
                    raw_delta_x_mm=float(raw_delta[0]),
                    raw_delta_y_mm=float(raw_delta[1]),
                    raw_delta_z_mm=float(raw_delta[2]),
                    compensated_delta_x_mm=(float(final_delta[0]) if applied else None),
                    compensated_delta_y_mm=(float(final_delta[1]) if applied else None),
                    compensated_delta_z_mm=(float(final_delta[2]) if applied else None),
                    delta_x_mm=float(final_delta[0]),
                    delta_y_mm=float(final_delta[1]),
                    delta_z_mm=float(final_delta[2]),
                )
            )
        return processed

    def _valid_result(
        self,
        frame: int,
        state: PointState,
        right_xy: tuple[float, float],
        disparity: float,
        match_cost: float | None,
        lr_error: float | None,
        confidence: float,
        flow_ms: float,
        matching_ms: float,
        total_ms: float,
        flow_error: float | None = None,
        raw_disparity: float | None = None,
        measured_disparity: float | None = None,
        estimated_disparity: float | None = None,
        estimated_right_xy: tuple[float, float] | None = None,
        integer_disparity: float | None = None,
        subpixel_offset: float | None = None,
        neighbor_disparity: float | None = None,
        used_search_radius: int = 0,
        quality_stage: str = "",
        recovery_stage: str = "",
        temporal_right_xy: tuple[float, float] | None = None,
        right_flow_fb_error_px: float | None = None,
        cycle_error_px: float | None = None,
        cycle_cost: float | None = None,
        cycle_status: str = "not_used",
        texture_std: float | None = None,
        second_best_cost: float | None = None,
        uniqueness_margin_value: float | None = None,
        cost_curvature: float | None = None,
        icgn_status: str = "not_attempted",
        icgn_converged: bool = False,
        icgn_iterations: int = 0,
        icgn_residual: float | None = None,
        icgn_hessian: float | None = None,
        icgn_cost_curvature: float | None = None,
        icgn_hessian_density: float | None = None,
        zncc_cost_curvature: float | None = None,
        curvature_sample_step_px: float | None = None,
        icgn_termination_reason: str = "",
        icgn_fallback_used: bool = False,
        icgn_fallback_method: str = "none",
        icgn_iterative_disparity: float | None = None,
        icgn_final_increment_px: float | None = None,
        subpixel_final_status: str = "not_attempted",
    ) -> FramePointResult:
        measured = float(
            measured_disparity
            if measured_disparity is not None
            else (raw_disparity if raw_disparity is not None else disparity)
        )
        try:
            measured_xyz_array = reproject_point_m(
                state.left_xy[0],
                state.left_xy[1],
                measured,
                self.q,
                self.calibration_unit,
            )
        except ValueError:
            state.record_failure("ambiguous", self.config.max_failures)
            return FramePointResult.invalid(
                self.method,
                frame,
                state,
                "lost" if state.status == "lost" else "ambiguous",
                flow_ms=flow_ms,
                matching_ms=matching_ms,
                total_ms=total_ms,
                flow_fb_error_px=flow_error,
            )
        if not self.config.min_depth_m <= float(measured_xyz_array[2]) <= self.config.max_depth_m:
            state.record_failure("depth_out_of_range", self.config.max_failures)
            return FramePointResult.invalid(
                self.method,
                frame,
                state,
                "lost" if state.status == "lost" else "depth_out_of_range",
                flow_ms=flow_ms,
                matching_ms=matching_ms,
                total_ms=total_ms,
                flow_fb_error_px=flow_error,
            )
        if (
            self.config.max_motion_m_per_frame is not None
            and state.xyz is not None
            and float(
                np.linalg.norm(measured_xyz_array - np.asarray(state.xyz, dtype=np.float64))
            )
            > self.config.max_motion_m_per_frame
        ):
            state.record_failure("motion_rejected", self.config.max_failures)
            return FramePointResult.invalid(
                self.method,
                frame,
                state,
                "lost" if state.status == "lost" else "motion_rejected",
                flow_ms=flow_ms,
                matching_ms=matching_ms,
                total_ms=total_ms,
                flow_fb_error_px=flow_error,
            )

        estimated = measured
        estimated_left_xy = state.left_xy
        left_variance_px2: float | None = None
        disparity_variance_px2: float | None = None
        measurement_quality_score: float | None = None
        kalman_innovation = (None, None, None)
        kalman_innovation_norm: float | None = None
        kalman_gain_disparity: float | None = None
        kalman_nis: float | None = None
        kalman_update_status = "not_used"
        kalman_predict_only_frames = 0
        measurement_accepted_for_state = True
        state_update_source = "measured"
        kalman_recovery_needed = False
        latest_valid = state.disparity
        if (
            self.profile.use_adaptive_filter
            and self.config.enable_adaptive_filter
        ):
            uncertainty = estimate_match_uncertainty(
                texture_std=texture_std,
                photo_cost=match_cost,
                uniqueness_margin=uniqueness_margin_value,
                cost_curvature=cost_curvature,
                flow_fb_error_px=flow_error,
                right_flow_fb_error_px=right_flow_fb_error_px,
                lr_error_px=lr_error,
                cycle_error_px=cycle_error_px,
                icgn_residual=icgn_residual,
                icgn_hessian=icgn_hessian,
                config=self.config,
                icgn_hessian_density=icgn_hessian_density,
            )
            left_variance_px2 = uncertainty.left_position_variance_px2
            disparity_variance_px2 = uncertainty.disparity_variance_px2
            measurement_quality_score = uncertainty.quality_score
            point_filter = self.filters.get(state.point_id)
            if point_filter is None:
                point_filter = AdaptivePointKalman(self.config)
                point_filter.initialize(state.left_xy, measured)
                self.filters[state.point_id] = point_filter
                kalman_update_status = "initialized"
            else:
                point_filter.predict()
                update = point_filter.update(
                    state.left_xy,
                    measured,
                    left_variance_px2,
                    disparity_variance_px2,
                )
                estimated_left_xy = update.estimated_left_xy
                estimated = update.estimated_disparity
                kalman_innovation = update.innovation
                kalman_innovation_norm = update.innovation_norm
                kalman_gain_disparity = update.kalman_gain_disparity
                kalman_nis = update.nis
                kalman_update_status = update.status
                kalman_predict_only_frames = point_filter.predict_only_frames
                measurement_accepted_for_state = update.status in {
                    "updated",
                    "variance_inflated",
                }
                state_update_source = (
                    "filtered" if measurement_accepted_for_state else "predicted"
                )
                kalman_recovery_needed = (
                    update.status == "predict_only_outlier"
                    and point_filter.predict_only_frames
                    >= self.config.kalman_max_predict_only_frames
                )
                state.last_kalman_innovation = update.innovation_norm
                state.last_kalman_gain_disparity = update.kalman_gain_disparity
                state.last_nis = update.nis
            state.last_left_variance_px2 = left_variance_px2
            state.last_disparity_variance_px2 = disparity_variance_px2
        elif (
            self.method == "full_quality"
            and self.config.enable_temporal_estimation
            and latest_valid is not None
            and measured < self.config.quality_far_disparity_px
            and abs(measured - latest_valid)
            <= self.config.quality_max_smoothing_innovation_px
        ):
            estimated = float(
                latest_valid
                + self.config.quality_disparity_alpha * (measured - latest_valid)
            )
            quality_stage = (
                "context_temporal_stabilized"
                if quality_stage == "context_recovery"
                else "temporal_stabilized"
            )
        estimated_right = (
            float(estimated_left_xy[0] - estimated),
            float(estimated_left_xy[1] + (right_xy[1] - state.left_xy[1])),
        )
        try:
            estimated_xyz_array = reproject_point_m(
                estimated_left_xy[0],
                estimated_left_xy[1],
                estimated,
                self.q,
                self.calibration_unit,
            )
        except ValueError:
            state.record_failure("ambiguous", self.config.max_failures)
            return FramePointResult.invalid(
                self.method,
                frame,
                state,
                "lost" if state.status == "lost" else "ambiguous",
                flow_ms=flow_ms,
                matching_ms=matching_ms,
                total_ms=total_ms,
                flow_fb_error_px=flow_error,
            )

        measured_xyz = tuple(float(value) for value in measured_xyz_array)
        estimated_xyz = tuple(float(value) for value in estimated_xyz_array)
        was_recovering = (
            self.method in {"full_quality", "research_full"}
            and self.config.enable_recovery
            and state.status in {"recovering", "lost"}
        )
        if measurement_accepted_for_state:
            state.record_stereo_measurement(
                right_xy,
                measured,
                confidence,
                estimated_disparity=estimated,
                xyz=measured_xyz,
                estimated_xyz=estimated_xyz,
            )
            if state_update_source == "filtered":
                state.left_xy = estimated_left_xy
                state.right_xy = estimated_right
                state.latest_disparity = estimated
                state.last_estimated_left_xy = estimated_left_xy
                state.last_estimated_right_xy = estimated_right
                state.last_estimated_disparity = estimated
        else:
            quality_stage = (
                f"{quality_stage}_kalman_predict_only"
                if quality_stage
                else "kalman_predict_only"
            )
            state.record_estimated_only(
                estimated_left_xy=estimated_left_xy,
                estimated_right_xy=estimated_right,
                estimated_disparity=estimated,
                estimated_xyz_m=estimated_xyz,
                frame=frame,
                quality_stage=quality_stage,
            )
        state.confirm_valid(confidence)
        if kalman_recovery_needed:
            state.status = "recovering"
            if state.recovery_started_frame is None:
                state.recovery_started_frame = frame
        if was_recovering and measurement_accepted_for_state:
            state.confirm_recovery(frame)
        reference = state.reference_estimated_xyz or estimated_xyz
        delta_mm = tuple(
            (estimated_xyz[index] - reference[index]) * 1000.0
            for index in range(3)
        )
        return FramePointResult(
            method=self.method,
            frame=frame,
            point_id=state.point_id,
            status="valid",
            left_x=state.left_xy[0],
            left_y=state.left_xy[1],
            right_x=right_xy[0],
            right_y=right_xy[1],
            disparity=estimated,
            x_m=estimated_xyz[0],
            y_m=estimated_xyz[1],
            z_m=estimated_xyz[2],
            distance_m=float(np.linalg.norm(estimated_xyz_array)),
            match_cost=match_cost,
            flow_fb_error_px=flow_error,
            lr_error_px=lr_error,
            confidence=confidence,
            raw_disparity=measured,
            integer_disparity=integer_disparity,
            measured_disparity=measured,
            estimated_disparity=estimated,
            measured_right_x=right_xy[0],
            measured_right_y=right_xy[1],
            estimated_right_x=estimated_right[0],
            estimated_right_y=estimated_right[1],
            measured_x_m=measured_xyz[0],
            measured_y_m=measured_xyz[1],
            measured_z_m=measured_xyz[2],
            estimated_x_m=estimated_xyz[0],
            estimated_y_m=estimated_xyz[1],
            estimated_z_m=estimated_xyz[2],
            delta_x_mm=delta_mm[0],
            delta_y_mm=delta_mm[1],
            delta_z_mm=delta_mm[2],
            subpixel_offset=subpixel_offset,
            neighbor_disparity=neighbor_disparity,
            used_search_radius=used_search_radius,
            quality_stage=quality_stage,
            recovery_stage=recovery_stage,
            recovery_attempt_count=state.recovery_attempt_count,
            recovery_success_count=state.recovery_success_count,
            mean_recovery_frames=(
                state.recovery_frame_total / state.recovery_success_count
                if state.recovery_success_count
                else None
            ),
            flow_ms=flow_ms,
            matching_ms=matching_ms,
            total_ms=total_ms,
            temporal_right_x=(temporal_right_xy[0] if temporal_right_xy else None),
            temporal_right_y=(temporal_right_xy[1] if temporal_right_xy else None),
            right_flow_fb_error_px=right_flow_fb_error_px,
            cycle_error_px=cycle_error_px,
            cycle_cost=cycle_cost,
            cycle_status=cycle_status,
            texture_std=texture_std,
            second_best_cost=second_best_cost,
            uniqueness_margin_value=uniqueness_margin_value,
            cost_curvature=cost_curvature,
            icgn_status=icgn_status,
            icgn_converged=icgn_converged,
            icgn_iterations=icgn_iterations,
            icgn_residual=icgn_residual,
            icgn_hessian=icgn_hessian,
            icgn_cost_curvature=icgn_cost_curvature,
            icgn_hessian_density=icgn_hessian_density,
            zncc_cost_curvature=zncc_cost_curvature,
            curvature_sample_step_px=curvature_sample_step_px,
            icgn_termination_reason=icgn_termination_reason,
            icgn_fallback_used=icgn_fallback_used,
            icgn_fallback_method=icgn_fallback_method,
            icgn_iterative_disparity=icgn_iterative_disparity,
            icgn_final_increment_px=icgn_final_increment_px,
            subpixel_final_status=subpixel_final_status,
            point_role=state.point_role,
            final_x_m=estimated_xyz[0],
            final_y_m=estimated_xyz[1],
            final_z_m=estimated_xyz[2],
            raw_delta_x_mm=delta_mm[0],
            raw_delta_y_mm=delta_mm[1],
            raw_delta_z_mm=delta_mm[2],
            left_variance_px2=left_variance_px2,
            disparity_variance_px2=disparity_variance_px2,
            measurement_quality_score=measurement_quality_score,
            kalman_innovation_u=kalman_innovation[0],
            kalman_innovation_v=kalman_innovation[1],
            kalman_innovation_d=kalman_innovation[2],
            kalman_innovation_norm=kalman_innovation_norm,
            kalman_gain_disparity=kalman_gain_disparity,
            kalman_nis=kalman_nis,
            kalman_update_status=kalman_update_status,
            kalman_predict_only_frames=kalman_predict_only_frames,
            measurement_accepted_for_state=measurement_accepted_for_state,
            state_update_source=state_update_source,
        )

    @staticmethod
    def _gray_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_gray = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = right if right.ndim == 2 else cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        if left_gray.shape != right_gray.shape:
            raise ValueError("Left and right images must have identical shapes")
        return left_gray.astype(np.uint8, copy=False), right_gray.astype(np.uint8, copy=False)

    def _initial_patch_status(
        self,
        left_gray: np.ndarray,
        left_xy: tuple[float, float],
    ) -> str:
        radius = self.config.patch_size // 2
        x, y = left_xy
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius >= left_gray.shape[1]
            or y + radius >= left_gray.shape[0]
        ):
            return "out_of_bounds"
        patch = cv2.getRectSubPix(
            left_gray,
            (self.config.patch_size, self.config.patch_size),
            left_xy,
        )
        return "low_texture" if float(np.std(patch)) < self.config.min_texture_std else "valid"
