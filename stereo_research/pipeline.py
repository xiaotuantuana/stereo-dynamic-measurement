from __future__ import annotations

import time
from dataclasses import replace
from typing import Iterable

import cv2
import numpy as np

from stereo_dynamic_measurement.innovation1.precision_planner import (
    TargetAccuracySpec,
    build_precision_plan,
)

from .camera_compensation import CameraCompensationResult, apply_rigid_transform, estimate_camera_compensation, rotation_matrix_to_euler_xyz_deg
from .accuracy_policy import MeasurementPolicyDecision, decide_measurement_policy
from .confidence import ConfidenceDecision, decide_confidence
from .cycle_consistency import evaluate_cycle_consistency
from .filtering import AdaptivePointKalman
from .enhanced_processor import EnhancedPointProcessor, I2Evidence, TimedObservation
from .final_arbitration import (
    ArbitrationDecision,
    ExperimentAuthority,
    FinalArbitrator,
    FinalDecision,
    I1BaselineView,
    I3Action,
    I3Recommendation,
    I3Risk,
    ResultSource,
)
from .geometry import reproject_point_m
from .global_matching import GlobalSample, GlobalStereoMatcher, GlobalStereoResult
from .local_matching import LocalMatchResult, LocalMatcher, QualityLocalMatcher, predict_disparity
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
from .shadow_analysis import ShadowAnalyzer


def _confidence_component(risk: float | None, config: MatcherConfig) -> float | None:
    return None if risk is None else float(np.clip(1.0 - risk / config.uncertainty_max_component, 0.0, 1.0))


class TemporalStereoPipeline:
    def __init__(
        self,
        method: MethodName,
        q: np.ndarray,
        calibration_unit: str,
        config: MatcherConfig | None = None,
        experiment_authority: ExperimentAuthority | None = None,
    ):
        self.method = method
        self.profile = method_profile(method)
        self.q = np.asarray(q, dtype=np.float64)
        self.calibration_unit = calibration_unit
        self.config = config or MatcherConfig()
        # No production or GUI caller receives this opt-in authority by default.
        self.experiment_authority = experiment_authority
        self.enhanced_processors: dict[str, EnhancedPointProcessor] = {}
        self.final_arbitrator = FinalArbitrator()
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
        self.shadow_analyzer = (
            ShadowAnalyzer()
            if self.config.enable_physics_shadow or self.config.enable_fault_shadow
            else None
        )
        self.target_accuracy_spec = (
            TargetAccuracySpec(
                metric=self.config.target_metric,
                target_sigma_x_mm=self.config.target_sigma_x_mm,
                target_sigma_y_mm=self.config.target_sigma_y_mm,
                target_sigma_z_mm=self.config.target_sigma_z_mm,
            )
            if self.config.enable_target_accuracy_policy
            else None
        )
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
            initial_diagnostics = None
            if self.config.enable_initial_confidence_calibration:
                initial_diagnostics = self.local_matcher.diagnose_initial(
                    left_gray,
                    right_gray,
                    point.xy,
                    sample.disparity,
                    max_disparity=self.config.expanded_num_disparities - 1,
                    lr_error_px=sample.lr_error_px,
                )
                initial_rejected = (
                    initial_diagnostics.status != "valid"
                    or initial_diagnostics.uniqueness_margin is None
                    or initial_diagnostics.uniqueness_margin < self.config.uniqueness_margin
                    or initial_diagnostics.confidence < self.config.confidence_medium_threshold
                )
                if initial_rejected:
                    state.record_failure("initial_ambiguous", self.config.max_failures)
                    rejected = FramePointResult.invalid(
                        method=self.method,
                        frame=frame,
                        point_state=state,
                        status="initial_ambiguous",
                        matching_ms=initialization_matching_ms,
                        total_ms=initialization_matching_ms,
                    )
                    results.append(replace(
                        rejected,
                        measured_right_x=sample.right_xy[0],
                        measured_right_y=sample.right_xy[1],
                        measured_disparity=sample.disparity,
                        raw_disparity=sample.disparity,
                        integer_disparity=float(round(sample.disparity)),
                        subpixel_offset=sample.disparity - round(sample.disparity),
                        match_cost=initial_diagnostics.predicted_cost,
                        lr_error_px=sample.lr_error_px,
                        confidence=initial_diagnostics.confidence,
                        confidence_total=initial_diagnostics.confidence,
                        confidence_state="LOW",
                        confidence_source="initial_runtime_evidence",
                        measurement_accepted_for_state=False,
                        state_update_source="rejected",
                        texture_std=initial_diagnostics.texture_std,
                        second_best_cost=initial_diagnostics.second_best_cost,
                        uniqueness_margin_value=initial_diagnostics.uniqueness_margin,
                        candidate_count=initial_diagnostics.candidate_count,
                        initial_best_disparity=initial_diagnostics.best_disparity,
                        initial_disparity_disagreement=initial_diagnostics.predicted_best_disagreement,
                    ))
                    continue
            result = self._valid_result(
                    frame,
                    state,
                    sample.right_xy,
                    sample.disparity,
                    match_cost=(initial_diagnostics.predicted_cost if initial_diagnostics else None),
                    lr_error=sample.lr_error_px,
                    confidence=(initial_diagnostics.confidence if initial_diagnostics else 1.0),
                    flow_ms=0.0,
                    matching_ms=initialization_matching_ms,
                    total_ms=initialization_matching_ms,
                    raw_disparity=(sample.disparity if initial_diagnostics else None),
                    integer_disparity=(float(round(sample.disparity)) if initial_diagnostics else None),
                    subpixel_offset=(sample.disparity - round(sample.disparity) if initial_diagnostics else None),
                    used_search_radius=(self.config.expanded_num_disparities - 1 if initial_diagnostics else 0),
                    texture_std=(initial_diagnostics.texture_std if initial_diagnostics else None),
                    second_best_cost=(initial_diagnostics.second_best_cost if initial_diagnostics else None),
                    uniqueness_margin_value=(initial_diagnostics.uniqueness_margin if initial_diagnostics else None),
                )
            if initial_diagnostics is not None:
                result = replace(
                    result,
                    confidence_source="initial_runtime_evidence",
                    candidate_count=initial_diagnostics.candidate_count,
                    initial_best_disparity=initial_diagnostics.best_disparity,
                    initial_disparity_disagreement=initial_diagnostics.predicted_best_disagreement,
                )
            if result.status == "valid":
                result = self._annotate_accuracy_policy(
                    result,
                    base_search_radius_px=self.config.search_radius,
                    precision_retry_count=0,
                    warmup=True,
                )
                result = self._finalize_right_flow_reference(result, frame, right_gray)
            results.append(result)
        self.previous_left_gray = left_gray.copy()
        self.previous_right_gray = right_gray.copy()
        self.initialized = True
        return self._finalize_frame_results(results)

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
                    (self.profile.adaptive_search or self.profile.use_confidence_feedback or self.method == "full_quality")
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
                recovery_stage = "local_recovery"
                state.recovery_attempt_count += 1
                if state.recovery_started_frame is None:
                    state.recovery_started_frame = frame
            elif (
                state.status == "recovering"
                and (self.profile.adaptive_search or self.profile.use_confidence_feedback or self.method == "full_quality")
                and self.config.enable_recovery
            ):
                recovery_attempt = True
                recovery_stage = "local_recovery"
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
                if self.profile.use_confidence_feedback and state.estimated_disparity_history
                else state.disparity_history
            )
            latest = prediction_history[-1] if prediction_history else None
            # Only C3/THESIS_FULL turns LOST into a confidence-driven global
            # reinitialization. Legacy quality recovery keeps its prior local path.
            must_reinitialize = (
                recovery_attempt and state.status == "lost"
                and self.profile.use_confidence_feedback
            )
            if latest is None or must_reinitialize:
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
                        # A global correspondence is a new track origin: do not let
                        # old prediction or Kalman state bias the recovered trajectory.
                        state.disparity_history.clear()
                        state.estimated_disparity_history.clear()
                        state.latest_disparity = None
                        self.filters.pop(state.point_id, None)
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
                                recovery_stage=(
                                    "global_reinitialization"
                                    if must_reinitialize else "sgbm_reinitialized"
                                ),
                            )
                        if recovered_result.status == "valid" and recovered_result.measurement_accepted_for_state:
                            if must_reinitialize:
                                recovered_result = replace(
                                    recovered_result,
                                    recovery_stage="reinitialization_success",
                                )
                            recovered_result = self._finalize_right_flow_reference(
                                recovered_result,
                                frame,
                                right_gray,
                            )
                        elif must_reinitialize:
                            recovered_result = replace(
                                recovered_result,
                                recovery_stage="reinitialization_failed",
                            )
                        results.append(recovered_result)
                        continue
                    if must_reinitialize:
                        state.record_failure("reinitialization_failed", self.config.max_failures)
                        results.append(
                            FramePointResult.invalid(
                                self.method, frame, state, "lost", flow_ms=flow_ms,
                                matching_ms=recovery_matching_ms,
                                total_ms=(time.perf_counter() - point_start) * 1000.0,
                                flow_fb_error_px=flow_error,
                                recovery_stage="reinitialization_failed",
                            )
                        )
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
            previous_disparity = prediction_history[-2] if len(prediction_history) >= 2 else None
            disparity_velocity = (
                None if previous_disparity is None else float(latest - previous_disparity)
            )
            if temporal_right_xy is not None:
                cycle_predicted = state.left_xy[0] - temporal_right_xy[0]
                blend = self.config.cycle_prediction_blend
                predicted = float((1.0 - blend) * predicted + blend * cycle_predicted)
            matching_start = time.perf_counter()
            matcher = self.recovery_matcher if recovery_attempt else self.local_matcher
            adaptive_radius, radius_reason = self._determine_search_radius(
                state, disparity_velocity, recovery_attempt
            )
            if self.profile.use_confidence_feedback and self.config.confidence_mode == "closed_loop":
                overlay = {
                    "HIGH": self.config.adaptive_search_small_radius,
                    "MEDIUM": self.config.adaptive_search_normal_radius,
                    "LOW": self.config.adaptive_search_large_radius,
                    "LOST": self.config.adaptive_search_recovery_radius,
                }[state.confidence_state]
                if overlay > adaptive_radius:
                    adaptive_radius, radius_reason = overlay, f"confidence_{state.confidence_state.lower()}"
            pre_match_policy = self._pre_match_accuracy_policy(state, adaptive_radius)
            refinement_level = (
                pre_match_policy.refinement_level if pre_match_policy is not None else 0
            )
            match = matcher.match(
                left_gray,
                right_gray,
                state.left_xy,
                predicted,
                self.profile,
                latest_disparity=latest,
                temporal_right_xy=temporal_right_xy,
                right_flow_fb_error_px=right_flow_error,
                search_radius=adaptive_radius,
                refinement_level=refinement_level,
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
            precision_retry_count = 0
            retry_triggered = False
            retry_candidate_accepted = False
            retry_sigma_before: float | None = None
            retry_sigma_after: float | None = None
            retry_match_cost_before: float | None = None
            retry_match_cost_after: float | None = None
            retry_lr_error_before: float | None = None
            retry_lr_error_after: float | None = None
            post_match_policy = self._match_accuracy_policy(
                match,
                state,
                base_search_radius_px=adaptive_radius,
                flow_fb_error_px=flow_error,
                right_flow_fb_error_px=right_flow_error,
                precision_retry_count=0,
            )
            for retry_index in range(1, self.config.max_precision_retry + 1):
                if (
                    post_match_policy is None
                    or post_match_policy.precision_status != "RETRY_STRONGER"
                ):
                    break
                retry_triggered = True
                retry_sigma_before = post_match_policy.estimated_sigma_d_px
                retry_match_cost_before = match.cost
                retry_lr_error_before = match.lr_error_px
                retry_level = min(
                    refinement_level + retry_index,
                    self.config.max_precision_refinement_level,
                )
                retry_match = matcher.match(
                    left_gray,
                    right_gray,
                    state.left_xy,
                    predicted,
                    self.profile,
                    latest_disparity=latest,
                    temporal_right_xy=temporal_right_xy,
                    right_flow_fb_error_px=right_flow_error,
                    search_radius=adaptive_radius,
                    refinement_level=retry_level,
                )
                precision_retry_count = retry_index
                if (
                    retry_match.status == "valid"
                    and retry_match.disparity is not None
                    and retry_match.right_xy is not None
                ):
                    retry_usable = True
                    if temporal_right_xy is not None:
                        retry_cycle = evaluate_cycle_consistency(
                            retry_match.right_xy,
                            temporal_right_xy,
                            self.config,
                        )
                        retry_usable = retry_cycle.status != "cycle_failed"
                        if retry_usable:
                            retry_match = replace(
                                retry_match,
                                confidence=retry_match.confidence * retry_cycle.confidence_scale,
                                cycle_error_px=retry_cycle.error_px,
                                cycle_cost=retry_cycle.cost,
                            )
                    if retry_usable:
                        retry_policy = self._match_accuracy_policy(
                            retry_match,
                            state,
                            base_search_radius_px=adaptive_radius,
                            flow_fb_error_px=flow_error,
                            right_flow_fb_error_px=right_flow_error,
                            precision_retry_count=retry_index,
                        )
                        if (
                            retry_policy is not None
                            and retry_policy.estimated_sigma_d_px is not None
                            and (
                                post_match_policy.estimated_sigma_d_px is None
                                or retry_policy.estimated_sigma_d_px
                                <= post_match_policy.estimated_sigma_d_px
                            )
                        ):
                            match = retry_match
                            retry_candidate_accepted = True
                            retry_sigma_after = retry_policy.estimated_sigma_d_px
                            retry_match_cost_after = retry_match.cost
                            retry_lr_error_after = retry_match.lr_error_px
                        elif retry_policy is not None:
                            retry_sigma_after = retry_policy.estimated_sigma_d_px
                            retry_match_cost_after = retry_match.cost
                            retry_lr_error_after = retry_match.lr_error_px
                post_match_policy = self._match_accuracy_policy(
                    match,
                    state,
                    base_search_radius_px=adaptive_radius,
                    flow_fb_error_px=flow_error,
                    right_flow_fb_error_px=right_flow_error,
                    precision_retry_count=retry_index,
                )
            if precision_retry_count:
                matching_ms = (time.perf_counter() - matching_start) * 1000.0
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
                    neighbor_disparity_mad=match.neighbor_disparity_mad,
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
                point_result = self._annotate_accuracy_policy(
                    point_result,
                    base_search_radius_px=adaptive_radius,
                    precision_retry_count=precision_retry_count,
                    warmup=False,
                )
                point_result = replace(
                    point_result,
                    retry_triggered=retry_triggered,
                    retry_candidate_accepted=retry_candidate_accepted if retry_triggered else None,
                    retry_sigma_before_px=retry_sigma_before,
                    retry_sigma_after_px=retry_sigma_after,
                    retry_match_cost_before=retry_match_cost_before,
                    retry_match_cost_after=retry_match_cost_after,
                    retry_lr_error_before_px=retry_lr_error_before,
                    retry_lr_error_after_px=retry_lr_error_after,
                )
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
        return self._finalize_frame_results(results)

    def _finalize_frame_results(self, results: list[FramePointResult]) -> list[FramePointResult]:
        completed = self._apply_camera_compensation(results)
        if self.shadow_analyzer is None:
            shadowed = completed
        else:
            shadowed = self.shadow_analyzer.process_frame(completed)
        if self.experiment_authority is None:
            return shadowed
        try:
            return self._process_enhanced_results(shadowed)
        except Exception as exc:
            return [self._enhanced_fallback(result, exc) for result in shadowed]

    def _process_enhanced_results(
        self,
        results: list[FramePointResult],
    ) -> list[FramePointResult]:
        """Run experiment-only I2/I3/arbitration after the I1 result is frozen."""

        if self.experiment_authority is None:
            return results
        processed: list[FramePointResult] = []
        for result in results:
            baseline = I1BaselineView.from_result(result)
            xyz_mm = (
                tuple(float(value) * 1000.0 for value in baseline.xyz_m)
                if baseline.xyz_m is not None
                else (float("nan"), float("nan"), float("nan"))
            )
            hard_failure = not baseline.valid
            fault_class = result.fault_class.strip().upper()
            confirmed_anomaly = (
                not hard_failure
                and fault_class not in {"", "NORMAL"}
                and result.c_phy_valid is True
                and result.c_phy is not None
                and result.c_phy < 0.55
            )
            evidence = I2Evidence(
                confirmed_anomaly=confirmed_anomaly,
                legitimate_motion=bool(result.transient_protected),
                hard_failure=hard_failure,
                geometry_valid=baseline.valid,
                evidence_sufficient=(result.c_phy_valid is True) if confirmed_anomaly else True,
                post_correction_safe=True,
                suspicious=(fault_class not in {"", "NORMAL"}),
            )
            processor = self.enhanced_processors.setdefault(result.point_id, EnhancedPointProcessor())
            outcome = processor.process(
                TimedObservation(
                    frame=result.frame,
                    timestamp_s=float(result.frame),
                    xyz_mm=xyz_mm,
                    evidence=evidence,
                )
            )
            diagnosis = self._structured_i3_recommendation(result, hard_failure=hard_failure)
            decision = self.final_arbitrator.decide(
                baseline=baseline,
                candidate_safety=outcome.candidate_safety,
                diagnosis=diagnosis,
                authority=self.experiment_authority,
            )
            prediction = outcome.prediction_xyz_mm
            audited = replace(
                result,
                i1_status=baseline.status,
                i2_state=outcome.state.value,
                i2_episode_id=outcome.episode_id,
                i2_prediction_x_m=(None if prediction is None else prediction[0] / 1000.0),
                i2_prediction_y_m=(None if prediction is None else prediction[1] / 1000.0),
                i2_prediction_z_m=(None if prediction is None else prediction[2] / 1000.0),
                candidate_safe=outcome.candidate_safety.safe,
                candidate_safety_reasons=";".join(outcome.candidate_safety.failed_reasons),
            )
            if decision.write_committed:
                processed.append(self._apply_final_decision(audited, decision))
            else:
                processed.append(
                    replace(
                        audited,
                        final_valid=baseline.valid,
                        proposed_decision=decision.proposed_decision.value,
                        committed_decision=decision.committed_decision.value,
                        write_committed=False,
                        result_source=decision.result_source.value,
                        final_decision_reason=decision.reason,
                    )
                )
        return processed

    @staticmethod
    def _structured_i3_recommendation(
        result: FramePointResult,
        *,
        hard_failure: bool,
    ) -> I3Recommendation:
        if hard_failure:
            return I3Recommendation(I3Risk.BLOCKING, "i1_hard_failure", I3Action.BLOCK_FINAL)
        fault_class = result.fault_class.strip().upper()
        if fault_class in {"", "NORMAL"}:
            return I3Recommendation.normal()
        if (result.fault_confidence or 0.0) >= 0.8:
            return I3Recommendation(I3Risk.BLOCKING, f"i3:{fault_class}", I3Action.BLOCK_FINAL)
        if fault_class == "STEREO_MISMATCH":
            return I3Recommendation(I3Risk.WARNING, f"i3:{fault_class}", I3Action.ALLOW_CORRECTION)
        return I3Recommendation(I3Risk.WARNING, f"i3:{fault_class}", I3Action.WARN)

    @staticmethod
    def _enhanced_fallback(result: FramePointResult, error: Exception) -> FramePointResult:
        """Keep the original I1 final value and validity if enhanced processing fails."""

        baseline = I1BaselineView.from_result(result)
        if baseline.valid:
            return replace(
                result,
                final_valid=True,
                proposed_decision=FinalDecision.REJECT.value,
                committed_decision=FinalDecision.ACCEPT_WITH_WARNING.value,
                write_committed=False,
                result_source=ResultSource.I1_BASELINE.value,
                final_decision_reason=f"enhanced_exception_fallback:{type(error).__name__}",
            )
        return replace(
            result,
            final_valid=False,
            proposed_decision=FinalDecision.REJECT.value,
            committed_decision=FinalDecision.REJECT.value,
            write_committed=False,
            result_source=ResultSource.REJECTED.value,
            final_decision_reason=f"enhanced_exception_invalid_baseline:{type(error).__name__}",
        )

    def _apply_final_decision(
        self,
        result: FramePointResult,
        decision: ArbitrationDecision,
    ) -> FramePointResult:
        """Atomically apply an already-authorized experimental final decision.

        The arbitrator itself is pure.  This method is intentionally the only
        Phase 4 location that can alter the final coordinate view.
        """

        if not decision.write_committed:
            return result
        if not decision.authority.may_write_final:
            raise PermissionError("experimental authority is required for final writes")
        if decision.committed_decision is FinalDecision.USE_CORRECTED:
            xyz = decision.final_xyz_m
            if (
                not decision.final_valid
                or decision.result_source is not ResultSource.I2_CORRECTED
                or xyz is None
                or not np.isfinite(np.asarray(xyz, dtype=float)).all()
            ):
                raise ValueError("invalid corrected final commit")
            final_distance = float(np.linalg.norm(np.asarray(xyz, dtype=float)))
            return replace(
                result,
                final_x_m=float(xyz[0]),
                final_y_m=float(xyz[1]),
                final_z_m=float(xyz[2]),
                final_distance_m=final_distance,
                final_valid=True,
                proposed_decision=decision.proposed_decision.value,
                committed_decision=decision.committed_decision.value,
                write_committed=True,
                result_source=decision.result_source.value,
                final_decision_reason=decision.reason,
            )
        if decision.committed_decision is FinalDecision.REJECT:
            if decision.final_valid or decision.result_source is not ResultSource.REJECTED:
                raise ValueError("invalid reject final commit")
            return replace(
                result,
                status="rejected",
                final_x_m=None,
                final_y_m=None,
                final_z_m=None,
                final_distance_m=None,
                final_valid=False,
                proposed_decision=decision.proposed_decision.value,
                committed_decision=decision.committed_decision.value,
                write_committed=False,
                result_source=decision.result_source.value,
                final_decision_reason=decision.reason,
            )
        raise ValueError("only corrected or rejected decisions may write a final result")

    def _accuracy_geometry(self) -> tuple[float, float]:
        focal_length_px = abs(float(self.q[2, 3]))
        inverse_baseline = abs(float(self.q[3, 2]))
        if focal_length_px <= 0.0 or inverse_baseline <= 0.0:
            raise ValueError("target-accuracy policy requires a standard finite Q geometry")
        baseline = 1.0 / inverse_baseline
        baseline_mm = baseline * 1000.0 if self.calibration_unit == "m" else baseline
        return focal_length_px, baseline_mm

    def _pre_match_accuracy_policy(
        self,
        state: PointState,
        base_search_radius_px: int,
    ) -> MeasurementPolicyDecision | None:
        if self.target_accuracy_spec is None:
            return None
        xyz = state.estimated_xyz or state.xyz
        if xyz is None or not np.isfinite(xyz[2]) or xyz[2] <= 0.0:
            return None
        variance = (
            state.last_disparity_variance_px2
            if state.last_disparity_variance_px2 is not None
            else self.config.uncertainty_base_disparity_variance_px2
        )
        focal_length_px, baseline_mm = self._accuracy_geometry()
        plan = build_precision_plan(
            self.target_accuracy_spec,
            current_depth_m=float(xyz[2]),
            focal_length_px=focal_length_px,
            baseline_mm=baseline_mm,
            estimated_sigma_disparity_px=float(np.sqrt(variance)),
            minimum_achievable_sigma_disparity_px=float(
                np.sqrt(self.config.uncertainty_min_disparity_variance_px2)
            ),
        )
        return decide_measurement_policy(
            plan,
            base_search_radius_px=base_search_radius_px,
            vision_state=state.confidence_state,
            max_precision_retry=self.config.max_precision_retry,
            max_refinement_level=self.config.max_precision_refinement_level,
        )

    def _match_accuracy_policy(
        self,
        match: LocalMatchResult,
        state: PointState,
        *,
        base_search_radius_px: int,
        flow_fb_error_px: float | None,
        right_flow_fb_error_px: float | None,
        precision_retry_count: int,
    ) -> MeasurementPolicyDecision | None:
        if self.target_accuracy_spec is None or match.disparity is None:
            return None
        disparity = float(
            match.measured_disparity
            if match.measured_disparity is not None
            else match.raw_disparity
            if match.raw_disparity is not None
            else match.disparity
        )
        try:
            xyz = reproject_point_m(
                state.left_xy[0], state.left_xy[1], disparity,
                self.q, self.calibration_unit,
            )
        except ValueError:
            return None
        uncertainty = estimate_match_uncertainty(
            texture_std=match.texture_std,
            photo_cost=match.cost,
            uniqueness_margin=match.uniqueness_margin_value,
            cost_curvature=match.cost_curvature,
            flow_fb_error_px=flow_fb_error_px,
            right_flow_fb_error_px=right_flow_fb_error_px,
            lr_error_px=match.lr_error_px,
            cycle_error_px=match.cycle_error_px,
            icgn_residual=match.icgn_residual,
            icgn_hessian=match.icgn_hessian,
            config=self.config,
            icgn_hessian_density=match.icgn_hessian_density,
            neighbor_disparity_mad=match.neighbor_disparity_mad,
        )
        focal_length_px, baseline_mm = self._accuracy_geometry()
        plan = build_precision_plan(
            self.target_accuracy_spec,
            current_depth_m=float(xyz[2]),
            focal_length_px=focal_length_px,
            baseline_mm=baseline_mm,
            estimated_sigma_disparity_px=float(
                np.sqrt(uncertainty.disparity_variance_px2)
            ),
            minimum_achievable_sigma_disparity_px=float(
                np.sqrt(self.config.uncertainty_min_disparity_variance_px2)
            ),
        )
        return decide_measurement_policy(
            plan,
            base_search_radius_px=base_search_radius_px,
            vision_state=state.confidence_state,
            max_precision_retry=self.config.max_precision_retry,
            max_refinement_level=self.config.max_precision_refinement_level,
            precision_retry_count=precision_retry_count,
        )

    def _annotate_accuracy_policy(
        self,
        result: FramePointResult,
        *,
        base_search_radius_px: int,
        precision_retry_count: int,
        warmup: bool,
    ) -> FramePointResult:
        if self.target_accuracy_spec is None:
            return result
        depth_m = next(
            (
                float(value)
                for value in (result.estimated_z_m, result.measured_z_m, result.z_m)
                if value is not None and np.isfinite(value) and value > 0.0
            ),
            None,
        )
        common = {
            "accuracy_policy_enabled": True,
            "precision_policy_warmup": warmup,
            "target_metric": self.target_accuracy_spec.metric,
            "target_x_mm": self.target_accuracy_spec.target_sigma_x_mm,
            "target_y_mm": self.target_accuracy_spec.target_sigma_y_mm,
            "target_z_mm": self.target_accuracy_spec.target_sigma_z_mm,
            "policy_base_search_radius_px": base_search_radius_px,
            "policy_precision_retry_count": precision_retry_count,
        }
        if depth_m is None:
            return replace(
                result,
                **common,
                precision_feasible=False,
                precision_status="UNAVAILABLE",
                limiting_axis="z",
                policy_final_search_radius_px=base_search_radius_px,
                policy_refinement_level=0,
                policy_retry_budget=0,
                policy_acceptance_reason="stereo_measurement_unavailable",
                policy_reason="causal_depth_unavailable",
            )
        focal_length_px, baseline_mm = self._accuracy_geometry()
        estimated_sigma_d = (
            float(np.sqrt(result.disparity_variance_px2))
            if result.disparity_variance_px2 is not None
            and np.isfinite(result.disparity_variance_px2)
            and result.disparity_variance_px2 > 0.0
            else None
        )
        plan = build_precision_plan(
            self.target_accuracy_spec,
            current_depth_m=depth_m,
            focal_length_px=focal_length_px,
            baseline_mm=baseline_mm,
            estimated_sigma_disparity_px=estimated_sigma_d,
            minimum_achievable_sigma_disparity_px=float(
                np.sqrt(self.config.uncertainty_min_disparity_variance_px2)
            ),
        )
        decision = decide_measurement_policy(
            plan,
            base_search_radius_px=base_search_radius_px,
            vision_state=result.confidence_state,
            max_precision_retry=self.config.max_precision_retry,
            max_refinement_level=self.config.max_precision_refinement_level,
            precision_retry_count=precision_retry_count,
            warmup=warmup,
        )
        return replace(
            result,
            **common,
            required_sigma_d_px=decision.required_sigma_d_px,
            estimated_sigma_d_px=decision.estimated_sigma_d_px,
            estimated_sigma_x_mm=decision.estimated_sigma_x_mm,
            estimated_sigma_y_mm=decision.estimated_sigma_y_mm,
            estimated_sigma_z_mm=decision.estimated_sigma_z_mm,
            precision_ratio=decision.precision_ratio,
            precision_feasible=decision.precision_feasible,
            precision_status=decision.precision_status,
            limiting_axis=decision.limiting_axis,
            policy_final_search_radius_px=decision.final_search_radius_px,
            policy_refinement_level=decision.refinement_level,
            policy_retry_budget=decision.retry_budget,
            policy_acceptance_reason=decision.acceptance_reason,
            policy_reason=decision.policy_reason,
        )

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

    def _determine_search_radius(
        self,
        state: PointState,
        disparity_velocity: float | None,
        recovery_attempt: bool,
    ) -> tuple[int, str]:
        """Innovation-1 base window: motion and prediction error only."""
        if not self.profile.adaptive_search:
            return self.config.search_radius, "fixed"
        if recovery_attempt or state.status in {"lost", "recovering"}:
            return self.config.adaptive_search_recovery_radius, "base_recovery"
        if (
            (disparity_velocity is None or abs(disparity_velocity) <= 1.0)
            and (state.last_prediction_residual_px is None or state.last_prediction_residual_px <= 1.0)
        ):
            return self.config.adaptive_search_small_radius, "stable_prediction"
        if disparity_velocity is not None and abs(disparity_velocity) > self.config.max_disparity_velocity * 0.5:
            return self.config.adaptive_search_large_radius, "high_motion"
        if state.last_prediction_residual_px is not None and state.last_prediction_residual_px > 1.0:
            return self.config.adaptive_search_large_radius, "prediction_error"
        return self.config.adaptive_search_normal_radius, "normal"

    def _with_confidence_feedback(
        self,
        result: FramePointResult,
        state: PointState,
        *,
        predicted_disparity: float,
        previous_disparity: float | None,
        disparity_velocity: float | None,
        prediction_residual: float,
        adaptive_radius: int,
        radius_reason: str,
    ) -> FramePointResult:
        """Export interpretable confidence and advance the feedback state machine."""
        uncertainty = estimate_match_uncertainty(
            texture_std=result.texture_std,
            photo_cost=result.match_cost,
            uniqueness_margin=result.uniqueness_margin_value,
            cost_curvature=result.cost_curvature,
            flow_fb_error_px=result.flow_fb_error_px,
            right_flow_fb_error_px=result.right_flow_fb_error_px,
            lr_error_px=result.lr_error_px,
            cycle_error_px=result.cycle_error_px,
            icgn_residual=result.icgn_residual,
            icgn_hessian=result.icgn_hessian,
            config=self.config,
            icgn_hessian_density=result.icgn_hessian_density,
        )
        confidence = float(np.clip(uncertainty.quality_score * np.exp(-prediction_residual / max(self.config.adaptive_search_normal_radius, 1)), 0.0, 1.0))
        if confidence >= self.config.confidence_high_threshold:
            confidence_state = "HIGH"
            state.low_confidence_frames = 0
        elif confidence >= self.config.confidence_medium_threshold:
            confidence_state = "MEDIUM"
            state.low_confidence_frames = 0
        else:
            state.low_confidence_frames += 1
            confidence_state = "LOST" if state.low_confidence_frames >= self.config.confidence_low_to_lost_frames else "LOW"
        state.confidence_state = confidence_state
        components = {key: float(np.clip(1.0 - value / self.config.uncertainty_max_component, 0.0, 1.0)) for key, value in uncertainty.components.items()}
        return replace(
            result,
            confidence=confidence,
            measurement_quality_score=confidence,
            predicted_disparity=predicted_disparity,
            previous_disparity=previous_disparity,
            disparity_velocity=disparity_velocity,
            prediction_residual_px=prediction_residual,
            adaptive_search_radius=adaptive_radius,
            search_radius_reason=radius_reason,
            confidence_texture=components.get("texture"), confidence_photo=components.get("photo"),
            confidence_margin=components.get("margin"), confidence_flow=components.get("flow"),
            confidence_lr=components.get("lr"), confidence_cycle=components.get("cycle"),
            confidence_icgn=components.get("icgn"), confidence_curvature=components.get("curvature"),
            confidence_temporal=float(np.exp(-prediction_residual / max(self.config.adaptive_search_normal_radius, 1))),
            confidence_total=confidence,
            confidence_state=confidence_state,
        )

    def _apply_camera_compensation(
        self,
        results: list[FramePointResult],
    ) -> list[FramePointResult]:
        if not (self.profile.use_camera_compensation and self.config.enable_camera_compensation) or self.config.camera_compensation_mode == "none":
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
        if self.config.camera_compensation_mode == "single_reference" and current_points:
            current = np.asarray(current_points[0], dtype=np.float64)
            initial = np.asarray(initial_points[0], dtype=np.float64)
            translation = initial - current
            compensation = CameraCompensationResult(
                "single_reference", np.eye(3), translation,
                (reference_results[0].point_id,), 0.0,
            )
        elif current_points:
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
            compensation.status in {"valid", "single_reference"}
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
            rotation_xyz_deg = rotation_matrix_to_euler_xyz_deg(compensation.rotation)
        else:
            rotation_angle = None
            translation_mm = np.asarray([None, None, None], dtype=object)
            rotation_xyz_deg = (None, None, None)

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
                    camera_rx_deg=rotation_xyz_deg[0],
                    camera_ry_deg=rotation_xyz_deg[1],
                    camera_rz_deg=rotation_xyz_deg[2],
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
        neighbor_disparity_mad: float | None = None,
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
        # Confidence is deliberately computed before Kalman/PointState mutation.
        # Optional unavailable evidence is excluded by uncertainty.py.
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
                neighbor_disparity_mad=neighbor_disparity_mad,
            )
        prediction_residual = abs(measured - (state.latest_disparity or measured))
        score = float(np.clip(uncertainty.quality_score * np.exp(-prediction_residual / max(self.config.adaptive_search_normal_radius, 1)), 0.0, 1.0))
        decision_mode = self.config.confidence_mode if self.profile.use_confidence_feedback else "none"
        decision, state.low_confidence_frames = decide_confidence(
            score, state.low_confidence_frames,
            high=self.config.confidence_high_threshold, medium=self.config.confidence_medium_threshold,
            low=self.config.confidence_low_threshold, lost_after=self.config.confidence_low_to_lost_frames,
            mode=decision_mode, uniqueness_margin=uniqueness_margin_value,
            single_margin_reference=self.config.uncertainty_margin_reference,
        )
        state.confidence_state = decision.state
        state.last_prediction_residual_px = prediction_residual
        left_variance_px2 = uncertainty.left_position_variance_px2
        disparity_variance_px2 = uncertainty.disparity_variance_px2
        measurement_quality_score = score
        measurement_accepted_for_state = decision.accept_measurement
        state_update_source = "measured" if decision.accept_measurement else "predicted"
        if self.config.confidence_mode in {"single_margin", "lr_only", "multi_reject"} and not decision.accept_measurement:
            return replace(
                FramePointResult(
                    method=self.method, frame=frame, point_id=state.point_id,
                    status="confidence_rejected", left_x=state.left_xy[0], left_y=state.left_xy[1],
                    measured_right_x=right_xy[0], measured_right_y=right_xy[1],
                    measured_disparity=measured, match_cost=match_cost, lr_error_px=lr_error,
                    confidence=score, measurement_accepted_for_state=False,
                    state_update_source="rejected", measurement_quality_score=score,
                    confidence_total=score, confidence_state=decision.state,
                    neighbor_disparity_mad=neighbor_disparity_mad,
                ),
                recovery_stage=recovery_stage,
            )
        use_confidence_filter = self.profile.use_confidence_feedback and self.config.confidence_mode == "closed_loop"
        if (self.profile.use_adaptive_filter or use_confidence_filter) and self.config.enable_adaptive_filter:
            point_filter = self.filters.get(state.point_id)
            if point_filter is None:
                if decision.accept_measurement:
                    point_filter = AdaptivePointKalman(self.config)
                    point_filter.initialize(state.left_xy, measured)
                    self.filters[state.point_id] = point_filter
                    kalman_update_status = "initialized"
                else:
                    kalman_update_status = "not_initialized_rejected"
            else:
                point_filter.predict()
                if decision.accept_measurement:
                    update = point_filter.update(state.left_xy, measured, left_variance_px2, disparity_variance_px2)
                    estimated_left_xy, estimated = update.estimated_left_xy, update.estimated_disparity
                    kalman_innovation, kalman_innovation_norm = update.innovation, update.innovation_norm
                    kalman_gain_disparity, kalman_nis, kalman_update_status = update.kalman_gain_disparity, update.nis, update.status
                    measurement_accepted_for_state = update.status in {"updated", "variance_inflated"}
                else:
                    estimated_left_xy, estimated = point_filter.current_measurement_state()
                    kalman_update_status = "predict_only_confidence"
                    point_filter.predict_only_frames += 1
                kalman_predict_only_frames = point_filter.predict_only_frames
                state_update_source = (
                    "filtered" if measurement_accepted_for_state else "predicted"
                )
                kalman_recovery_needed = (
                    (not measurement_accepted_for_state)
                    and point_filter.predict_only_frames >= self.config.kalman_max_predict_only_frames
                )
                state.last_kalman_innovation = kalman_innovation_norm
                state.last_kalman_gain_disparity = kalman_gain_disparity
                state.last_nis = kalman_nis
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
            (self.profile.adaptive_search or self.profile.use_confidence_feedback or self.method == "full_quality")
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
        state.confirm_valid(score)
        if decision.state == "LOST":
            state.status = "lost"
        elif decision.trigger_recovery or kalman_recovery_needed:
            state.status = "recovering"
            if state.recovery_started_frame is None:
                state.recovery_started_frame = frame
        if was_recovering and measurement_accepted_for_state:
            state.confirm_recovery(frame)
            state.confidence_state = "MEDIUM"
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
            neighbor_disparity_mad=neighbor_disparity_mad,
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
            confidence_texture=_confidence_component(uncertainty.components.get("texture"), self.config),
            confidence_photo=_confidence_component(uncertainty.components.get("photo"), self.config),
            confidence_margin=_confidence_component(uncertainty.components.get("margin"), self.config),
            confidence_flow=_confidence_component(uncertainty.components.get("flow"), self.config),
            confidence_lr=_confidence_component(uncertainty.components.get("lr"), self.config),
            confidence_cycle=_confidence_component(uncertainty.components.get("cycle"), self.config),
            confidence_icgn=_confidence_component(uncertainty.components.get("icgn"), self.config),
            confidence_curvature=_confidence_component(uncertainty.components.get("curvature"), self.config),
            confidence_neighbor=_confidence_component(uncertainty.components.get("neighbor"), self.config),
            confidence_temporal=float(np.exp(-prediction_residual / max(self.config.adaptive_search_normal_radius, 1))),
            confidence_total=score,
            confidence_state=decision.state,
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
