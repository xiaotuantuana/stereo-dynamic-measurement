from __future__ import annotations

import time
from dataclasses import replace
from typing import Iterable

import cv2
import numpy as np

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
from .tracking import track_point_lk


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
            if method == "full_quality"
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
        self.flow_reference_gray: dict[str, np.ndarray] = {}
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
            results.append(
                self._valid_result(
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
            )
        self.previous_left_gray = left_gray.copy()
        self.initialized = True
        return results

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
                    self.method == "full_quality"
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
                and self.method == "full_quality"
                and self.config.enable_recovery
            ):
                recovery_attempt = True
                recovery_stage = "attempt"
                state.recovery_attempt_count += 1
                if state.recovery_started_frame is None:
                    state.recovery_started_frame = frame

            flow_ms = 0.0
            flow_error: float | None = None
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

            latest = state.disparity
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
                        results.append(
                            self._valid_result(
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
                    state.previous_disparity,
                    self.config.max_disparity_velocity,
                )
                if self.profile.use_prediction and self.config.enable_prediction
                else latest
            )
            matching_start = time.perf_counter()
            matcher = self.recovery_matcher if recovery_attempt else self.local_matcher
            match = matcher.match(
                left_gray,
                right_gray,
                state.left_xy,
                predicted,
                self.profile,
                latest_disparity=latest,
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
            results.append(
                self._valid_result(
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
                )
            )
        self.previous_left_gray = left_gray.copy()
        return results

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
        latest_valid = state.disparity
        if (
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
            float(state.left_xy[0] - estimated),
            float(right_xy[1]),
        )
        try:
            estimated_xyz_array = reproject_point_m(
                state.left_xy[0],
                state.left_xy[1],
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
            self.method == "full_quality"
            and self.config.enable_recovery
            and state.status in {"recovering", "lost"}
        )
        state.record_stereo_measurement(
            right_xy,
            measured,
            confidence,
            estimated_disparity=estimated,
            xyz=measured_xyz,
            estimated_xyz=estimated_xyz,
        )
        state.confirm_valid(confidence)
        if was_recovering:
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
