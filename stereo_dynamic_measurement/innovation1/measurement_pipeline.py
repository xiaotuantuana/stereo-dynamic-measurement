from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..calibration.camera_model import StereoCameraModel
from ..calibration.triangulation import triangulate_points
from .adaptive_search import AdaptiveSearchConfig, compute_search_radius
from .confidence import neighbor_disparity_residual
from .gradient_quality import assess_image_quality
from .motion_predictor import MotionPredictor, PointMeasurementState
from .optical_flow import track_forward_backward
from .precision_planner import required_disparity_precision
from .stereo_matcher import LocalStereoMatcher
from .subpixel_refiner import refine_parabolic


@dataclass(frozen=True)
class Innovation1Config:
    motion_model: str = "kalman"
    dt_s: float = 1.0 / 30.0
    patch_size: int = 15
    confidence_high: float = 0.75
    confidence_medium: float = 0.45
    confidence_low: float = 0.25
    fb_threshold_px: float = 1.0
    reinitialize_after: int = 3
    adaptive_search: AdaptiveSearchConfig = field(default_factory=AdaptiveSearchConfig)


@dataclass(frozen=True)
class AdaptiveMeasurementResult:
    point_id: str
    disparity_px: float | None
    xyz_mm: np.ndarray | None
    confidence: float
    confidence_state: str
    confidence_components: dict[str, float]
    search_radius_px: int
    fallback_used: bool
    action: str
    status: str


class AdaptiveStereoMeasurementPipeline:
    """Innovation-1 closed loop: visual evidence controls matching, fallback and prediction."""

    def __init__(self, camera: StereoCameraModel, config: Innovation1Config = Innovation1Config()) -> None:
        self.camera, self.config = camera, config
        self.predictor = MotionPredictor(config.motion_model, config.dt_s)
        self.matcher = LocalStereoMatcher(config.patch_size)
        self._low_counts: dict[str, int] = {}

    def initialize(self, state: PointMeasurementState) -> None:
        self.predictor.update(state)
        self._low_counts[state.point_id] = 0

    def process(self, previous_left: np.ndarray, current_left: np.ndarray, current_right: np.ndarray, *, point_id: str, target_depth_error_mm: float, neighbor_disparities_px: list[float] | None = None) -> AdaptiveMeasurementResult:
        prediction = self.predictor.predict(point_id)
        quality = assess_image_quality(current_left, prediction.xy_px, self.config.patch_size)
        flow = track_forward_backward(previous_left, current_left, self.predictor._states[point_id].xy_px, initial_velocity=(prediction.xy_px[0] - self.predictor._states[point_id].xy_px[0], prediction.xy_px[1] - self.predictor._states[point_id].xy_px[1]), fb_threshold_px=self.config.fb_threshold_px)
        left_point = flow.point if flow.valid and flow.point is not None else prediction.xy_px
        temporal_residual = float(np.linalg.norm(np.asarray(left_point) - np.asarray(prediction.xy_px)))
        required_precision = required_disparity_precision(prediction.xyz_mm[2], self.camera.baseline_mm, self.camera.K_left[0, 0], target_depth_error_mm)
        decision = compute_search_radius(self.config.adaptive_search, gradient_quality=quality.quality_score, fb_error_px=flow.fb_error_px, prediction_residual_px=temporal_residual, previous_confidence=prediction.confidence, required_disparity_precision_px=required_precision)
        local = self.matcher.match(current_left, current_right, left_point=left_point, predicted_disparity_px=prediction.predicted_disparity_px, search_radius_px=decision.radius_px)
        candidate, lr_error = self._refine_and_check(current_left, current_right, left_point, local)
        components = self._components(quality.quality_score, local.match_cost, lr_error, flow.fb_error_px, temporal_residual, candidate, neighbor_disparities_px)
        score = self._score(components)
        state = self._state(score)
        fallback = False
        if state == "MEDIUM":
            expanded = min(self.config.adaptive_search.max_radius_px, max(decision.radius_px + 2, decision.radius_px * 2))
            local = self.matcher.match(current_left, current_right, left_point=left_point, predicted_disparity_px=prediction.predicted_disparity_px, search_radius_px=expanded)
            candidate, lr_error = self._refine_and_check(current_left, current_right, left_point, local)
            components = self._components(quality.quality_score, local.match_cost, lr_error, flow.fb_error_px, temporal_residual, candidate, neighbor_disparities_px)
            score = self._score(components)
            state = self._state(score)
            decision = type(decision)(expanded, decision.risk_score, decision.reasons + ("medium_expand",))
        if state == "LOW":
            fallback = True
            candidate = self._sgbm_fallback(current_left, current_right, left_point)
            if candidate is not None:
                score = max(score, self.config.confidence_low)
        if candidate is None:
            self._low_counts[point_id] = self._low_counts.get(point_id, 0) + 1
            action = "reinitialize" if self._low_counts[point_id] >= self.config.reinitialize_after else ("fallback" if fallback else "predict_only")
            return AdaptiveMeasurementResult(point_id, None, None, score, "LOW", components, decision.radius_px, fallback, action, "measurement_unavailable")
        self._low_counts[point_id] = 0 if state != "LOW" else self._low_counts.get(point_id, 0)
        right_point = np.array([[left_point[0] - candidate, left_point[1]]], dtype=np.float64)
        xyz = triangulate_points(np.asarray([left_point]), right_point, self.camera.P1, self.camera.P2)[0]
        updated = PointMeasurementState(point_id, left_point, candidate, tuple(float(v) for v in xyz), self._estimate_velocity(point_id, xyz), float(np.clip(score, 0.0, 1.0)), disparity_velocity_px_s=(candidate - self.predictor._states[point_id].disparity_px) / self.config.dt_s)
        self.predictor.update(updated)
        return AdaptiveMeasurementResult(point_id, candidate, xyz, float(np.clip(score, 0.0, 1.0)), state, components, decision.radius_px, fallback, "fallback" if fallback else ("expand" if "medium_expand" in decision.reasons else "accept"), "valid")

    def _refine_and_check(self, left: np.ndarray, right: np.ndarray, point: tuple[float, float], local) -> tuple[float | None, float | None]:
        if not local.valid or local.integer_disparity_px is None:
            return None, None
        d = local.integer_disparity_px
        refined = refine_parabolic(d, cost_minus=local.cost_curve.get(d - 1.0, np.nan), cost_center=local.cost_curve[d], cost_plus=local.cost_curve.get(d + 1.0, np.nan), at_boundary=d - 1.0 not in local.cost_curve or d + 1.0 not in local.cost_curve)
        disparity = refined.disparity_px
        reverse = self.matcher.match(right, left, left_point=(point[0] - disparity, point[1]), predicted_disparity_px=disparity, search_radius_px=2, direction=-1)
        lr_error = None if reverse.integer_disparity_px is None else abs(disparity - reverse.integer_disparity_px)
        return disparity, lr_error

    def _components(self, quality: float, match_cost: float, lr_error: float | None, fb_error: float | None, temporal: float, disparity: float | None, neighbors: list[float] | None) -> dict[str, float]:
        finite_cost = match_cost if np.isfinite(match_cost) else 10.0
        neighbor = neighbor_disparity_residual(disparity, neighbors or []) if disparity is not None and neighbors else 0.0
        return {
            "gradient": float(np.clip(quality, 0, 1)), "matching": float(np.exp(-max(finite_cost, 0.0))),
            "left_right": 0.0 if lr_error is None else float(np.exp(-lr_error)),
            "flow": 0.0 if fb_error is None else float(np.exp(-fb_error / self.config.fb_threshold_px)),
            "temporal": float(np.exp(-temporal)), "neighbor": float(np.exp(-neighbor)),
        }

    def _state(self, score: float) -> str:
        return "HIGH" if score >= self.config.confidence_high else "MEDIUM" if score >= self.config.confidence_medium else "LOW"

    @staticmethod
    def _score(components: dict[str, float]) -> float:
        weights = {"gradient": 0.30, "matching": 0.25, "left_right": 0.15, "flow": 0.15, "temporal": 0.10, "neighbor": 0.05}
        return float(sum(weights[key] * components[key] for key in weights))

    def _sgbm_fallback(self, left: np.ndarray, right: np.ndarray, point: tuple[float, float]) -> float | None:
        max_disparities = 64
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disparities, blockSize=5, P1=8 * 25, P2=32 * 25, uniquenessRatio=5, speckleWindowSize=0)
        disparity = sgbm.compute(left, right).astype(np.float32) / 16.0
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]): return None
        value = float(disparity[y, x])
        return value if np.isfinite(value) and value > 0 else None

    def _estimate_velocity(self, point_id: str, xyz: np.ndarray) -> tuple[float, float, float]:
        previous = self.predictor._states[point_id]
        return tuple(float((xyz[i] - previous.xyz_mm[i]) / self.config.dt_s) for i in range(3))
