from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast


MethodName = Literal[
    "M0",
    "M1",
    "M2",
    "M3",
    "THESIS_FULL",
    "sgbm",
    "sgbm_fixed",
    "local",
    "local_fixed",
    "local_flow",
    "full",
    "full_quality",
    "sgbm_flow",
    "research_full",
]
METHOD_NAMES: tuple[MethodName, ...] = (
    "M0", "M1", "M2", "M3", "THESIS_FULL",
    "sgbm",
    "sgbm_fixed",
    "local",
    "local_fixed",
    "local_flow",
    "full",
    "full_quality",
    "sgbm_flow",
    "research_full",
)


class SystemMode(str, Enum):
    """System-level runtime intent; distinct from thesis MethodProfile labels."""

    BASELINE = "BASELINE"
    INNOVATION1 = "INNOVATION1"
    ENHANCED_SHADOW = "ENHANCED_SHADOW"
    FULL_ENHANCED = "FULL_ENHANCED"


class Innovation2Mode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    CONTROLLED = "CONTROLLED"


class Innovation3Mode(str, Enum):
    OFF = "OFF"
    DIAGNOSE = "DIAGNOSE"
    CONTROLLED_RECOVERY = "CONTROLLED_RECOVERY"


FINAL_RESULT_OWNER = "TemporalStereoPipeline"


@dataclass(frozen=True)
class MethodProfile:
    dense_sgbm_each_frame: bool = False
    use_flow: bool = False
    use_prediction: bool = False
    use_epipolar: bool = False
    use_neighborhood: bool = False
    use_subpixel: bool = False
    use_lr_check: bool = False
    use_cycle_consistency: bool = False
    use_icgn: bool = False
    use_adaptive_filter: bool = False
    use_camera_compensation: bool = False
    adaptive_search: bool = False
    use_confidence_feedback: bool = False


_METHOD_PROFILES: dict[MethodName, MethodProfile] = {
    # Thesis-facing, mutually controlled Innovation-1 comparisons.
    "M0": MethodProfile(dense_sgbm_each_frame=True),
    "M1": MethodProfile(use_flow=True),
    "M2": MethodProfile(use_flow=True, use_prediction=True),
    "M3": MethodProfile(
        use_flow=True, use_prediction=True, use_epipolar=True,
        use_neighborhood=True, use_subpixel=True, use_lr_check=True,
        adaptive_search=True,
    ),
    "THESIS_FULL": MethodProfile(
        use_flow=True, use_prediction=True, use_epipolar=True,
        use_neighborhood=True, use_subpixel=True, use_lr_check=True,
        adaptive_search=True, use_confidence_feedback=True,
        use_camera_compensation=True,
    ),
    "sgbm": MethodProfile(dense_sgbm_each_frame=True),
    "sgbm_fixed": MethodProfile(dense_sgbm_each_frame=True),
    "local": MethodProfile(),
    "local_fixed": MethodProfile(),
    "local_flow": MethodProfile(use_flow=True),
    "full": MethodProfile(
        use_flow=True,
        use_prediction=True,
        use_epipolar=True,
        use_neighborhood=True,
        use_subpixel=True,
        use_lr_check=True,
    ),
    "full_quality": MethodProfile(
        use_flow=True,
        use_prediction=True,
        use_epipolar=True,
        use_neighborhood=True,
        use_subpixel=True,
        use_lr_check=True,
    ),
    "sgbm_flow": MethodProfile(dense_sgbm_each_frame=True, use_flow=True),
    "research_full": MethodProfile(
        use_flow=True,
        use_prediction=True,
        use_epipolar=True,
        use_neighborhood=True,
        use_subpixel=True,
        use_lr_check=True,
        use_cycle_consistency=True,
        use_icgn=True,
        use_adaptive_filter=True,
        use_camera_compensation=True,
        adaptive_search=True,
    ),
}


def method_profile(method: str) -> MethodProfile:
    if method not in _METHOD_PROFILES:
        raise ValueError(f"Unknown method {method!r}; expected one of {', '.join(METHOD_NAMES)}")
    return _METHOD_PROFILES[cast(MethodName, method)]


@dataclass(frozen=True)
class MatcherConfig:
    patch_size: int = 11
    search_radius: int = 8
    expanded_search_radius: int = 16
    vertical_radius: int = 1
    support_offset: int = 3
    min_texture_std: float = 5.0
    support_intensity_threshold: float = 20.0
    max_photo_cost: float = 0.45
    quality_context_patch_size: int = 15
    quality_far_disparity_px: float = 8.0
    quality_disparity_alpha: float = 0.2
    quality_max_smoothing_innovation_px: float = 0.75
    quality_motion_agreement_px: float = 0.75
    photo_weight: float = 0.55
    prediction_weight: float = 0.20
    epipolar_weight: float = 0.10
    neighborhood_weight: float = 0.15
    uniqueness_margin: float = 0.05
    flow_fb_threshold: float = 1.0
    lk_window_size: int = 21
    lk_max_level: int = 3
    recovery_lk_window_size: int = 41
    recovery_lk_max_level: int = 4
    lr_threshold: float = 1.0
    min_depth_m: float = 0.1
    max_depth_m: float = 100.0
    max_disparity_velocity: float = 3.0
    max_failures: int = 3
    recovery_interval_frames: int = 5
    recovery_search_radius: int = 32
    adaptive_search_small_radius: int = 4
    adaptive_search_normal_radius: int = 8
    adaptive_search_large_radius: int = 16
    adaptive_search_recovery_radius: int = 32
    confidence_high_threshold: float = 0.75
    confidence_medium_threshold: float = 0.45
    confidence_low_threshold: float = 0.25
    confidence_low_to_lost_frames: int = 3
    enable_initial_confidence_calibration: bool = False
    camera_compensation_mode: str = "multi_reference_rigid"
    confidence_mode: str = "closed_loop"
    uncertainty_neighbor_mad_reference: float = 1.0
    uncertainty_weight_neighbor: float = 0.8
    enable_recovery: bool = True
    enable_pyramid: bool = True
    pyramid_levels: int = 3
    pyramid_search_radius: int = 12
    pyramid_refine_radius: int = 3
    subpixel_method: str = "continuous"
    subpixel_step: float = 0.1
    enable_temporal_estimation: bool = True
    max_motion_m_per_frame: float | None = None
    num_disparities: int = 64
    expanded_num_disparities: int = 128
    block_size: int = 3
    uniqueness_ratio: int = 15
    speckle_window_size: int = 150
    speckle_range: int = 2
    disp12_max_diff: int = 1
    enable_prediction: bool = True
    enable_flow: bool = True
    enable_epipolar: bool = True
    enable_neighborhood: bool = True
    enable_subpixel: bool = True
    enable_lr_check: bool = True
    enable_cycle_consistency: bool = True
    cycle_weight: float = 0.20
    cycle_soft_threshold_px: float = 0.50
    cycle_hard_threshold_px: float = 1.50
    cycle_recovery_threshold_px: float = 3.00
    right_flow_fb_threshold: float = 1.00
    cycle_prediction_blend: float = 0.50
    enable_icgn: bool = True
    research_subpixel_method: str = "icgn"
    icgn_patch_size: int = 15
    icgn_max_iterations: int = 20
    icgn_epsilon: float = 1e-3
    icgn_max_offset_px: float = 1.5
    icgn_min_hessian: float = 1e-6
    icgn_max_residual: float = 0.35
    icgn_fallback_method: str = "line_search"
    icgn_fallback_search_radius_px: float = 0.6
    icgn_fallback_sample_count: int = 13
    curvature_sample_step_px: float = 0.10
    curvature_min_value: float = 1e-6
    curvature_max_value: float = 1e4
    uncertainty_texture_reference: float = 10.0
    uncertainty_margin_reference: float = 0.10
    uncertainty_max_component: float = 3.0
    uncertainty_base_disparity_variance_px2: float = 0.01
    uncertainty_min_disparity_variance_px2: float = 0.0025
    uncertainty_max_disparity_variance_px2: float = 4.0
    uncertainty_base_left_variance_px2: float = 0.01
    uncertainty_min_left_variance_px2: float = 0.0025
    uncertainty_max_left_variance_px2: float = 4.0
    uncertainty_weight_texture: float = 0.8
    uncertainty_weight_photo: float = 1.0
    uncertainty_weight_margin: float = 1.0
    uncertainty_weight_flow: float = 1.0
    uncertainty_weight_lr: float = 1.0
    uncertainty_weight_cycle: float = 1.2
    uncertainty_weight_icgn: float = 0.8
    uncertainty_weight_icgn_hessian: float = 0.4
    uncertainty_weight_curvature: float = 0.4
    uncertainty_curvature_reference: float = 0.1
    enable_adaptive_filter: bool = True
    kalman_process_position_variance: float = 0.01
    kalman_process_velocity_variance: float = 0.0025
    kalman_initial_position_variance: float = 1.0
    kalman_initial_velocity_variance: float = 1.0
    kalman_nis_soft_threshold: float = 7.815
    kalman_nis_hard_threshold: float = 16.266
    kalman_nis_variance_scale: float = 4.0
    kalman_max_predict_only_frames: int = 3
    enable_camera_compensation: bool = True
    camera_compensation_min_points: int = 3
    camera_compensation_inlier_threshold_mm: float = 3.0
    camera_compensation_ransac_iterations: int = 100
    camera_compensation_min_inlier_ratio: float = 0.60
    camera_compensation_max_rmse_mm: float = 3.0
    enable_physics_shadow: bool = True
    enable_fault_shadow: bool = True
    allow_physics_correction_to_final: bool = False
    allow_fault_recovery_to_final: bool = False
    enable_target_accuracy_policy: bool = False
    target_metric: str = "sigma"
    target_sigma_x_mm: float | None = None
    target_sigma_y_mm: float | None = None
    target_sigma_z_mm: float | None = None
    max_precision_retry: int = 1
    max_precision_refinement_level: int = 2

    def __post_init__(self) -> None:
        if self.allow_physics_correction_to_final or self.allow_fault_recovery_to_final:
            raise NotImplementedError("first-round Shadow Mode cannot modify final output")
        if self.target_metric != "sigma":
            raise ValueError("target metric must be sigma")
        target_sigmas = (
            self.target_sigma_x_mm,
            self.target_sigma_y_mm,
            self.target_sigma_z_mm,
        )
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in target_sigmas):
            raise ValueError("target sigma values must be positive and finite")
        if self.enable_target_accuracy_policy and all(value is None for value in target_sigmas):
            raise ValueError("enabled target-accuracy policy requires a target sigma")
        if self.max_precision_retry < 0:
            raise ValueError("max precision retry must be non-negative")
        if self.max_precision_refinement_level < 0:
            raise ValueError("max precision refinement level must be non-negative")
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer of at least 3")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer of at least 3")
        if self.search_radius < 1 or self.expanded_search_radius < self.search_radius:
            raise ValueError("expanded_search_radius must be at least search_radius")
        radii = (
            self.adaptive_search_small_radius, self.adaptive_search_normal_radius,
            self.adaptive_search_large_radius, self.adaptive_search_recovery_radius,
        )
        if any(radius < 1 for radius in radii) or tuple(sorted(radii)) != radii:
            raise ValueError("adaptive search radii must be positive and nondecreasing")
        if not 0 <= self.confidence_low_threshold < self.confidence_medium_threshold < self.confidence_high_threshold <= 1:
            raise ValueError("confidence thresholds must be ordered in [0, 1]")
        if self.confidence_low_to_lost_frames < 1:
            raise ValueError("confidence_low_to_lost_frames must be positive")
        if self.camera_compensation_mode not in {"none", "single_reference", "multi_reference_rigid"}:
            raise ValueError("camera_compensation_mode is invalid")
        if self.confidence_mode not in {"none", "lr_only", "single_margin", "multi_reject", "closed_loop"}:
            raise ValueError("confidence_mode is invalid")
        if self.uncertainty_neighbor_mad_reference <= 0 or self.uncertainty_weight_neighbor < 0:
            raise ValueError("neighbor uncertainty settings are invalid")
        if self.vertical_radius < 0:
            raise ValueError("vertical_radius must be non-negative")
        if (
            self.lk_window_size < 3
            or self.lk_window_size % 2 == 0
            or self.recovery_lk_window_size < self.lk_window_size
            or self.recovery_lk_window_size % 2 == 0
        ):
            raise ValueError("LK windows must be odd and recovery window must not be smaller")
        if self.lk_max_level < 0 or self.recovery_lk_max_level < self.lk_max_level:
            raise ValueError("recovery LK level must cover the normal LK level")
        if self.max_failures < 1:
            raise ValueError("max_failures must be positive")
        if self.recovery_interval_frames < 1:
            raise ValueError("recovery_interval_frames must be positive")
        if self.recovery_search_radius < self.expanded_search_radius:
            raise ValueError("recovery_search_radius must cover expanded_search_radius")
        if self.pyramid_levels < 1 or self.pyramid_search_radius < 1 or self.pyramid_refine_radius < 1:
            raise ValueError("pyramid levels and radii must be positive")
        if self.subpixel_method not in {"parabolic", "continuous", "icgn"}:
            raise ValueError("subpixel_method must be 'parabolic', 'continuous', or 'icgn'")
        if not 0.0 < self.subpixel_step <= 1.0:
            raise ValueError("subpixel_step must be in (0, 1]")
        if self.min_depth_m <= 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("depth range must be positive and increasing")
        if not 0.0 < self.max_photo_cost <= 1.0:
            raise ValueError("max_photo_cost must be in (0, 1]")
        if self.quality_context_patch_size < 3 or self.quality_context_patch_size % 2 == 0:
            raise ValueError("quality_context_patch_size must be an odd integer of at least 3")
        if self.quality_far_disparity_px <= 0:
            raise ValueError("quality_far_disparity_px must be positive")
        if not 0.0 < self.quality_disparity_alpha <= 1.0:
            raise ValueError("quality_disparity_alpha must be in (0, 1]")
        if self.quality_max_smoothing_innovation_px <= 0 or self.quality_motion_agreement_px <= 0:
            raise ValueError("quality innovation and agreement thresholds must be positive")
        weights = (
            self.photo_weight,
            self.prediction_weight,
            self.epipolar_weight,
            self.neighborhood_weight,
        )
        if any(weight < 0 for weight in weights) or self.photo_weight <= 0:
            raise ValueError("constraint weights must be non-negative and photo_weight must be positive")
        if self.max_motion_m_per_frame is not None and self.max_motion_m_per_frame <= 0:
            raise ValueError("max_motion_m_per_frame must be positive when configured")
        if not (
            0.0 < self.cycle_soft_threshold_px < self.cycle_hard_threshold_px
            <= self.cycle_recovery_threshold_px
        ):
            raise ValueError("cycle thresholds must be positive and strictly ordered")
        if not 0.0 <= self.cycle_prediction_blend <= 1.0:
            raise ValueError("cycle_prediction_blend must be in [0, 1]")
        if self.cycle_weight < 0 or self.right_flow_fb_threshold <= 0:
            raise ValueError("cycle_weight must be non-negative and right flow threshold positive")
        if self.research_subpixel_method not in {"continuous", "parabolic", "icgn"}:
            raise ValueError("research_subpixel_method is invalid")
        if self.icgn_patch_size < 3 or self.icgn_patch_size % 2 == 0:
            raise ValueError("icgn_patch_size must be an odd integer of at least 3")
        if self.icgn_max_iterations <= 0:
            raise ValueError("icgn_max_iterations must be positive")
        if self.icgn_epsilon <= 0 or self.icgn_max_offset_px <= 0:
            raise ValueError("IC-GN epsilon and maximum offset must be positive")
        if self.icgn_min_hessian <= 0 or self.icgn_max_residual <= 0:
            raise ValueError("IC-GN Hessian and residual limits must be positive")
        if self.icgn_fallback_method not in {"line_search", "continuous", "parabolic", "integer"}:
            raise ValueError("icgn fallback must be line_search, continuous, parabolic, or integer")
        if self.icgn_fallback_search_radius_px <= 0:
            raise ValueError("IC-GN fallback search radius must be positive")
        if self.icgn_fallback_sample_count < 5 or self.icgn_fallback_sample_count % 2 == 0:
            raise ValueError("IC-GN fallback sample count must be odd and at least 5")
        if (
            self.curvature_sample_step_px <= 0
            or self.curvature_min_value <= 0
            or self.curvature_max_value < self.curvature_min_value
        ):
            raise ValueError("curvature controls must be positive and ordered")
        positive_uncertainty = (
            self.uncertainty_texture_reference,
            self.uncertainty_margin_reference,
            self.uncertainty_max_component,
            self.uncertainty_base_disparity_variance_px2,
            self.uncertainty_min_disparity_variance_px2,
            self.uncertainty_max_disparity_variance_px2,
            self.uncertainty_base_left_variance_px2,
            self.uncertainty_min_left_variance_px2,
            self.uncertainty_max_left_variance_px2,
            self.uncertainty_curvature_reference,
        )
        if any(value <= 0 for value in positive_uncertainty):
            raise ValueError("uncertainty references and variance bounds must be positive")
        if not (
            self.uncertainty_min_disparity_variance_px2
            <= self.uncertainty_base_disparity_variance_px2
            <= self.uncertainty_max_disparity_variance_px2
            and self.uncertainty_min_left_variance_px2
            <= self.uncertainty_base_left_variance_px2
            <= self.uncertainty_max_left_variance_px2
        ):
            raise ValueError("uncertainty variance bounds must contain their base values")
        uncertainty_weights = (
            self.uncertainty_weight_texture,
            self.uncertainty_weight_photo,
            self.uncertainty_weight_margin,
            self.uncertainty_weight_flow,
            self.uncertainty_weight_lr,
            self.uncertainty_weight_cycle,
            self.uncertainty_weight_icgn,
            self.uncertainty_weight_icgn_hessian,
            self.uncertainty_weight_curvature,
        )
        if any(weight < 0 for weight in uncertainty_weights):
            raise ValueError("uncertainty weights must be non-negative")
        if (
            self.kalman_process_position_variance <= 0
            or self.kalman_process_velocity_variance <= 0
            or self.kalman_initial_position_variance <= 0
            or self.kalman_initial_velocity_variance <= 0
        ):
            raise ValueError("Kalman variances must be positive")
        if not 0 < self.kalman_nis_soft_threshold < self.kalman_nis_hard_threshold:
            raise ValueError("Kalman NIS thresholds must be positive and ordered")
        if self.kalman_nis_variance_scale <= 1 or self.kalman_max_predict_only_frames < 1:
            raise ValueError("Kalman recovery settings are invalid")
        if self.camera_compensation_min_points < 3:
            raise ValueError("camera_compensation_min_points must be at least 3")
        if (
            self.camera_compensation_inlier_threshold_mm <= 0
            or self.camera_compensation_ransac_iterations < 1
            or not 0 < self.camera_compensation_min_inlier_ratio <= 1
            or self.camera_compensation_max_rmse_mm <= 0
        ):
            raise ValueError("camera compensation settings are invalid")


@dataclass(frozen=True)
class ResolvedSystemMode:
    """Validated system intent without changing matching or measurement algorithms."""

    mode: SystemMode
    matcher_config: MatcherConfig
    enable_unified_innovation1: bool
    innovation2_mode: Innovation2Mode
    innovation3_mode: Innovation3Mode
    final_owner: str = FINAL_RESULT_OWNER


def resolve_system_mode(
    mode: SystemMode | str,
    *,
    matcher_config: MatcherConfig | None = None,
) -> ResolvedSystemMode:
    """Resolve system modes deterministically and reject unavailable authority.

    ``BASELINE`` is deliberately the frozen GUI production configuration, not a
    request to disable all innovation-related observability flags.
    """

    try:
        resolved_mode = mode if isinstance(mode, SystemMode) else SystemMode(mode)
    except ValueError as exc:
        raise ValueError(f"unknown system mode: {mode!r}") from exc

    if resolved_mode is SystemMode.FULL_ENHANCED:
        raise PermissionError(
            "FULL_ENHANCED is unauthorized until controlled correction and recovery are validated"
        )

    resolved_config = matcher_config if matcher_config is not None else MatcherConfig()
    frozen_baseline = MatcherConfig()
    if resolved_mode is SystemMode.BASELINE and resolved_config != frozen_baseline:
        raise ValueError(
            "BASELINE requires the Phase 0 frozen MatcherConfig() without legacy flag overrides"
        )

    if resolved_mode in {SystemMode.INNOVATION1, SystemMode.ENHANCED_SHADOW}:
        if not resolved_config.enable_physics_shadow or not resolved_config.enable_fault_shadow:
            raise ValueError("system modes with Innovation 1 preserve frozen Shadow and Diagnose flags")
        return ResolvedSystemMode(
            mode=resolved_mode,
            matcher_config=resolved_config,
            enable_unified_innovation1=True,
            innovation2_mode=Innovation2Mode.SHADOW,
            innovation3_mode=Innovation3Mode.DIAGNOSE,
        )

    return ResolvedSystemMode(
        mode=SystemMode.BASELINE,
        matcher_config=frozen_baseline,
        enable_unified_innovation1=False,
        innovation2_mode=Innovation2Mode.SHADOW,
        innovation3_mode=Innovation3Mode.DIAGNOSE,
    )


@dataclass(frozen=True)
class PointSpec:
    point_id: str
    xy: tuple[float, float]
    role: Literal["measurement", "reference"] = "measurement"


@dataclass(frozen=True)
class SequenceManifest:
    name: str
    video: Path
    start_frame: int
    end_frame: int | None
    calibration: str
    points_path: Path
    point_specs: tuple[PointSpec, ...]
    ground_truth: Path | None = None
    stereo_ground_truth: Path | None = None
    external_ground_truth: Path | None = None
    timestamps: Path | None = None
    output_dir: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "SequenceManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent
        start_frame = int(payload.get("start_frame", 0))
        end_value = payload.get("end_frame")
        end_frame = None if end_value is None else int(end_value)
        if start_frame < 0 or (end_frame is not None and end_frame < start_frame):
            raise ValueError("Manifest frame range is invalid")
        points_path = _resolve_path(base, payload["points"])
        point_payload = json.loads(points_path.read_text(encoding="utf-8"))
        point_frame = int(point_payload.get("frame", start_frame))
        if point_frame != start_frame:
            raise ValueError(
                f"Point annotation frame {point_frame} does not match manifest start frame {start_frame}"
            )
        specs: list[PointSpec] = []
        seen: set[str] = set()
        for item in point_payload.get("points", []):
            point_id = str(item["id"])
            if point_id in seen:
                raise ValueError(f"Duplicate point id: {point_id}")
            seen.add(point_id)
            role = str(item.get("role", "measurement"))
            if role not in {"measurement", "reference"}:
                raise ValueError(f"Point {point_id!r} has invalid role {role!r}")
            specs.append(
                PointSpec(
                    point_id=point_id,
                    xy=(float(item["x"]), float(item["y"])),
                    role=cast(Literal["measurement", "reference"], role),
                )
            )
        if not specs:
            raise ValueError("Point annotation must contain at least one point")
        ground_truth = payload.get("ground_truth")
        stereo_ground_truth = payload.get("stereo_ground_truth")
        external_ground_truth = payload.get("external_ground_truth")
        if ground_truth and not stereo_ground_truth and not external_ground_truth:
            candidate = _resolve_path(base, ground_truth)
            header = candidate.read_text(encoding="utf-8-sig").splitlines()[0] if candidate.exists() else ""
            if "timestamp_s" in header:
                external_ground_truth = ground_truth
            else:
                stereo_ground_truth = ground_truth
        timestamps = payload.get("timestamps")
        output_dir = payload.get("output_dir")
        return cls(
            name=str(payload["name"]),
            video=_resolve_path(base, payload["video"]),
            start_frame=start_frame,
            end_frame=end_frame,
            calibration=str(payload.get("calibration", "builtin_640x480")),
            points_path=points_path,
            point_specs=tuple(specs),
            ground_truth=None if ground_truth is None else _resolve_path(base, ground_truth),
            stereo_ground_truth=None if stereo_ground_truth is None else _resolve_path(base, stereo_ground_truth),
            external_ground_truth=None if external_ground_truth is None else _resolve_path(base, external_ground_truth),
            timestamps=None if timestamps is None else _resolve_path(base, timestamps),
            output_dir=None if output_dir is None else _resolve_path(base, output_dir),
        )


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass
class PointState:
    point_id: str
    initial_left_xy: tuple[float, float]
    left_xy: tuple[float, float]
    previous_left_xy: tuple[float, float] | None = None
    right_xy: tuple[float, float] | None = None
    previous_right_xy: tuple[float, float] | None = None
    right_flow_reference_xy: tuple[float, float] | None = None
    right_flow_reference_frame: int | None = None
    point_role: Literal["measurement", "reference"] = "measurement"
    disparity_history: list[float] = field(default_factory=list)
    estimated_disparity_history: list[float] = field(default_factory=list)
    latest_disparity: float | None = None
    last_accepted_measured_right_xy: tuple[float, float] | None = None
    last_accepted_measured_disparity: float | None = None
    last_estimated_left_xy: tuple[float, float] | None = None
    last_estimated_right_xy: tuple[float, float] | None = None
    last_estimated_disparity: float | None = None
    xyz: tuple[float, float, float] | None = None
    estimated_xyz: tuple[float, float, float] | None = None
    reference_xyz: tuple[float, float, float] | None = None
    reference_measured_xyz: tuple[float, float, float] | None = None
    reference_estimated_xyz: tuple[float, float, float] | None = None
    reference_compensated_xyz: tuple[float, float, float] | None = None
    compensated_xyz: tuple[float, float, float] | None = None
    confidence: float = 0.0
    status: str = "uninitialized"
    consecutive_failures: int = 0
    recovery_attempt_count: int = 0
    recovery_success_count: int = 0
    recovery_started_frame: int | None = None
    recovery_frame_total: int = 0
    last_cycle_error_px: float | None = None
    last_right_flow_fb_error_px: float | None = None
    last_disparity_variance_px2: float | None = None
    last_left_variance_px2: float | None = None
    last_kalman_innovation: float | None = None
    last_kalman_gain_disparity: float | None = None
    last_nis: float | None = None
    confidence_state: str = "HIGH"
    low_confidence_frames: int = 0
    last_prediction_residual_px: float | None = None

    @property
    def disparity(self) -> float | None:
        return self.disparity_history[-1] if self.disparity_history else None

    @property
    def previous_disparity(self) -> float | None:
        return self.disparity_history[-2] if len(self.disparity_history) >= 2 else None

    @property
    def velocity(self) -> tuple[float, float]:
        if self.previous_left_xy is None:
            return (0.0, 0.0)
        return (
            self.left_xy[0] - self.previous_left_xy[0],
            self.left_xy[1] - self.previous_left_xy[1],
        )

    @property
    def right_velocity(self) -> tuple[float, float]:
        if self.right_xy is None or self.previous_right_xy is None:
            return (0.0, 0.0)
        return (
            self.right_xy[0] - self.previous_right_xy[0],
            self.right_xy[1] - self.previous_right_xy[1],
        )

    def record_success(
        self,
        left_xy: tuple[float, float],
        right_xy: tuple[float, float],
        disparity: float,
        confidence: float,
    ) -> None:
        self.previous_left_xy = self.left_xy
        self.left_xy = left_xy
        self.record_stereo_success(right_xy, disparity, confidence)

    def record_flow(self, left_xy: tuple[float, float]) -> None:
        self.previous_left_xy = self.left_xy
        self.left_xy = left_xy

    def record_stereo_success(
        self,
        right_xy: tuple[float, float],
        disparity: float,
        confidence: float,
    ) -> None:
        self.record_stereo_measurement(right_xy, disparity, confidence)
        self.confirm_valid(confidence)

    def record_stereo_measurement(
        self,
        right_xy: tuple[float, float],
        disparity: float,
        confidence: float,
        estimated_disparity: float | None = None,
        xyz: tuple[float, float, float] | None = None,
        estimated_xyz: tuple[float, float, float] | None = None,
    ) -> None:
        self.previous_right_xy = self.right_xy
        self.right_xy = right_xy
        self.disparity_history.append(float(disparity))
        if len(self.disparity_history) > 2:
            self.disparity_history = self.disparity_history[-2:]
        estimate = float(disparity if estimated_disparity is None else estimated_disparity)
        self.estimated_disparity_history.append(estimate)
        if len(self.estimated_disparity_history) > 2:
            self.estimated_disparity_history = self.estimated_disparity_history[-2:]
        self.xyz = xyz
        self.estimated_xyz = estimated_xyz if estimated_xyz is not None else xyz
        self.latest_disparity = float(disparity)
        self.last_accepted_measured_right_xy = (
            float(right_xy[0]),
            float(right_xy[1]),
        )
        self.last_accepted_measured_disparity = float(disparity)
        self.last_estimated_left_xy = self.left_xy
        self.last_estimated_right_xy = (
            float(self.left_xy[0] - estimate),
            float(right_xy[1]),
        )
        self.last_estimated_disparity = estimate
        if self.reference_xyz is None and xyz is not None:
            self.reference_xyz = xyz
        if self.reference_measured_xyz is None and xyz is not None:
            self.reference_measured_xyz = xyz
        if self.reference_estimated_xyz is None and self.estimated_xyz is not None:
            self.reference_estimated_xyz = self.estimated_xyz
        self.confidence = float(confidence)

    @property
    def measured_disparity_history(self) -> list[float]:
        """Accepted raw measurements; rejected measurements are never appended."""
        return self.disparity_history

    def record_estimated_only(
        self,
        estimated_left_xy: tuple[float, float],
        estimated_right_xy: tuple[float, float],
        estimated_disparity: float,
        estimated_xyz_m: tuple[float, float, float],
        frame: int,
        quality_stage: str,
    ) -> None:
        """Advance tracking from a filter prediction without accepting raw measurement data."""
        del frame, quality_stage  # retained in the API for traceability at the call site
        self.left_xy = (float(estimated_left_xy[0]), float(estimated_left_xy[1]))
        self.previous_right_xy = self.right_xy
        self.right_xy = (float(estimated_right_xy[0]), float(estimated_right_xy[1]))
        estimate = float(estimated_disparity)
        self.estimated_disparity_history.append(estimate)
        if len(self.estimated_disparity_history) > 2:
            self.estimated_disparity_history = self.estimated_disparity_history[-2:]
        self.latest_disparity = estimate
        self.last_estimated_left_xy = self.left_xy
        self.last_estimated_right_xy = self.right_xy
        self.last_estimated_disparity = estimate
        self.estimated_xyz = tuple(float(value) for value in estimated_xyz_m)
        if self.reference_estimated_xyz is None:
            self.reference_estimated_xyz = self.estimated_xyz

    def set_right_flow_reference(
        self,
        frame: int,
        xy: tuple[float, float],
    ) -> None:
        self.right_flow_reference_frame = int(frame)
        self.right_flow_reference_xy = (float(xy[0]), float(xy[1]))

    def confirm_valid(self, confidence: float) -> None:
        self.confidence = float(confidence)
        self.status = "tracking"
        self.consecutive_failures = 0

    def confirm_recovery(self, frame: int) -> None:
        self.recovery_success_count += 1
        if self.recovery_started_frame is not None:
            self.recovery_frame_total += max(1, frame - self.recovery_started_frame)
        self.recovery_started_frame = None
        self.previous_left_xy = None

    def record_failure(self, status: str, max_failures: int) -> None:
        self.consecutive_failures += 1
        self.status = "lost" if self.consecutive_failures >= max_failures else "recovering"


@dataclass(frozen=True)
class FramePointResult:
    method: MethodName
    frame: int
    point_id: str
    status: str
    left_x: float | None = None
    left_y: float | None = None
    right_x: float | None = None
    right_y: float | None = None
    disparity: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    distance_m: float | None = None
    match_cost: float | None = None
    flow_fb_error_px: float | None = None
    lr_error_px: float | None = None
    confidence: float = 0.0
    raw_disparity: float | None = None
    integer_disparity: float | None = None
    measured_disparity: float | None = None
    estimated_disparity: float | None = None
    measured_right_x: float | None = None
    measured_right_y: float | None = None
    estimated_right_x: float | None = None
    estimated_right_y: float | None = None
    measured_x_m: float | None = None
    measured_y_m: float | None = None
    measured_z_m: float | None = None
    estimated_x_m: float | None = None
    estimated_y_m: float | None = None
    estimated_z_m: float | None = None
    delta_x_mm: float | None = None
    delta_y_mm: float | None = None
    delta_z_mm: float | None = None
    subpixel_offset: float | None = None
    neighbor_disparity: float | None = None
    neighbor_disparity_mad: float | None = None
    used_search_radius: int = 0
    quality_stage: str = ""
    recovery_stage: str = ""
    recovery_attempt_count: int = 0
    recovery_success_count: int = 0
    mean_recovery_frames: float | None = None
    flow_ms: float = 0.0
    matching_ms: float = 0.0
    total_ms: float = 0.0
    temporal_right_x: float | None = None
    temporal_right_y: float | None = None
    right_flow_fb_error_px: float | None = None
    cycle_error_px: float | None = None
    cycle_cost: float | None = None
    cycle_status: str = "not_used"
    texture_std: float | None = None
    second_best_cost: float | None = None
    uniqueness_margin_value: float | None = None
    cost_curvature: float | None = None
    icgn_status: str = "not_attempted"
    icgn_converged: bool = False
    icgn_iterations: int = 0
    icgn_residual: float | None = None
    icgn_hessian: float | None = None
    icgn_cost_curvature: float | None = None
    icgn_hessian_density: float | None = None
    zncc_cost_curvature: float | None = None
    curvature_sample_step_px: float | None = None
    icgn_termination_reason: str = ""
    icgn_fallback_used: bool = False
    icgn_fallback_method: str = "none"
    icgn_iterative_disparity: float | None = None
    icgn_final_increment_px: float | None = None
    subpixel_final_status: str = "not_attempted"
    left_variance_px2: float | None = None
    disparity_variance_px2: float | None = None
    measurement_quality_score: float | None = None
    kalman_innovation_u: float | None = None
    kalman_innovation_v: float | None = None
    kalman_innovation_d: float | None = None
    kalman_innovation_norm: float | None = None
    kalman_gain_disparity: float | None = None
    kalman_nis: float | None = None
    kalman_update_status: str = "not_used"
    kalman_predict_only_frames: int = 0
    measurement_accepted_for_state: bool = True
    state_update_source: str = "measured"
    right_flow_reference_frame: int | None = None
    right_reference_source: str = "not_enabled"
    right_reference_age_frames: int | None = None
    point_role: str = "measurement"
    compensation_status: str = "not_used"
    compensation_applied: bool = False
    compensation_inlier_count: int = 0
    compensation_reference_count: int = 0
    compensation_rmse_mm: float | None = None
    camera_tx_mm: float | None = None
    camera_ty_mm: float | None = None
    camera_tz_mm: float | None = None
    camera_rotation_angle_deg: float | None = None
    camera_rx_deg: float | None = None
    camera_ry_deg: float | None = None
    camera_rz_deg: float | None = None
    compensated_x_m: float | None = None
    compensated_y_m: float | None = None
    compensated_z_m: float | None = None
    final_x_m: float | None = None
    final_y_m: float | None = None
    final_z_m: float | None = None
    system_mode: str = SystemMode.BASELINE.value
    final_owner: str = FINAL_RESULT_OWNER
    raw_delta_x_mm: float | None = None
    raw_delta_y_mm: float | None = None
    raw_delta_z_mm: float | None = None
    compensated_delta_x_mm: float | None = None
    compensated_delta_y_mm: float | None = None
    compensated_delta_z_mm: float | None = None
    predicted_disparity: float | None = None
    previous_disparity: float | None = None
    disparity_velocity: float | None = None
    prediction_residual_px: float | None = None
    adaptive_search_radius: int | None = None
    search_radius_reason: str = ""
    confidence_texture: float | None = None
    confidence_photo: float | None = None
    confidence_margin: float | None = None
    confidence_flow: float | None = None
    confidence_lr: float | None = None
    confidence_cycle: float | None = None
    confidence_icgn: float | None = None
    confidence_curvature: float | None = None
    confidence_temporal: float | None = None
    confidence_neighbor: float | None = None
    confidence_total: float | None = None
    confidence_state: str = "HIGH"
    confidence_source: str = "legacy_or_temporal"
    candidate_count: int | None = None
    initial_best_disparity: float | None = None
    initial_disparity_disagreement: float | None = None
    shadow_input_x_m: float | None = None
    shadow_input_y_m: float | None = None
    shadow_input_z_m: float | None = None
    shadow_input_stage: str = ""
    candidate_corrected_x_m: float | None = None
    candidate_corrected_y_m: float | None = None
    candidate_corrected_z_m: float | None = None
    c_phy: float | None = None
    c_phy_valid: bool | None = None
    c_phy_valid_terms: str = ""
    c_phy_missing_terms: str = ""
    transient_protected: bool | None = None
    fault_class: str = ""
    fault_confidence: float | None = None
    recommended_recovery: str = ""
    r_2d3d: float | None = None
    r_temporal: float | None = None
    r_spatial: float | None = None
    r_frequency: float | None = None
    r_phase: float | None = None
    r_coherence: float | None = None
    r_lr: float | None = None
    r_epi: float | None = None
    r_fb: float | None = None
    r_ref: float | None = None
    r_calib: float | None = None
    accuracy_policy_enabled: bool = False
    precision_policy_warmup: bool | None = None
    target_metric: str = ""
    target_x_mm: float | None = None
    target_y_mm: float | None = None
    target_z_mm: float | None = None
    required_sigma_d_px: float | None = None
    estimated_sigma_d_px: float | None = None
    estimated_sigma_x_mm: float | None = None
    estimated_sigma_y_mm: float | None = None
    estimated_sigma_z_mm: float | None = None
    precision_ratio: float | None = None
    precision_feasible: bool | None = None
    precision_status: str = ""
    limiting_axis: str = ""
    policy_base_search_radius_px: int | None = None
    policy_final_search_radius_px: int | None = None
    policy_refinement_level: int | None = None
    policy_retry_budget: int | None = None
    policy_precision_retry_count: int | None = None
    policy_acceptance_reason: str = ""
    policy_reason: str = ""
    retry_triggered: bool | None = None
    retry_candidate_accepted: bool | None = None
    retry_sigma_before_px: float | None = None
    retry_sigma_after_px: float | None = None
    retry_match_cost_before: float | None = None
    retry_match_cost_after: float | None = None
    retry_lr_error_before_px: float | None = None
    retry_lr_error_after_px: float | None = None
    # Phase 4 experiment-only diagnostics.  They are deliberately omitted from
    # ``as_csv_row`` to preserve the legacy production CSV contract.
    final_distance_m: float | None = None
    final_valid: bool | None = None
    proposed_decision: str = ""
    committed_decision: str = ""
    write_committed: bool | None = None
    result_source: str = ""
    final_decision_reason: str = ""
    i2_state: str = ""
    i2_episode_id: int | None = None
    i2_prediction_x_m: float | None = None
    i2_prediction_y_m: float | None = None
    i2_prediction_z_m: float | None = None
    candidate_safe: bool | None = None
    candidate_safety_reasons: str = ""

    def __post_init__(self) -> None:
        if self.final_owner != FINAL_RESULT_OWNER:
            raise ValueError(f"FINAL owner must be {FINAL_RESULT_OWNER}")

    @staticmethod
    def _xyz_or_none(
        x_m: float | None, y_m: float | None, z_m: float | None,
    ) -> tuple[float, float, float] | None:
        if x_m is None or y_m is None or z_m is None:
            return None
        return (x_m, y_m, z_m)

    @property
    def triangulated_xyz_m(self) -> tuple[float, float, float] | None:
        """Direct accepted-match reprojection, named separately from filtered output."""

        return self._xyz_or_none(self.measured_x_m, self.measured_y_m, self.measured_z_m)

    @property
    def pipeline_baseline_xyz_m(self) -> tuple[float, float, float] | None:
        """Existing Pipeline output used as the only Innovation 2 comparison input."""

        compensated = self._xyz_or_none(
            self.compensated_x_m, self.compensated_y_m, self.compensated_z_m,
        )
        if compensated is not None:
            return compensated
        return self._xyz_or_none(self.estimated_x_m, self.estimated_y_m, self.estimated_z_m)

    @property
    def pipeline_baseline_source(self) -> str:
        if self._xyz_or_none(self.compensated_x_m, self.compensated_y_m, self.compensated_z_m) is not None:
            return "COMPENSATED"
        if self._xyz_or_none(self.estimated_x_m, self.estimated_y_m, self.estimated_z_m) is not None:
            return "ESTIMATED"
        return "UNAVAILABLE"

    @property
    def candidate_xyz_m(self) -> tuple[float, float, float] | None:
        return self._xyz_or_none(
            self.candidate_corrected_x_m,
            self.candidate_corrected_y_m,
            self.candidate_corrected_z_m,
        )

    @property
    def final_xyz_m(self) -> tuple[float, float, float] | None:
        return self._xyz_or_none(self.final_x_m, self.final_y_m, self.final_z_m)

    @classmethod
    def invalid(
        cls,
        method: MethodName,
        frame: int,
        point_state: PointState,
        status: str,
        flow_ms: float = 0.0,
        matching_ms: float = 0.0,
        total_ms: float = 0.0,
        flow_fb_error_px: float | None = None,
        recovery_stage: str = "",
        temporal_right_xy: tuple[float, float] | None = None,
        right_flow_fb_error_px: float | None = None,
        cycle_error_px: float | None = None,
        cycle_cost: float | None = None,
        cycle_status: str = "not_used",
    ) -> "FramePointResult":
        return cls(
            method=method,
            frame=frame,
            point_id=point_state.point_id,
            status=status,
            left_x=point_state.left_xy[0],
            left_y=point_state.left_xy[1],
            flow_fb_error_px=flow_fb_error_px,
            flow_ms=flow_ms,
            matching_ms=matching_ms,
            total_ms=total_ms,
            recovery_stage=recovery_stage,
            recovery_attempt_count=point_state.recovery_attempt_count,
            recovery_success_count=point_state.recovery_success_count,
            mean_recovery_frames=(
                point_state.recovery_frame_total / point_state.recovery_success_count
                if point_state.recovery_success_count
                else None
            ),
            point_role=point_state.point_role,
            temporal_right_x=(temporal_right_xy[0] if temporal_right_xy else None),
            temporal_right_y=(temporal_right_xy[1] if temporal_right_xy else None),
            right_flow_fb_error_px=right_flow_fb_error_px,
            cycle_error_px=cycle_error_px,
            cycle_cost=cycle_cost,
            cycle_status=cycle_status,
        )

    def as_csv_row(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "method": self.method,
            "frame": self.frame,
            "point_id": self.point_id,
            "status": self.status,
            "left_x": self.left_x,
            "left_y": self.left_y,
            "right_x": self.right_x,
            "right_y": self.right_y,
            "disparity": self.disparity,
            "X_m": self.x_m,
            "Y_m": self.y_m,
            "Z_m": self.z_m,
            "distance_m": self.distance_m,
            "match_cost": self.match_cost,
            "flow_fb_error_px": self.flow_fb_error_px,
            "lr_error_px": self.lr_error_px,
            "confidence": self.confidence,
            "raw_disparity": self.raw_disparity,
            "integer_disparity": self.integer_disparity,
            "measured_disparity": self.measured_disparity,
            "estimated_disparity": self.estimated_disparity,
            "measured_right_x": self.measured_right_x,
            "measured_right_y": self.measured_right_y,
            "estimated_right_x": self.estimated_right_x,
            "estimated_right_y": self.estimated_right_y,
            "measured_X_m": self.measured_x_m,
            "measured_Y_m": self.measured_y_m,
            "measured_Z_m": self.measured_z_m,
            "estimated_X_m": self.estimated_x_m,
            "estimated_Y_m": self.estimated_y_m,
            "estimated_Z_m": self.estimated_z_m,
            "delta_X_mm": self.delta_x_mm,
            "delta_Y_mm": self.delta_y_mm,
            "delta_Z_mm": self.delta_z_mm,
            "subpixel_offset": self.subpixel_offset,
            "neighbor_disparity": self.neighbor_disparity,
            "neighbor_disparity_mad": self.neighbor_disparity_mad,
            "used_search_radius": self.used_search_radius,
            "quality_stage": self.quality_stage,
            "recovery_stage": self.recovery_stage,
            "recovery_attempt_count": self.recovery_attempt_count,
            "recovery_success_count": self.recovery_success_count,
            "mean_recovery_frames": self.mean_recovery_frames,
            "flow_ms": self.flow_ms,
            "matching_ms": self.matching_ms,
            "total_ms": self.total_ms,
            "temporal_right_x": self.temporal_right_x,
            "temporal_right_y": self.temporal_right_y,
            "right_flow_fb_error_px": self.right_flow_fb_error_px,
            "cycle_error_px": self.cycle_error_px,
            "cycle_cost": self.cycle_cost,
            "cycle_status": self.cycle_status,
            "texture_std": self.texture_std,
            "second_best_cost": self.second_best_cost,
            "uniqueness_margin_value": self.uniqueness_margin_value,
            "cost_curvature": self.cost_curvature,
            "icgn_status": self.icgn_status,
            "icgn_converged": self.icgn_converged,
            "icgn_iterations": self.icgn_iterations,
            "icgn_residual": self.icgn_residual,
            "icgn_hessian": self.icgn_hessian,
            "icgn_cost_curvature": self.icgn_cost_curvature,
            "icgn_hessian_density": self.icgn_hessian_density,
            "zncc_cost_curvature": self.zncc_cost_curvature,
            "curvature_sample_step_px": self.curvature_sample_step_px,
            "icgn_termination_reason": self.icgn_termination_reason,
            "icgn_fallback_used": self.icgn_fallback_used,
            "icgn_fallback_method": self.icgn_fallback_method,
            "icgn_iterative_disparity": self.icgn_iterative_disparity,
            "icgn_final_increment_px": self.icgn_final_increment_px,
            "subpixel_final_status": self.subpixel_final_status,
            "left_variance_px2": self.left_variance_px2,
            "disparity_variance_px2": self.disparity_variance_px2,
            "measurement_quality_score": self.measurement_quality_score,
            "kalman_innovation_u": self.kalman_innovation_u,
            "kalman_innovation_v": self.kalman_innovation_v,
            "kalman_innovation_d": self.kalman_innovation_d,
            "kalman_innovation_norm": self.kalman_innovation_norm,
            "kalman_gain_disparity": self.kalman_gain_disparity,
            "kalman_nis": self.kalman_nis,
            "kalman_update_status": self.kalman_update_status,
            "kalman_predict_only_frames": self.kalman_predict_only_frames,
            "measurement_accepted_for_state": self.measurement_accepted_for_state,
            "state_update_source": self.state_update_source,
            "right_flow_reference_frame": self.right_flow_reference_frame,
            "right_reference_source": self.right_reference_source,
            "right_reference_age_frames": self.right_reference_age_frames,
            "point_role": self.point_role,
            "compensation_status": self.compensation_status,
            "compensation_applied": self.compensation_applied,
            "compensation_inlier_count": self.compensation_inlier_count,
            "compensation_reference_count": self.compensation_reference_count,
            "compensation_rmse_mm": self.compensation_rmse_mm,
            "camera_tx_mm": self.camera_tx_mm,
            "camera_ty_mm": self.camera_ty_mm,
            "camera_tz_mm": self.camera_tz_mm,
            "camera_rotation_angle_deg": self.camera_rotation_angle_deg,
            "camera_rx_deg": self.camera_rx_deg,
            "camera_ry_deg": self.camera_ry_deg,
            "camera_rz_deg": self.camera_rz_deg,
            "compensated_X_m": self.compensated_x_m,
            "compensated_Y_m": self.compensated_y_m,
            "compensated_Z_m": self.compensated_z_m,
            "final_X_m": self.final_x_m,
            "final_Y_m": self.final_y_m,
            "final_Z_m": self.final_z_m,
            "raw_delta_X_mm": self.raw_delta_x_mm,
            "raw_delta_Y_mm": self.raw_delta_y_mm,
            "raw_delta_Z_mm": self.raw_delta_z_mm,
            "compensated_delta_X_mm": self.compensated_delta_x_mm,
            "compensated_delta_Y_mm": self.compensated_delta_y_mm,
            "compensated_delta_Z_mm": self.compensated_delta_z_mm,
            "predicted_disparity": self.predicted_disparity,
            "previous_disparity": self.previous_disparity,
            "disparity_velocity": self.disparity_velocity,
            "prediction_residual_px": self.prediction_residual_px,
            "adaptive_search_radius": self.adaptive_search_radius,
            "search_radius_reason": self.search_radius_reason,
            "confidence_texture": self.confidence_texture,
            "confidence_photo": self.confidence_photo,
            "confidence_margin": self.confidence_margin,
            "confidence_flow": self.confidence_flow,
            "confidence_lr": self.confidence_lr,
            "confidence_cycle": self.confidence_cycle,
            "confidence_icgn": self.confidence_icgn,
            "confidence_curvature": self.confidence_curvature,
            "confidence_temporal": self.confidence_temporal,
            "confidence_neighbor": self.confidence_neighbor,
            "confidence_total": self.confidence_total,
            "confidence_state": self.confidence_state,
            "shadow_input_x_m": self.shadow_input_x_m,
            "shadow_input_y_m": self.shadow_input_y_m,
            "shadow_input_z_m": self.shadow_input_z_m,
            "shadow_input_stage": self.shadow_input_stage,
            "candidate_corrected_x_m": self.candidate_corrected_x_m,
            "candidate_corrected_y_m": self.candidate_corrected_y_m,
            "candidate_corrected_z_m": self.candidate_corrected_z_m,
            "c_phy": self.c_phy,
            "c_phy_valid": self.c_phy_valid,
            "c_phy_valid_terms": self.c_phy_valid_terms,
            "c_phy_missing_terms": self.c_phy_missing_terms,
            "transient_protected": self.transient_protected,
            "fault_class": self.fault_class,
            "fault_confidence": self.fault_confidence,
            "recommended_recovery": self.recommended_recovery,
            "r_2d3d": self.r_2d3d,
            "r_temporal": self.r_temporal,
            "r_spatial": self.r_spatial,
            "r_frequency": self.r_frequency,
            "r_phase": self.r_phase,
            "r_coherence": self.r_coherence,
            "r_lr": self.r_lr,
            "r_epi": self.r_epi,
            "r_fb": self.r_fb,
            "r_ref": self.r_ref,
            "r_calib": self.r_calib,
            "accuracy_policy_enabled": self.accuracy_policy_enabled,
            "precision_policy_warmup": self.precision_policy_warmup,
            "target_metric": self.target_metric,
            "target_x_mm": self.target_x_mm,
            "target_y_mm": self.target_y_mm,
            "target_z_mm": self.target_z_mm,
            "required_sigma_d_px": self.required_sigma_d_px,
            "estimated_sigma_d_px": self.estimated_sigma_d_px,
            "estimated_sigma_x_mm": self.estimated_sigma_x_mm,
            "estimated_sigma_y_mm": self.estimated_sigma_y_mm,
            "estimated_sigma_z_mm": self.estimated_sigma_z_mm,
            "precision_ratio": self.precision_ratio,
            "precision_feasible": self.precision_feasible,
            "precision_status": self.precision_status,
            "limiting_axis": self.limiting_axis,
            "policy_base_search_radius_px": self.policy_base_search_radius_px,
            "policy_final_search_radius_px": self.policy_final_search_radius_px,
            "policy_refinement_level": self.policy_refinement_level,
            "policy_retry_budget": self.policy_retry_budget,
            "policy_precision_retry_count": self.policy_precision_retry_count,
            "policy_acceptance_reason": self.policy_acceptance_reason,
            "policy_reason": self.policy_reason,
            "retry_triggered": self.retry_triggered,
            "retry_candidate_accepted": self.retry_candidate_accepted,
            "retry_sigma_before_px": self.retry_sigma_before_px,
            "retry_sigma_after_px": self.retry_sigma_after_px,
            "retry_match_cost_before": self.retry_match_cost_before,
            "retry_match_cost_after": self.retry_match_cost_after,
            "retry_lr_error_before_px": self.retry_lr_error_before_px,
            "retry_lr_error_after_px": self.retry_lr_error_after_px,
        }
        return {
            key: "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else value
            for key, value in values.items()
        }
