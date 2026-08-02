from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast


MethodName = Literal[
    "sgbm",
    "sgbm_fixed",
    "local",
    "local_fixed",
    "local_flow",
    "full",
    "full_quality",
    "sgbm_flow",
]
METHOD_NAMES: tuple[MethodName, ...] = (
    "sgbm",
    "sgbm_fixed",
    "local",
    "local_fixed",
    "local_flow",
    "full",
    "full_quality",
    "sgbm_flow",
)


@dataclass(frozen=True)
class MethodProfile:
    dense_sgbm_each_frame: bool
    use_flow: bool
    use_prediction: bool
    use_epipolar: bool
    use_neighborhood: bool
    use_subpixel: bool
    use_lr_check: bool


_METHOD_PROFILES: dict[MethodName, MethodProfile] = {
    "sgbm": MethodProfile(True, False, False, False, False, False, False),
    "sgbm_fixed": MethodProfile(True, False, False, False, False, False, False),
    "local": MethodProfile(False, False, False, False, False, False, False),
    "local_fixed": MethodProfile(False, False, False, False, False, False, False),
    "local_flow": MethodProfile(False, True, False, False, False, False, False),
    "full": MethodProfile(False, True, True, True, True, True, True),
    "full_quality": MethodProfile(False, True, True, True, True, True, True),
    "sgbm_flow": MethodProfile(True, True, False, False, False, False, False),
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

    def __post_init__(self) -> None:
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer of at least 3")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer of at least 3")
        if self.search_radius < 1 or self.expanded_search_radius < self.search_radius:
            raise ValueError("expanded_search_radius must be at least search_radius")
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
        if self.subpixel_method not in {"parabolic", "continuous"}:
            raise ValueError("subpixel_method must be 'parabolic' or 'continuous'")
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


@dataclass(frozen=True)
class PointSpec:
    point_id: str
    xy: tuple[float, float]


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
            specs.append(PointSpec(point_id=point_id, xy=(float(item["x"]), float(item["y"]))))
        if not specs:
            raise ValueError("Point annotation must contain at least one point")
        ground_truth = payload.get("ground_truth")
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
    disparity_history: list[float] = field(default_factory=list)
    estimated_disparity_history: list[float] = field(default_factory=list)
    xyz: tuple[float, float, float] | None = None
    estimated_xyz: tuple[float, float, float] | None = None
    reference_xyz: tuple[float, float, float] | None = None
    reference_estimated_xyz: tuple[float, float, float] | None = None
    confidence: float = 0.0
    status: str = "uninitialized"
    consecutive_failures: int = 0
    recovery_attempt_count: int = 0
    recovery_success_count: int = 0
    recovery_started_frame: int | None = None
    recovery_frame_total: int = 0

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
        if self.reference_xyz is None and xyz is not None:
            self.reference_xyz = xyz
        if self.reference_estimated_xyz is None and self.estimated_xyz is not None:
            self.reference_estimated_xyz = self.estimated_xyz
        self.confidence = float(confidence)

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
    used_search_radius: int = 0
    quality_stage: str = ""
    recovery_stage: str = ""
    recovery_attempt_count: int = 0
    recovery_success_count: int = 0
    mean_recovery_frames: float | None = None
    flow_ms: float = 0.0
    matching_ms: float = 0.0
    total_ms: float = 0.0

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
            "used_search_radius": self.used_search_radius,
            "quality_stage": self.quality_stage,
            "recovery_stage": self.recovery_stage,
            "recovery_attempt_count": self.recovery_attempt_count,
            "recovery_success_count": self.recovery_success_count,
            "mean_recovery_frames": self.mean_recovery_frames,
            "flow_ms": self.flow_ms,
            "matching_ms": self.matching_ms,
            "total_ms": self.total_ms,
        }
        return {
            key: "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else value
            for key, value in values.items()
        }
