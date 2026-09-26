from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import MatcherConfig, PointState


@dataclass(frozen=True)
class FlowResult:
    status: str
    left_xy: tuple[float, float] | None = None
    fb_error_px: float | None = None
    lk_error: float | None = None


def track_point_lk(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    state: PointState,
    config: MatcherConfig,
    recovery: bool = False,
) -> FlowResult:
    return track_xy_lk(
        previous_gray,
        current_gray,
        previous_xy=state.left_xy,
        initial_velocity=state.velocity,
        config=config,
        recovery=recovery,
    )


def track_xy_lk(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_xy: tuple[float, float],
    initial_velocity: tuple[float, float],
    config: MatcherConfig,
    recovery: bool = False,
    fb_threshold: float | None = None,
) -> FlowResult:
    if previous_gray.ndim != 2 or current_gray.ndim != 2:
        raise ValueError("LK tracking expects grayscale images")
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Previous and current frames must have identical shapes")
    if not _point_has_patch(previous_xy, previous_gray.shape, config.patch_size):
        return FlowResult(status="out_of_bounds")
    patch = cv2.getRectSubPix(
        previous_gray,
        (config.patch_size, config.patch_size),
        previous_xy,
    )
    if float(np.std(patch)) < config.min_texture_std:
        return FlowResult(status="low_texture")

    previous_point = np.asarray(previous_xy, dtype=np.float32).reshape(1, 1, 2)
    velocity = np.asarray(initial_velocity, dtype=np.float32)
    initial_current = (previous_point.reshape(2) + velocity).reshape(1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    window_size = config.recovery_lk_window_size if recovery else config.lk_window_size
    max_level = config.recovery_lk_max_level if recovery else config.lk_max_level
    current_point, status_forward, error_forward = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_point,
        initial_current,
        winSize=(window_size, window_size),
        maxLevel=max_level,
        criteria=criteria,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        minEigThreshold=1e-4,
    )
    if (
        current_point is None
        or status_forward is None
        or int(status_forward.reshape(-1)[0]) != 1
    ):
        return FlowResult(status="flow_failed")
    current_xy = tuple(float(value) for value in current_point.reshape(2))
    if not _point_has_patch(current_xy, current_gray.shape, config.patch_size):
        return FlowResult(status="out_of_bounds")

    initial_previous = previous_point.copy()
    backward_point, status_backward, error_backward = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_point,
        initial_previous,
        winSize=(window_size, window_size),
        maxLevel=max_level,
        criteria=criteria,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        minEigThreshold=1e-4,
    )
    if (
        backward_point is None
        or status_backward is None
        or int(status_backward.reshape(-1)[0]) != 1
    ):
        return FlowResult(status="flow_failed", left_xy=current_xy)
    fb_error = float(np.linalg.norm(backward_point.reshape(2) - previous_point.reshape(2)))
    lk_error = float(error_forward.reshape(-1)[0]) if error_forward is not None else None
    threshold = config.flow_fb_threshold if fb_threshold is None else float(fb_threshold)
    if threshold <= 0:
        raise ValueError("fb_threshold must be positive")
    if fb_error > threshold:
        return FlowResult(
            status="flow_failed",
            left_xy=current_xy,
            fb_error_px=fb_error,
            lk_error=lk_error,
        )
    return FlowResult(
        status="valid",
        left_xy=current_xy,
        fb_error_px=fb_error,
        lk_error=lk_error,
    )


def _point_has_patch(
    xy: tuple[float, float],
    image_shape: tuple[int, ...],
    patch_size: int,
) -> bool:
    radius = patch_size // 2
    x, y = xy
    return (
        x - radius >= 0
        and y - radius >= 0
        and x + radius < image_shape[1]
        and y + radius < image_shape[0]
    )
