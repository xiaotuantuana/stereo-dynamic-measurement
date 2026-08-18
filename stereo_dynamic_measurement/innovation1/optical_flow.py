from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowResult:
    point: tuple[float, float] | None
    predicted_point: tuple[float, float]
    fb_error_px: float | None
    valid: bool
    status: str


def track_forward_backward(
    previous_gray: np.ndarray, current_gray: np.ndarray, point: tuple[float, float], initial_velocity: tuple[float, float] = (0.0, 0.0),
    fb_threshold_px: float = 1.0, window_size: int = 21, max_level: int = 3,
) -> FlowResult:
    """LK point tracking with an explicit reverse pass and FB acceptance criterion."""
    if previous_gray.ndim != 2 or current_gray.ndim != 2 or previous_gray.shape != current_gray.shape:
        raise ValueError("Forward-backward LK requires equally sized grayscale frames")
    if fb_threshold_px <= 0:
        raise ValueError("fb_threshold_px must be positive")
    source = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    predicted = tuple(float(point[i] + initial_velocity[i]) for i in range(2))
    initial = np.asarray(predicted, dtype=np.float32).reshape(1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, source, initial, winSize=(window_size, window_size), maxLevel=max_level, criteria=criteria, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    if forward is None or forward_status is None or int(forward_status[0, 0]) != 1:
        return FlowResult(None, predicted, None, False, "forward_failed")
    tracked = tuple(float(value) for value in forward.reshape(2))
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, forward, source.copy(), winSize=(window_size, window_size), maxLevel=max_level, criteria=criteria, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    if backward is None or backward_status is None or int(backward_status[0, 0]) != 1:
        return FlowResult(tracked, predicted, None, False, "backward_failed")
    fb_error = float(np.linalg.norm(backward.reshape(2) - source.reshape(2)))
    return FlowResult(tracked, predicted, fb_error, fb_error <= fb_threshold_px, "valid" if fb_error <= fb_threshold_px else "fb_rejected")
