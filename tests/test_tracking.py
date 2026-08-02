from __future__ import annotations

import cv2
import numpy as np

from stereo_research.models import MatcherConfig, PointState
from stereo_research.tracking import track_point_lk


def _translated_texture(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0.8)
    transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    current = cv2.warpAffine(
        previous,
        transform,
        (160, 120),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return previous, current


def test_lk_tracking_recovers_translation_with_forward_backward_check() -> None:
    previous, current = _translated_texture(2.0, 1.0)
    state = PointState(
        point_id="P1",
        initial_left_xy=(80.0, 60.0),
        previous_left_xy=(78.0, 59.0),
        left_xy=(80.0, 60.0),
    )

    result = track_point_lk(previous, current, state, MatcherConfig())

    assert result.status == "valid"
    assert np.allclose(result.left_xy, (82.0, 61.0), atol=0.15)
    assert result.fb_error_px is not None
    assert result.fb_error_px <= 0.15


def test_lk_tracking_rejects_low_texture() -> None:
    previous = np.full((100, 120), 90, dtype=np.uint8)
    current = previous.copy()
    state = PointState(point_id="P1", initial_left_xy=(60.0, 50.0), left_xy=(60.0, 50.0))

    result = track_point_lk(previous, current, state, MatcherConfig())

    assert result.status == "low_texture"


def test_lk_tracking_rejects_point_near_boundary() -> None:
    previous, current = _translated_texture(1.0, 0.0)
    state = PointState(point_id="P1", initial_left_xy=(2.0, 2.0), left_xy=(2.0, 2.0))

    result = track_point_lk(previous, current, state, MatcherConfig())

    assert result.status == "out_of_bounds"
