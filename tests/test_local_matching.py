from __future__ import annotations

import cv2
import numpy as np

import stereo_research.local_matching as local_matching_module
from stereo_research.local_matching import (
    LocalMatcher,
    QualityLocalMatcher,
    normalized_constraint_cost,
    predict_disparity,
    weighted_median,
)
from stereo_research.models import MatcherConfig, method_profile


def _textured_pair(disparity: float, width: int = 180, height: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    left = rng.integers(0, 256, (height, width), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0.45)
    transform = np.array([[1.0, 0.0, -disparity], [0.0, 1.0, 0.0]], dtype=np.float32)
    right = cv2.warpAffine(
        left,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return left, right


def test_local_match_recovers_known_integer_disparity() -> None:
    left, right = _textured_pair(9.0)
    config = MatcherConfig(uniqueness_margin=0.01)
    matcher = LocalMatcher(config)

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=8.0,
        profile=method_profile("local_flow"),
    )

    assert result.status == "valid"
    assert abs(result.disparity - 9.0) <= 0.1
    assert np.allclose(result.right_xy, (91.0, 60.0), atol=0.1)


def test_full_match_subpixel_refinement_recovers_fractional_disparity() -> None:
    left, right = _textured_pair(8.4)
    config = MatcherConfig(uniqueness_margin=0.005)
    matcher = LocalMatcher(config)

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
    )

    assert result.status == "valid"
    assert abs(result.disparity - 8.4) <= 0.2
    assert result.lr_error_px is not None
    assert result.lr_error_px <= 1.0


def test_local_match_rejects_low_texture_patch() -> None:
    left = np.full((100, 140), 128, dtype=np.uint8)
    right = left.copy()
    matcher = LocalMatcher(MatcherConfig())

    result = matcher.match(
        left,
        right,
        left_xy=(80.0, 50.0),
        predicted_disparity=6.0,
        profile=method_profile("full"),
    )

    assert result.status == "low_texture"


def test_local_match_reports_out_of_bounds() -> None:
    left, right = _textured_pair(8.0)
    matcher = LocalMatcher(MatcherConfig())

    result = matcher.match(
        left,
        right,
        left_xy=(2.0, 2.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
    )

    assert result.status == "out_of_bounds"


def test_disparity_prediction_clips_constant_velocity() -> None:
    assert predict_disparity(10.0, 8.0, max_velocity=3.0) == 12.0
    assert predict_disparity(10.0, 2.0, max_velocity=3.0) == 13.0
    assert predict_disparity(5.0, None, max_velocity=3.0) == 5.0


def test_weighted_median_ignores_a_low_weight_outlier() -> None:
    value = weighted_median(
        np.array([8.0, 8.2, 7.9, 40.0]),
        np.array([1.0, 1.0, 1.0, 0.01]),
    )

    assert value == 8.0


def test_lr_reverse_search_penalizes_epipolar_decoy() -> None:
    rng = np.random.default_rng(7)
    right = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    left = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    patch_size = 5
    radius = patch_size // 2
    right_xy = (70.0, 60.0)
    original_left_xy = (90.0, 60.0)
    source = right[58:63, 68:73].copy()
    noise = np.array(
        [
            [0, 1, 0, -1, 0],
            [1, 0, -1, 0, 1],
            [0, -1, 0, 1, 0],
            [-1, 0, 1, 0, -1],
            [0, 1, 0, -1, 0],
        ],
        dtype=np.int16,
    )
    true_patch = np.clip(source.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    true_x, true_y = map(int, original_left_xy)
    left[
        true_y - radius : true_y + radius + 1,
        true_x - radius : true_x + radius + 1,
    ] = true_patch
    decoy_x, decoy_y = 82, 59
    left[
        decoy_y - radius : decoy_y + radius + 1,
        decoy_x - radius : decoy_x + radius + 1,
    ] = source
    matcher = LocalMatcher(
        MatcherConfig(
            patch_size=patch_size,
            search_radius=8,
            expanded_search_radius=16,
        )
    )

    error = matcher._left_right_error(
        left,
        right,
        original_left_xy,
        right_xy,
        disparity=20.0,
    )

    assert error is not None
    assert error <= 1.0


def test_full_match_rejects_correspondence_outside_epipolar_band() -> None:
    left, _ = _textured_pair(8.0)
    transform = np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 4.0]], dtype=np.float32)
    right = cv2.warpAffine(
        left,
        transform,
        (left.shape[1], left.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    matcher = LocalMatcher(MatcherConfig(uniqueness_margin=0.005))

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
    )

    assert result.status != "valid"


def test_full_match_rejects_occluded_right_patch() -> None:
    left, right = _textured_pair(9.0)
    right[52:69, 83:100] = 128
    matcher = LocalMatcher(MatcherConfig(uniqueness_margin=0.005))

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=9.0,
        profile=method_profile("full"),
    )

    assert result.status in {"ambiguous", "lr_failed"}


def test_quality_matcher_recovers_repetitive_patch_with_larger_context() -> None:
    rng = np.random.default_rng(5)
    repeating_tile = rng.integers(0, 256, (120, 4), dtype=np.uint8)
    left = np.tile(repeating_tile, (1, 45))
    left[:, 106:108] = np.tile(np.array([[20, 240]], dtype=np.uint8), (120, 1))
    right = cv2.warpAffine(
        left,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (180, 120),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT101,
    )
    matcher_class = getattr(local_matching_module, "QualityLocalMatcher", LocalMatcher)
    matcher = matcher_class(MatcherConfig(max_photo_cost=0.6))

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
    )

    assert result.status == "valid"
    assert abs(result.disparity - 8.0) <= 0.1
    assert result.quality_stage == "context_recovery"


def test_quality_matcher_keeps_lr_on_raw_measurement_and_defers_time_estimation() -> None:
    left, right = _textured_pair(5.0)
    matcher = local_matching_module.QualityLocalMatcher(
        MatcherConfig(uniqueness_margin=0.005)
    )

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=5.0,
        profile=method_profile("full"),
        latest_disparity=5.5,
    )

    assert result.status == "valid"
    assert result.raw_disparity is not None
    assert abs(result.raw_disparity - 5.0) <= 0.1
    assert abs(result.disparity - 5.0) <= 0.1
    assert result.quality_stage == "primary"
    assert result.lr_error_px is not None
    assert result.lr_error_px <= 1.0


def test_quality_matcher_preserves_large_verified_disparity_motion() -> None:
    left, right = _textured_pair(7.5)
    matcher = local_matching_module.QualityLocalMatcher(
        MatcherConfig(uniqueness_margin=0.005)
    )

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=7.5,
        profile=method_profile("full"),
        latest_disparity=5.0,
    )

    assert result.status == "valid"
    assert abs(result.disparity - 7.5) <= 0.15
    assert result.quality_stage == "motion_verified"
    assert result.lr_error_px is not None
    assert result.lr_error_px <= 1.0


def test_constraint_cost_is_renormalized_when_terms_are_disabled() -> None:
    full = normalized_constraint_cost(
        [(0.55, 0.4), (0.20, 0.4), (0.10, 0.4), (0.15, 0.4)]
    )
    photo_only = normalized_constraint_cost([(0.55, 0.4)])

    assert abs(full - 0.4) < 1e-12
    assert abs(photo_only - 0.4) < 1e-12


def test_context_recovery_large_motion_requires_independent_pyramid_evidence(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(5)
    repeating_tile = rng.integers(0, 256, (120, 4), dtype=np.uint8)
    left = np.tile(repeating_tile, (1, 45))
    left[:, 106:108] = np.tile(np.array([[20, 240]], dtype=np.uint8), (120, 1))
    right = cv2.warpAffine(
        left,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (180, 120),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT101,
    )
    matcher = QualityLocalMatcher(MatcherConfig(max_photo_cost=0.6))
    monkeypatch.setattr(matcher, "_pyramid_disparity", lambda *_args: 2.0, raising=False)

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
        latest_disparity=2.0,
    )

    assert result.status == "ambiguous"
    assert result.quality_stage == "context_motion_rejected"


def test_no_pyramid_ablation_uses_non_pyramid_independent_evidence(monkeypatch) -> None:
    rng = np.random.default_rng(5)
    repeating_tile = rng.integers(0, 256, (120, 4), dtype=np.uint8)
    left = np.tile(repeating_tile, (1, 45))
    left[:, 106:108] = np.tile(np.array([[20, 240]], dtype=np.uint8), (120, 1))
    right = cv2.warpAffine(
        left,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (180, 120),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT101,
    )
    matcher = QualityLocalMatcher(MatcherConfig(max_photo_cost=0.6, enable_pyramid=False))

    def forbidden(*_args):
        raise AssertionError("pyramid was called by full_no_pyramid")

    monkeypatch.setattr(matcher, "_pyramid_disparity", forbidden)
    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
        latest_disparity=2.0,
    )

    assert result.status in {"valid", "ambiguous"}


def test_quality_result_keeps_measured_and_estimated_disparity_separate() -> None:
    left, right = _textured_pair(5.0)
    matcher = QualityLocalMatcher(MatcherConfig(uniqueness_margin=0.005))

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=5.0,
        profile=method_profile("full"),
        latest_disparity=5.5,
    )

    assert result.status == "valid"
    assert abs(result.measured_disparity - 5.0) <= 0.1
    assert abs(result.estimated_disparity - 5.0) <= 0.1
    assert abs(result.right_xy[0] - 95.0) <= 0.1
    assert result.estimated_right_xy is not None
    assert abs(result.estimated_right_xy[0] - 95.0) <= 0.1


def test_pyramid_recovers_disparity_outside_single_scale_expansion() -> None:
    left, right = _textured_pair(40.0, width=280, height=160)
    matcher = QualityLocalMatcher(
        MatcherConfig(uniqueness_margin=0.005, quality_far_disparity_px=50.0)
    )

    result = matcher.match(
        left,
        right,
        left_xy=(190.0, 80.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
        latest_disparity=8.0,
    )

    assert result.status == "valid"
    assert abs(result.measured_disparity - 40.0) <= 0.2
    assert result.quality_stage == "pyramid_recovery"


def test_continuous_subpixel_records_integer_and_offset() -> None:
    left, right = _textured_pair(8.4)
    matcher = LocalMatcher(
        MatcherConfig(
            uniqueness_margin=0.005,
            subpixel_method="continuous",
            subpixel_step=0.1,
        )
    )

    result = matcher.match(
        left,
        right,
        left_xy=(100.0, 60.0),
        predicted_disparity=8.0,
        profile=method_profile("full"),
    )

    assert result.integer_disparity == 8.0
    assert abs(result.measured_disparity - 8.4) <= 0.2
    assert abs(result.subpixel_offset - 0.4) <= 0.2
