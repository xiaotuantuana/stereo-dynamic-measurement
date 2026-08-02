from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo_research.icgn import refine_disparity_icgn
from stereo_research.local_matching import LocalMatcher
from stereo_research.models import MatcherConfig, method_profile


def _textured_image(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(127.0, 42.0, (150, 260)).astype(np.float32)
    image = cv2.GaussianBlur(image, (5, 5), 0.8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _right_from_disparity(left: np.ndarray, disparity: float) -> np.ndarray:
    transform = np.array([[1.0, 0.0, -disparity], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(
        left,
        transform,
        (left.shape[1], left.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )


@pytest.mark.parametrize("fraction", [0.25, 0.50, 0.75, 1.20])
def test_icgn_recovers_noise_free_fractional_disparity(fraction: float) -> None:
    left = _textured_image()
    true_disparity = 8.0 + fraction
    right = _right_from_disparity(left, true_disparity)

    result = refine_disparity_icgn(
        left,
        right,
        left_xy=(150.0, 75.0),
        initial_disparity=round(true_disparity),
        vertical_offset=0.0,
        patch_size=31,
        max_iterations=30,
        epsilon=1e-4,
        max_offset=1.5,
    )

    assert result.status == "valid"
    assert result.converged is True
    assert result.refined_disparity is not None
    assert abs(result.refined_disparity - true_disparity) < 0.02
    assert result.residual_rms is not None
    assert result.hessian is not None and result.hessian > 0


def test_icgn_is_robust_to_noise_and_brightness_offset() -> None:
    rng = np.random.default_rng(23)
    left = _textured_image()
    true_disparity = 9.65
    right = _right_from_disparity(left, true_disparity).astype(np.float32)
    right += 9.0 + rng.normal(0.0, 1.5, right.shape)
    right = np.clip(right, 0, 255).astype(np.uint8)

    result = refine_disparity_icgn(
        left,
        right,
        left_xy=(150.0, 75.0),
        initial_disparity=10.0,
        vertical_offset=0.0,
        patch_size=41,
        max_iterations=30,
        epsilon=1e-4,
        max_offset=1.5,
    )

    assert result.status == "valid"
    assert result.refined_disparity is not None
    assert abs(result.refined_disparity - true_disparity) < 0.05


def test_icgn_rejects_low_horizontal_gradient() -> None:
    flat = np.full((100, 180), 128, dtype=np.uint8)

    result = refine_disparity_icgn(
        flat,
        flat,
        left_xy=(90.0, 50.0),
        initial_disparity=8.0,
        vertical_offset=0.0,
        patch_size=15,
        max_iterations=20,
        epsilon=1e-3,
        max_offset=1.5,
    )

    assert result.status == "icgn_low_gradient"
    assert result.refined_disparity is None


def test_icgn_rejects_patch_outside_image() -> None:
    image = _textured_image()

    result = refine_disparity_icgn(
        image,
        image,
        left_xy=(3.0, 3.0),
        initial_disparity=1.0,
        vertical_offset=0.0,
        patch_size=15,
        max_iterations=20,
        epsilon=1e-3,
        max_offset=1.5,
    )

    assert result.status == "icgn_out_of_bounds"


def test_research_matcher_uses_icgn_and_exports_quality() -> None:
    left = _textured_image()
    true_disparity = 8.65
    right = _right_from_disparity(left, true_disparity)
    matcher = LocalMatcher(
        MatcherConfig(
            uniqueness_margin=0.005,
            icgn_patch_size=31,
            icgn_max_residual=0.5,
        )
    )

    result = matcher.match(
        left,
        right,
        left_xy=(150.0, 75.0),
        predicted_disparity=9.0,
        profile=method_profile("research_full"),
        latest_disparity=9.0,
    )

    assert result.status == "valid"
    assert result.icgn_status == "valid"
    assert result.icgn_converged is True
    assert result.icgn_iterations > 0
    assert result.icgn_residual is not None
    assert result.disparity is not None
    assert abs(result.disparity - true_disparity) < 0.05
    assert result.quality_stage.endswith("_icgn")


def test_matcher_reports_icgn_quality_rejection_and_falls_back() -> None:
    left = _textured_image()
    right = _right_from_disparity(left, 8.65)
    matcher = LocalMatcher(
        MatcherConfig(
            uniqueness_margin=0.005,
            icgn_patch_size=31,
            icgn_max_residual=1e-5,
            icgn_fallback_method="parabolic",
        )
    )

    result = matcher.match(
        left,
        right,
        left_xy=(150.0, 75.0),
        predicted_disparity=9.0,
        profile=method_profile("research_full"),
    )

    assert result.status == "valid"
    assert result.icgn_converged is True
    assert result.icgn_status == "icgn_residual_exceeded"
    assert result.quality_stage.endswith("_icgn_fallback_parabolic")
