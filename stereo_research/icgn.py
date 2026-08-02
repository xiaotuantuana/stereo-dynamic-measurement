from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ICGNResult:
    status: str
    refined_disparity: float | None = None
    offset_px: float | None = None
    iterations: int = 0
    residual_rms: float | None = None
    hessian: float | None = None
    hessian_density: float | None = None
    # Deprecated compatibility field.  True photometric curvature is computed
    # by LocalMatcher at the final disparity, not by the IC-GN Hessian.
    cost_curvature: float | None = None
    converged: bool = False
    fallback_used: bool = False
    fallback_method: str = "none"
    initial_disparity: float | None = None
    iterative_disparity: float | None = None
    final_disparity: float | None = None
    final_increment_px: float | None = None
    termination_reason: str = ""


def refine_disparity_icgn(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_xy: tuple[float, float],
    initial_disparity: float,
    vertical_offset: float,
    patch_size: int,
    max_iterations: int,
    epsilon: float,
    max_offset: float,
    max_residual: float = float("inf"),
) -> ICGNResult:
    if left_gray.ndim != 2 or right_gray.ndim != 2:
        raise ValueError("IC-GN expects grayscale images")
    if left_gray.shape != right_gray.shape:
        raise ValueError("Left and right images must have identical shapes")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer of at least 3")
    if max_iterations <= 0 or epsilon <= 0 or max_offset <= 0:
        raise ValueError("IC-GN iteration controls must be positive")
    if not np.isfinite([*left_xy, initial_disparity, vertical_offset]).all():
        return ICGNResult(
            status="icgn_diverged",
            initial_disparity=float(initial_disparity),
            termination_reason="icgn_diverged",
        )

    radius = patch_size // 2
    if not _patch_in_bounds(left_xy, left_gray.shape, radius):
        return ICGNResult(
            status="icgn_out_of_bounds",
            initial_disparity=float(initial_disparity),
            termination_reason="icgn_out_of_bounds",
        )
    initial_right_xy = (left_xy[0] - initial_disparity, left_xy[1] + vertical_offset)
    if not _patch_in_bounds(initial_right_xy, right_gray.shape, radius + 2):
        return ICGNResult(
            status="icgn_out_of_bounds",
            initial_disparity=float(initial_disparity),
            termination_reason="icgn_out_of_bounds",
        )

    template = _sample_patch_cubic(left_gray, left_xy, patch_size)
    template_normalized = _normalize_patch(template, epsilon)
    if template_normalized is None:
        return ICGNResult(
            status="icgn_low_gradient",
            initial_disparity=float(initial_disparity),
            termination_reason="icgn_low_gradient",
        )
    gradient_x = cv2.Sobel(
        template_normalized,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT101,
    ) / 8.0
    hessian = float(np.sum(gradient_x * gradient_x))
    if not np.isfinite(hessian) or hessian <= 1e-6:
        return ICGNResult(
            status="icgn_low_gradient",
            hessian=hessian,
            hessian_density=hessian / float(patch_size * patch_size),
            initial_disparity=float(initial_disparity),
            termination_reason="icgn_low_gradient",
        )

    disparity = float(initial_disparity)
    residual_rms: float | None = None
    iterations = 0
    best_disparity = disparity
    best_rms = float("inf")
    last_delta: float | None = None
    last_update_hessian: float | None = None
    residual_increases = 0
    previous_rms: float | None = None
    for iterations in range(1, max_iterations + 1):
        right_xy = (left_xy[0] - disparity, left_xy[1] + vertical_offset)
        if not _patch_in_bounds(right_xy, right_gray.shape, radius + 2):
            return ICGNResult(
                status="icgn_out_of_bounds",
                initial_disparity=float(initial_disparity),
                iterative_disparity=best_disparity,
                offset_px=best_disparity - float(initial_disparity),
                iterations=iterations - 1,
                hessian=hessian,
                hessian_density=(
                    last_update_hessian / float(patch_size * patch_size)
                    if last_update_hessian is not None
                    else hessian / float(patch_size * patch_size)
                ),
                termination_reason="icgn_out_of_bounds",
            )
        warped = _sample_patch_cubic(right_gray, right_xy, patch_size)
        warped_normalized = _normalize_patch(warped, epsilon)
        if warped_normalized is None:
            return _terminated_result(
                "icgn_low_gradient", initial_disparity, best_disparity,
                iterations - 1, residual_rms, hessian, last_update_hessian,
                patch_size, last_delta,
            )
        residual = warped_normalized - template_normalized
        residual_rms = float(np.sqrt(np.mean(residual * residual)))
        if residual_rms < best_rms:
            best_rms = residual_rms
            best_disparity = disparity
        if previous_rms is not None and residual_rms > previous_rms * (1.0 + 1e-4):
            residual_increases += 1
        else:
            residual_increases = 0
        previous_rms = residual_rms
        if residual_increases >= 3:
            return _terminated_result(
                "icgn_diverged", initial_disparity, best_disparity,
                iterations, best_rms, hessian, last_update_hessian,
                patch_size, last_delta,
            )
        # Zero-mean normalization changes the image Jacobian.  A symmetric
        # derivative keeps the inverse-compositional update photometrically
        # invariant while the template Hessian above remains the texture gate.
        derivative_step = 0.03
        plus = _normalize_patch(
            _sample_patch_cubic(
                right_gray,
                (right_xy[0] - derivative_step, right_xy[1]),
                patch_size,
            ),
            epsilon,
        )
        minus = _normalize_patch(
            _sample_patch_cubic(
                right_gray,
                (right_xy[0] + derivative_step, right_xy[1]),
                patch_size,
            ),
            epsilon,
        )
        if plus is None or minus is None:
            return _terminated_result(
                "icgn_low_gradient", initial_disparity, best_disparity,
                iterations, residual_rms, hessian, last_update_hessian,
                patch_size, last_delta,
            )
        normalized_jacobian = (plus - minus) / (2.0 * derivative_step)
        update_hessian = float(np.sum(normalized_jacobian * normalized_jacobian))
        if not np.isfinite(update_hessian) or update_hessian <= 1e-9:
            return _terminated_result(
                "icgn_low_gradient", initial_disparity, best_disparity,
                iterations, residual_rms, hessian, update_hessian,
                patch_size, last_delta,
            )
        last_update_hessian = update_hessian
        delta = float(-np.sum(normalized_jacobian * residual) / update_hessian)
        if not np.isfinite(delta):
            return _terminated_result(
                "icgn_diverged", initial_disparity, best_disparity,
                iterations, residual_rms, hessian, update_hessian,
                patch_size, delta,
            )
        last_delta = delta
        disparity += delta
        offset = disparity - float(initial_disparity)
        if abs(offset) > max_offset:
            return _terminated_result(
                "icgn_diverged", initial_disparity, best_disparity,
                iterations, residual_rms, hessian, update_hessian,
                patch_size, delta,
            )
        if abs(delta) < epsilon:
            final_patch = _normalize_patch(
                _sample_patch_cubic(
                    right_gray,
                    (left_xy[0] - disparity, left_xy[1] + vertical_offset),
                    patch_size,
                ),
                epsilon,
            )
            final_rms = (
                float(np.sqrt(np.mean((final_patch - template_normalized) ** 2)))
                if final_patch is not None
                else float("inf")
            )
            if not np.isfinite(final_rms) or final_rms > max_residual:
                return _terminated_result(
                    "icgn_high_residual", initial_disparity, disparity,
                    iterations, final_rms, hessian, update_hessian,
                    patch_size, delta,
                )
            return ICGNResult(
                status="icgn_converged",
                refined_disparity=disparity,
                offset_px=offset,
                iterations=iterations,
                residual_rms=final_rms,
                hessian=hessian,
                hessian_density=update_hessian / float(patch_size * patch_size),
                converged=True,
                initial_disparity=float(initial_disparity),
                iterative_disparity=disparity,
                final_disparity=disparity,
                final_increment_px=delta,
                termination_reason="icgn_converged",
            )

    return _terminated_result(
        "icgn_max_iterations", initial_disparity, best_disparity,
        iterations, best_rms, hessian, last_update_hessian,
        patch_size, last_delta,
    )


def _terminated_result(
    status: str,
    initial_disparity: float,
    iterative_disparity: float,
    iterations: int,
    residual_rms: float | None,
    hessian: float,
    update_hessian: float | None,
    patch_size: int,
    final_increment: float | None,
) -> ICGNResult:
    return ICGNResult(
        status=status,
        offset_px=float(iterative_disparity - initial_disparity),
        iterations=iterations,
        residual_rms=residual_rms,
        hessian=hessian,
        hessian_density=(
            update_hessian / float(patch_size * patch_size)
            if update_hessian is not None and np.isfinite(update_hessian)
            else None
        ),
        converged=False,
        initial_disparity=float(initial_disparity),
        iterative_disparity=float(iterative_disparity),
        final_increment_px=final_increment,
        termination_reason=status,
    )


def _normalize_patch(patch: np.ndarray, epsilon: float) -> np.ndarray | None:
    values = patch.astype(np.float64, copy=False)
    standard_deviation = float(values.std())
    if not np.isfinite(standard_deviation) or standard_deviation <= epsilon:
        return None
    return (values - float(values.mean())) / standard_deviation


def _sample_patch_cubic(
    image: np.ndarray,
    center: tuple[float, float],
    patch_size: int,
) -> np.ndarray:
    # getRectSubPix provides continuous bilinear coordinates.  cv2.remap's
    # interpolation table quantizes coordinates and can create a small IC-GN
    # limit cycle that never satisfies the requested increment tolerance.
    return cv2.getRectSubPix(
        image.astype(np.float32, copy=False),
        (patch_size, patch_size),
        (float(center[0]), float(center[1])),
    ).astype(np.float64, copy=False)


def _patch_in_bounds(
    xy: tuple[float, float],
    image_shape: tuple[int, ...],
    radius: int,
) -> bool:
    x, y = xy
    return (
        x - radius >= 0
        and y - radius >= 0
        and x + radius < image_shape[1]
        and y + radius < image_shape[0]
    )
