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
    cost_curvature: float | None = None
    converged: bool = False


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
        return ICGNResult(status="icgn_invalid_input")

    radius = patch_size // 2
    if not _patch_in_bounds(left_xy, left_gray.shape, radius):
        return ICGNResult(status="icgn_out_of_bounds")
    initial_right_xy = (left_xy[0] - initial_disparity, left_xy[1] + vertical_offset)
    if not _patch_in_bounds(initial_right_xy, right_gray.shape, radius + 2):
        return ICGNResult(status="icgn_out_of_bounds")

    template = _sample_patch_cubic(left_gray, left_xy, patch_size)
    template_normalized = _normalize_patch(template, epsilon)
    if template_normalized is None:
        return ICGNResult(status="icgn_low_gradient")
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
        return ICGNResult(status="icgn_low_gradient", hessian=hessian)

    disparity = float(initial_disparity)
    converged = False
    residual_rms: float | None = None
    iterations = 0
    best_disparity = disparity
    best_rms = float("inf")
    for iterations in range(1, max_iterations + 1):
        right_xy = (left_xy[0] - disparity, left_xy[1] + vertical_offset)
        if not _patch_in_bounds(right_xy, right_gray.shape, radius + 2):
            return ICGNResult(
                status="icgn_out_of_bounds",
                iterations=iterations - 1,
                hessian=hessian,
            )
        warped = _sample_patch_cubic(right_gray, right_xy, patch_size)
        warped_normalized = _normalize_patch(warped, epsilon)
        if warped_normalized is None:
            return ICGNResult(
                status="icgn_low_gradient",
                iterations=iterations - 1,
                hessian=hessian,
            )
        residual = warped_normalized - template_normalized
        residual_rms = float(np.sqrt(np.mean(residual * residual)))
        if residual_rms < best_rms:
            best_rms = residual_rms
            best_disparity = disparity
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
            return ICGNResult(status="icgn_low_gradient", hessian=hessian)
        normalized_jacobian = (plus - minus) / (2.0 * derivative_step)
        update_hessian = float(np.sum(normalized_jacobian * normalized_jacobian))
        if not np.isfinite(update_hessian) or update_hessian <= 1e-9:
            return ICGNResult(status="icgn_low_gradient", hessian=hessian)
        delta = float(-np.sum(normalized_jacobian * residual) / update_hessian)
        if not np.isfinite(delta):
            return ICGNResult(status="icgn_numerical_failure", iterations=iterations)
        disparity += delta
        offset = disparity - float(initial_disparity)
        if abs(offset) > max_offset:
            return ICGNResult(
                status="icgn_offset_exceeded",
                offset_px=offset,
                iterations=iterations,
                residual_rms=residual_rms,
                hessian=hessian,
                cost_curvature=update_hessian / float(patch_size * patch_size),
            )
        if abs(delta) < epsilon:
            converged = True
            break

    # OpenCV's cubic remap uses a finite interpolation table.  A bounded
    # deterministic line search resolves the small limit cycle that can remain
    # after the IC-GN updates without allowing the solution to leave the
    # verified integer disparity basin.
    search_center = best_disparity
    search_low = max(float(initial_disparity) - max_offset, search_center - 0.12)
    search_high = min(float(initial_disparity) + max_offset, search_center + 0.12)
    search_values = np.linspace(search_low, search_high, 97)
    search_costs: list[float] = []
    for candidate_disparity in search_values:
        candidate_patch = _normalize_patch(
            _sample_patch_cubic(
                right_gray,
                (
                    left_xy[0] - float(candidate_disparity),
                    left_xy[1] + vertical_offset,
                ),
                patch_size,
            ),
            epsilon,
        )
        if candidate_patch is None:
            search_costs.append(float("inf"))
        else:
            difference = candidate_patch - template_normalized
            search_costs.append(float(np.mean(difference * difference)))
    finite_costs = np.asarray(search_costs, dtype=np.float64)
    if not np.isfinite(finite_costs).any():
        return ICGNResult(status="icgn_numerical_failure", hessian=hessian)
    minimum = float(np.nanmin(finite_costs))
    tied = np.flatnonzero(np.isclose(finite_costs, minimum, rtol=1e-9, atol=1e-12))
    best_index = int(tied[len(tied) // 2])
    disparity = float(search_values[best_index])
    residual_rms = float(np.sqrt(finite_costs[best_index]))
    converged = True
    offset = disparity - float(initial_disparity)
    return ICGNResult(
        status="valid" if converged else "icgn_not_converged",
        refined_disparity=disparity if converged else None,
        offset_px=offset,
        iterations=iterations,
        residual_rms=residual_rms,
        hessian=hessian,
        cost_curvature=update_hessian / float(patch_size * patch_size),
        converged=converged,
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
    radius = patch_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    map_x, map_y = np.meshgrid(
        offsets + np.float32(center[0]),
        offsets + np.float32(center[1]),
    )
    return cv2.remap(
        image.astype(np.float64, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )


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
