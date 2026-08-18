from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalMatchResult:
    integer_disparity_px: float | None
    match_cost: float
    cost_curve: dict[float, float]
    valid: bool
    status: str


class LocalStereoMatcher:
    """Rectified local epipolar matcher using ZNCC plus normalized gradient cost."""

    def __init__(self, patch_size: int = 15, gradient_weight: float = 0.2) -> None:
        if patch_size < 3 or patch_size % 2 == 0 or not 0.0 <= gradient_weight <= 1.0:
            raise ValueError("patch_size must be odd >= 3 and gradient_weight in [0, 1]")
        self.patch_size, self.gradient_weight = patch_size, gradient_weight

    def match(self, left_gray: np.ndarray, right_gray: np.ndarray, *, left_point: tuple[float, float], predicted_disparity_px: float, search_radius_px: int) -> LocalMatchResult:
        if left_gray.ndim != 2 or right_gray.ndim != 2 or left_gray.shape != right_gray.shape:
            raise ValueError("Local stereo matching requires equally sized grayscale frames")
        if search_radius_px < 1:
            raise ValueError("search_radius_px must be positive")
        radius = self.patch_size // 2
        x, y = left_point
        if x < radius or x >= left_gray.shape[1] - radius or y < radius or y >= left_gray.shape[0] - radius:
            return LocalMatchResult(None, float("inf"), {}, False, "left_out_of_bounds")
        left_patch = cv2.getRectSubPix(left_gray, (self.patch_size, self.patch_size), left_point).astype(np.float32)
        left_gradient = cv2.Sobel(left_patch, cv2.CV_32F, 1, 0, ksize=3)
        center = int(round(predicted_disparity_px))
        candidates = range(max(1, center - search_radius_px), center + search_radius_px + 1)
        curve: dict[float, float] = {}
        for disparity in candidates:
            right_x = x - disparity
            if right_x < radius or right_x >= right_gray.shape[1] - radius:
                continue
            right_patch = cv2.getRectSubPix(right_gray, (self.patch_size, self.patch_size), (right_x, y)).astype(np.float32)
            right_gradient = cv2.Sobel(right_patch, cv2.CV_32F, 1, 0, ksize=3)
            curve[float(disparity)] = self._cost(left_patch, right_patch, left_gradient, right_gradient)
        if not curve:
            return LocalMatchResult(None, float("inf"), curve, False, "right_out_of_bounds")
        disparity, cost = min(curve.items(), key=lambda item: item[1])
        return LocalMatchResult(disparity, cost, curve, np.isfinite(cost), "valid")

    def _cost(self, left: np.ndarray, right: np.ndarray, left_gradient: np.ndarray, right_gradient: np.ndarray) -> float:
        left_zero, right_zero = left - np.mean(left), right - np.mean(right)
        denominator = float(np.linalg.norm(left_zero) * np.linalg.norm(right_zero))
        zncc_cost = 1.0 if denominator <= 1e-9 else 1.0 - float(np.dot(left_zero.ravel(), right_zero.ravel()) / denominator)
        gradient_cost = float(np.mean(np.abs(left_gradient - right_gradient)) / 255.0)
        return float((1.0 - self.gradient_weight) * zncc_cost + self.gradient_weight * gradient_cost)
