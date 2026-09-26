from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .models import MatcherConfig


@dataclass(frozen=True)
class GlobalStereoResult:
    left_disparity: np.ndarray
    right_disparity: np.ndarray
    elapsed_ms: float
    num_disparities: int


@dataclass(frozen=True)
class GlobalSample:
    status: str
    disparity: float | None = None
    right_xy: tuple[float, float] | None = None
    lr_error_px: float | None = None


class GlobalStereoMatcher:
    def __init__(self, config: MatcherConfig):
        self.config = config

    def compute(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        num_disparities: int | None = None,
    ) -> GlobalStereoResult:
        self._validate_images(left_gray, right_gray)
        disparities = num_disparities or self.config.num_disparities
        disparities = max(16, int(np.ceil(disparities / 16.0)) * 16)
        start = time.perf_counter()
        left_matcher = self._make_matcher(0, disparities)
        right_matcher = self._make_matcher(-disparities + 1, disparities)
        left_disparity = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        right_disparity = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return GlobalStereoResult(
            left_disparity=left_disparity,
            right_disparity=right_disparity,
            elapsed_ms=elapsed_ms,
            num_disparities=disparities,
        )

    def sample_initial(
        self,
        result: GlobalStereoResult,
        left_xy: tuple[float, float],
    ) -> GlobalSample:
        x = int(round(left_xy[0]))
        y = int(round(left_xy[1]))
        if not self._valid_index(result.left_disparity, x, y):
            return GlobalSample(status="out_of_bounds")
        disparity = float(result.left_disparity[y, x])
        if not np.isfinite(disparity) or disparity <= 0:
            return GlobalSample(status="ambiguous")
        if disparity >= result.num_disparities - 1.5:
            return GlobalSample(status="saturated", disparity=disparity)
        xr = float(left_xy[0] - disparity)
        right_disparity = self._bilinear_sample(result.right_disparity, xr, left_xy[1])
        if right_disparity is None:
            return GlobalSample(status="out_of_bounds", disparity=disparity)
        lr_error = abs(disparity + right_disparity)
        if lr_error > self.config.lr_threshold:
            return GlobalSample(
                status="lr_failed",
                disparity=disparity,
                right_xy=(xr, float(left_xy[1])),
                lr_error_px=lr_error,
            )
        return GlobalSample(
            status="valid",
            disparity=disparity,
            right_xy=(xr, float(left_xy[1])),
            lr_error_px=lr_error,
        )

    def sample_dense(
        self,
        result: GlobalStereoResult,
        left_xy: tuple[float, float],
    ) -> GlobalSample:
        x = int(round(left_xy[0]))
        y = int(round(left_xy[1]))
        if not self._valid_index(result.left_disparity, x, y):
            return GlobalSample(status="out_of_bounds")
        disparity = float(result.left_disparity[y, x])
        if not np.isfinite(disparity) or disparity <= 0:
            return GlobalSample(status="ambiguous")
        if disparity >= result.num_disparities - 1.5:
            return GlobalSample(status="saturated", disparity=disparity)
        return GlobalSample(
            status="valid",
            disparity=disparity,
            right_xy=(float(left_xy[0] - disparity), float(left_xy[1])),
        )

    def _make_matcher(self, min_disparity: int, num_disparities: int):
        channels = 1
        p1 = 8 * channels * self.config.block_size**2
        p2 = 32 * channels * self.config.block_size**2
        return cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=self.config.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self.config.disp12_max_diff,
            uniquenessRatio=self.config.uniqueness_ratio,
            speckleWindowSize=self.config.speckle_window_size,
            speckleRange=self.config.speckle_range,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    @staticmethod
    def _bilinear_sample(image: np.ndarray, x: float, y: float) -> float | None:
        if x < 0 or y < 0 or x >= image.shape[1] - 1 or y >= image.shape[0] - 1:
            return None
        value = cv2.getRectSubPix(image, (1, 1), (float(x), float(y)))
        return float(value[0, 0])

    @staticmethod
    def _valid_index(image: np.ndarray, x: int, y: int) -> bool:
        return 0 <= x < image.shape[1] and 0 <= y < image.shape[0]

    @staticmethod
    def _validate_images(left_gray: np.ndarray, right_gray: np.ndarray) -> None:
        if left_gray.ndim != 2 or right_gray.ndim != 2:
            raise ValueError("GlobalStereoMatcher expects grayscale images")
        if left_gray.shape != right_gray.shape:
            raise ValueError("Left and right images must have identical shapes")
        if left_gray.dtype != np.uint8 or right_gray.dtype != np.uint8:
            raise ValueError("GlobalStereoMatcher expects uint8 images")
