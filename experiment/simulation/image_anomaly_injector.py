"""Deterministic image-only anomaly injection for Phase 4.5 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import cv2
import numpy as np


_TYPE_PRIMITIVES = {
    "local_occlusion": {"occlusion"},
    "local_blur": {"blur"},
    "unilateral_roi_shift": {"roi_shift"},
    "continuous_anomaly": {"occlusion", "blur", "roi_shift"},
}


@dataclass(frozen=True)
class ImageAnomalyCase:
    """An immutable case specification which can transform only image copies."""

    case_id: str
    pair_id: str
    anomaly_type: str
    primitive: str
    active_frame_ids: tuple[str, ...]
    point_id: str
    side: str
    roi: tuple[int, int, int, int]
    seed: int
    shift_px: int = 0
    blur_kernel: int = 5

    def __post_init__(self) -> None:
        if self.anomaly_type not in _TYPE_PRIMITIVES:
            raise ValueError(f"unsupported anomaly_type: {self.anomaly_type}")
        if self.primitive not in _TYPE_PRIMITIVES[self.anomaly_type]:
            raise ValueError("primitive is incompatible with anomaly_type")
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        if not self.active_frame_ids:
            raise ValueError("active_frame_ids must not be empty")
        if self.anomaly_type == "continuous_anomaly":
            if not 2 <= len(self.active_frame_ids) <= 4 or not self._consecutive():
                raise ValueError("continuous anomaly requires two to four consecutive frames")
        elif len(self.active_frame_ids) != 1:
            raise ValueError("single-frame anomaly requires exactly one active frame")
        x, y, width, height = self.roi
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("roi must have non-negative origin and positive size")
        if self.primitive == "roi_shift" and self.shift_px not in {1, 2, 3}:
            raise ValueError("shift_px must be 1, 2, or 3 for roi_shift")
        if self.primitive == "blur" and (self.blur_kernel < 3 or self.blur_kernel % 2 != 1):
            raise ValueError("blur_kernel must be an odd integer of at least 3")

    def _consecutive(self) -> bool:
        try:
            values = [int(frame_id) for frame_id in self.active_frame_ids]
        except ValueError as error:
            raise ValueError("continuous frame IDs must be numeric") from error
        return all(right == left + 1 for left, right in zip(values, values[1:]))


class ImageAnomalyInjector:
    """Apply a case to a fresh image array, never to a source image or result."""

    version = "phase4_5_image_copy_v1"

    def __init__(self, case: ImageAnomalyCase) -> None:
        self.case = case

    def transform(self, image: np.ndarray, *, frame_id: str, side: str) -> np.ndarray:
        """Return an independent deterministic copy, transformed only if active."""

        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        source = np.asarray(image)
        if source.ndim not in {2, 3}:
            raise ValueError("image must be two-dimensional or three-dimensional")
        output = source.copy()
        if frame_id not in self.case.active_frame_ids or side != self.case.side:
            return output
        x, y, width, height = self.case.roi
        if x + width > source.shape[1] or y + height > source.shape[0]:
            raise ValueError("roi exceeds image bounds")
        patch = output[y:y + height, x:x + width]
        if self.case.primitive == "occlusion":
            patch[...] = self._occluder(patch.shape, source.dtype, frame_id)
        elif self.case.primitive == "blur":
            patch[...] = cv2.GaussianBlur(patch, (self.case.blur_kernel, self.case.blur_kernel), 0)
        else:
            patch[...] = np.roll(patch, self.case.shift_px, axis=1)
        return output

    def _occluder(self, shape: tuple[int, ...], dtype: np.dtype, frame_id: str) -> np.ndarray:
        digest = hashlib.sha256(
            f"{self.case.case_id}|{self.case.seed}|{frame_id}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        if len(shape) == 2:
            return rng.integers(0, 256, size=shape, dtype=dtype)
        return rng.integers(0, 256, size=shape, dtype=dtype)
