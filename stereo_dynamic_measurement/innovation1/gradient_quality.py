from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class ImageQuality:
    gradient_mean: float
    gradient_std: float
    laplacian_variance: float
    gradient_score: float
    texture_score: float
    blur_score: float
    quality_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "gradient_score": self.gradient_score, "texture_score": self.texture_score,
            "blur_score": self.blur_score, "quality_score": self.quality_score,
        }


def assess_image_quality(gray: np.ndarray, patch_center: tuple[float, float] | None = None, patch_size: int = 31) -> ImageQuality:
    """Compute globally or locally normalized texture/blur evidence from a gray image."""
    if gray.ndim != 2:
        raise ValueError("Image quality expects a grayscale image")
    image = gray
    if patch_center is not None:
        image = cv2.getRectSubPix(gray, (patch_size, patch_size), patch_center)
    image_f = image.astype(np.float32)
    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    gradient_mean = float(np.mean(magnitude))
    gradient_std = float(np.std(magnitude))
    laplacian_variance = float(np.var(cv2.Laplacian(image_f, cv2.CV_32F)))
    gradient_score = _unit(gradient_mean / 80.0)
    texture_score = _unit(gradient_std / 80.0)
    blur_score = _unit(laplacian_variance / 500.0)
    return ImageQuality(
        gradient_mean, gradient_std, laplacian_variance, gradient_score, texture_score, blur_score,
        _unit(0.45 * gradient_score + 0.30 * texture_score + 0.25 * blur_score),
    )
