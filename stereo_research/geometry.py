from __future__ import annotations

import numpy as np


def reproject_point_m(
    x: float,
    y: float,
    disparity: float,
    q: np.ndarray,
    calibration_unit: str = "mm",
) -> np.ndarray:
    if not np.isfinite(disparity) or disparity <= 0:
        raise ValueError("Disparity must be finite and positive")
    if q.shape != (4, 4):
        raise ValueError(f"Q must have shape (4, 4), received {q.shape}")
    homogeneous = q @ np.array([x, y, disparity, 1.0], dtype=np.float64)
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[3])) <= 1e-12:
        raise ValueError("Q reprojection produced an invalid homogeneous coordinate")
    xyz = homogeneous[:3] / homogeneous[3]
    if calibration_unit == "mm":
        xyz = xyz / 1000.0
    elif calibration_unit != "m":
        raise ValueError(f"Unsupported calibration unit: {calibration_unit}")
    if not np.isfinite(xyz).all():
        raise ValueError("Q reprojection produced a non-finite point")
    return xyz.astype(np.float64)
