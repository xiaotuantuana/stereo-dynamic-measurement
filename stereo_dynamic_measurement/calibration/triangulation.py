from __future__ import annotations

import cv2
import numpy as np


def triangulate_points(
    left_points_px: np.ndarray, right_points_px: np.ndarray, P1: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """Triangulate corresponding image points; output is mm when P matrices use mm."""
    left = np.asarray(left_points_px, dtype=np.float64)
    right = np.asarray(right_points_px, dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 2 or right.shape != left.shape:
        raise ValueError("left_points_px and right_points_px must both have shape (N, 2)")
    p1, p2 = np.asarray(P1, dtype=np.float64), np.asarray(P2, dtype=np.float64)
    if p1.shape != (3, 4) or p2.shape != (3, 4):
        raise ValueError("P1 and P2 must have shape (3, 4)")
    homogeneous = cv2.triangulatePoints(p1, p2, left.T, right.T)
    if np.any(np.isclose(homogeneous[3], 0.0)):
        raise ValueError("Triangulation produced a point at infinity")
    return (homogeneous[:3] / homogeneous[3]).T
