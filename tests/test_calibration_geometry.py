from __future__ import annotations

import numpy as np
import pytest

from stereo_research.calibration import StereoCalibration, builtin_640x480
from stereo_research.geometry import reproject_point_m


def test_builtin_calibration_produces_expected_rectified_geometry() -> None:
    calibration = builtin_640x480()

    rectification = calibration.rectification()

    assert calibration.image_size == (640, 480)
    assert abs(rectification.focal_px - 514.8282456) < 1e-4
    assert abs(rectification.baseline_m - 0.1203579594) < 1e-7
    assert rectification.left_map1.shape[:2] == (480, 640)
    assert rectification.right_map1.shape[:2] == (480, 640)


def test_calibration_rejects_mismatched_image_size() -> None:
    calibration = builtin_640x480()
    left = np.zeros((240, 320, 3), dtype=np.uint8)
    right = np.zeros_like(left)

    with pytest.raises(ValueError, match="640x480"):
        calibration.rectify_pair(left, right)


def test_q_reprojection_returns_metre_coordinates() -> None:
    rectification = builtin_640x480().rectification()
    q = rectification.q
    x = float(-q[0, 3])
    y = float(-q[1, 3])
    disparity = rectification.focal_px * rectification.baseline_m / 2.0

    xyz = reproject_point_m(x, y, disparity, q, calibration_unit="mm")

    assert np.allclose(xyz, [0.0, 0.0, 2.0], atol=1e-5)


def test_q_reprojection_rejects_nonpositive_disparity() -> None:
    q = builtin_640x480().rectification().q

    with pytest.raises(ValueError, match="positive"):
        reproject_point_m(100.0, 100.0, 0.0, q)


def test_calibration_json_round_trip_preserves_numeric_parameters(tmp_path) -> None:
    source = builtin_640x480()
    path = tmp_path / "calibration.json"
    source.to_json(path)

    loaded = StereoCalibration.from_json(path)

    assert loaded.name == source.name
    assert loaded.image_size == source.image_size
    assert loaded.unit == "mm"
    assert np.allclose(loaded.left_camera_matrix, source.left_camera_matrix)
    assert np.allclose(loaded.translation, source.translation)
