from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from stereo_dynamic_measurement.calibration.camera_model import StereoCameraModel
from stereo_dynamic_measurement.calibration.triangulation import triangulate_points


def _payload() -> dict[str, object]:
    return {
        "name": "unit-test-rig",
        "image_size": [640, 480],
        "K_left": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        "D_left": [0.0, 0.0, 0.0, 0.0, 0.0],
        "K_right": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        "D_right": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "T": [-120.0, 0.0, 0.0],
        "unit": "mm",
    }


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".npz", ".xml"])
def test_camera_model_loads_all_supported_formats(tmp_path, suffix: str) -> None:
    path = tmp_path / f"calibration{suffix}"
    payload = _payload()
    if suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif suffix == ".yaml":
        path.write_text(
            "\n".join(f"{key}: {json.dumps(value)}" for key, value in payload.items()),
            encoding="utf-8",
        )
    elif suffix == ".npz":
        np.savez(path, **payload)
    else:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        for key, value in payload.items():
            if key == "name" or key == "unit":
                storage.write(key, value)
            else:
                storage.write(key, np.asarray(value))
        storage.release()

    model = StereoCameraModel.load(path)

    assert model.image_size == (640, 480)
    assert model.unit == "mm"
    assert model.baseline_mm == pytest.approx(120.0)
    assert model.P1.shape == (3, 4)
    assert model.P2.shape == (3, 4)
    assert model.Q.shape == (4, 4)


def test_camera_model_rejects_non_mm_calibration() -> None:
    payload = _payload()
    payload["unit"] = "m"

    with pytest.raises(ValueError, match="mm"):
        StereoCameraModel.from_mapping(payload)


def test_dlt_triangulation_recovers_mm_coordinates_without_noise() -> None:
    model = StereoCameraModel.from_mapping(_payload())
    ground_truth = np.array([[15.0, -20.0, 1800.0], [-30.0, 10.0, 2400.0]])
    left = model.project(ground_truth, camera="left")
    right = model.project(ground_truth, camera="right")

    estimated = triangulate_points(left, right, model.P1, model.P2)

    assert np.max(np.abs(estimated - ground_truth)) < 1e-6
