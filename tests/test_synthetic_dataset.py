from __future__ import annotations

import json

import numpy as np

from stereo_dynamic_measurement.calibration.camera_model import StereoCameraModel
from stereo_dynamic_measurement.simulation.stereo_scene import StereoScene
from stereo_dynamic_measurement.simulation.synthetic_dataset import (
    SimulationConfig,
    generate_synthetic_dataset,
    write_synthetic_dataset,
)


def _camera() -> StereoCameraModel:
    return StereoCameraModel.from_mapping(
        {
            "image_size": [640, 480],
            "K_left": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
            "D_left": [0, 0, 0, 0, 0], "K_right": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
            "D_right": [0, 0, 0, 0, 0], "R": np.eye(3).tolist(), "T": [-120, 0, 0], "unit": "mm",
        }
    )


def test_stereo_scene_projects_points_and_injects_independent_pixel_noises() -> None:
    scene = StereoScene(_camera(), image_noise_std_px=0.1, localization_noise_std_px=0.05, random_seed=11)
    xyz = np.array([[0.0, 0.0, 2000.0], [100.0, 20.0, 2400.0]])

    projection = scene.project(xyz)

    assert projection.left_points_px.shape == (2, 2)
    assert projection.right_points_px.shape == (2, 2)
    assert np.any(np.abs(projection.left_points_px - projection.left_ideal_px) > 0)
    assert np.all(projection.left_points_px[:, 0] > projection.right_points_px[:, 0])


def test_synthetic_dataset_exports_errors_and_summary_in_mm(tmp_path) -> None:
    config = SimulationConfig(camera=_camera(), duration_s=0.5, fps=10.0, image_noise_std_px=0.0, localization_noise_std_px=0.0)

    result = generate_synthetic_dataset(config)
    paths = write_synthetic_dataset(result, tmp_path)

    assert {"frame", "time_s", "point_id", "X_gt_mm", "X_estimated_mm", "error_x_mm", "rmse_3d_mm"} <= set(result.samples.columns)
    assert result.summary["rmse_3d_mm"] < 1e-6
    assert paths["samples_csv"].exists() and paths["dataset_npz"].exists() and paths["summary_json"].exists()
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["unit"] == "mm"
    assert summary["sample_count"] == 24
