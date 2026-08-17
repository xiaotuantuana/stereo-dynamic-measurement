from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..calibration.camera_model import StereoCameraModel
from ..calibration.triangulation import triangulate_points
from .stereo_scene import StereoScene
from .trajectory_generator import MultiPointStructure, generate_multipt_trajectory


@dataclass(frozen=True)
class SimulationConfig:
    camera: StereoCameraModel
    duration_s: float = 5.0
    fps: float = 30.0
    image_noise_std_px: float = 0.0
    localization_noise_std_px: float = 0.0
    random_seed: int = 7
    structure: MultiPointStructure = field(default_factory=MultiPointStructure.default)

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.fps <= 0:
            raise ValueError("duration_s and fps must be positive")


@dataclass(frozen=True)
class SyntheticDataset:
    samples: pd.DataFrame
    summary: dict[str, float | int | str]


def generate_synthetic_dataset(config: SimulationConfig) -> SyntheticDataset:
    frame_count = int(round(config.duration_s * config.fps)) + 1
    time_s = np.linspace(0.0, config.duration_s, frame_count)
    truth = generate_multipt_trajectory(time_s, config.structure)
    truth.insert(0, "frame", np.tile(np.arange(frame_count, dtype=int), 4))
    scene = StereoScene(
        config.camera, config.image_noise_std_px, config.localization_noise_std_px, config.random_seed
    )
    xyz_gt = truth[["X_gt_mm", "Y_gt_mm", "Z_gt_mm"]].to_numpy(dtype=np.float64)
    projection = scene.project(xyz_gt)
    xyz_estimated = triangulate_points(projection.left_points_px, projection.right_points_px, config.camera.P1, config.camera.P2)
    result = truth.copy()
    result["left_x_px"] = projection.left_points_px[:, 0]
    result["left_y_px"] = projection.left_points_px[:, 1]
    result["right_x_px"] = projection.right_points_px[:, 0]
    result["right_y_px"] = projection.right_points_px[:, 1]
    result["X_estimated_mm"] = xyz_estimated[:, 0]
    result["Y_estimated_mm"] = xyz_estimated[:, 1]
    result["Z_estimated_mm"] = xyz_estimated[:, 2]
    errors = xyz_estimated - xyz_gt
    result["error_x_mm"] = errors[:, 0]
    result["error_y_mm"] = errors[:, 1]
    result["error_z_mm"] = errors[:, 2]
    error_3d = np.linalg.norm(errors, axis=1)
    rmse_3d = float(np.sqrt(np.mean(np.square(error_3d))))
    result["rmse_3d_mm"] = rmse_3d
    summary: dict[str, float | int | str] = {
        "unit": "mm", "sample_count": len(result), "frame_count": frame_count,
        "rmse_3d_mm": rmse_3d, "error_x_rmse_mm": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "error_y_rmse_mm": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "error_z_rmse_mm": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        "baseline_mm": config.camera.baseline_mm, "image_noise_std_px": config.image_noise_std_px,
        "localization_noise_std_px": config.localization_noise_std_px,
    }
    return SyntheticDataset(result, summary)


def write_synthetic_dataset(dataset: SyntheticDataset, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples_csv = output / "synthetic_measurements.csv"
    dataset_npz = output / "synthetic_measurements.npz"
    summary_json = output / "summary.json"
    dataset.samples.to_csv(samples_csv, index=False, encoding="utf-8-sig")
    np.savez_compressed(dataset_npz, **{column: dataset.samples[column].to_numpy() for column in dataset.samples.columns})
    summary_json.write_text(json.dumps(dataset.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"samples_csv": samples_csv, "dataset_npz": dataset_npz, "summary_json": summary_json}
