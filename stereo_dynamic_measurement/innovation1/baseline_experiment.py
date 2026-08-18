from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineExperimentConfig:
    baselines_mm: tuple[float, ...] = (60.0, 100.0, 140.0, 180.0, 220.0, 260.0, 300.0)
    distances_mm: tuple[float, ...] = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0)
    disparity_noises_px: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    focal_length_px: float = 800.0
    samples_per_condition: int = 1000
    random_seed: int = 7


@dataclass(frozen=True)
class BaselineExperimentResult:
    csv_path: Path
    plot_path: Path


def run_baseline_distance_experiment(config: BaselineExperimentConfig, output_dir: str | Path) -> BaselineExperimentResult:
    """Monte-Carlo depth reconstruction under direct subpixel disparity noise."""
    if config.samples_per_condition < 2 or config.focal_length_px <= 0:
        raise ValueError("At least two samples and a positive focal length are required")
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, float]] = []
    for noise_px in config.disparity_noises_px:
        for baseline_mm in config.baselines_mm:
            for distance_mm in config.distances_mm:
                true_disparity = config.focal_length_px * baseline_mm / distance_mm
                measured_disparity = true_disparity + rng.normal(0.0, noise_px, config.samples_per_condition)
                estimated_depth = config.focal_length_px * baseline_mm / np.maximum(measured_disparity, 1e-6)
                errors = estimated_depth - distance_mm
                rows.append({
                    "baseline_mm": baseline_mm, "distance_mm": distance_mm, "disparity_noise_px": noise_px,
                    "true_disparity_px": true_disparity, "depth_rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
                    "depth_bias_mm": float(np.mean(errors)),
                    "theoretical_depth_sigma_mm": float(distance_mm ** 2 / (config.focal_length_px * baseline_mm) * noise_px),
                })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    csv_path = output / "baseline-distance-error.csv"
    plot_path = output / "baseline_distance_rmse.png"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _plot_heatmaps(table, config, plot_path)
    return BaselineExperimentResult(csv_path, plot_path)


def _plot_heatmaps(table: pd.DataFrame, config: BaselineExperimentConfig, output: Path) -> None:
    fig, axes = plt.subplots(1, len(config.disparity_noises_px), figsize=(16, 4), sharey=True, layout="constrained")
    for axis, noise in zip(np.atleast_1d(axes), config.disparity_noises_px, strict=True):
        subset = table[table["disparity_noise_px"] == noise]
        grid = subset.pivot(index="distance_mm", columns="baseline_mm", values="depth_rmse_mm")
        image = axis.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(f"$\\sigma_d$={noise:.2f} px")
        axis.set_xticks(range(len(grid.columns)), [f"{value:.0f}" for value in grid.columns])
        axis.set_yticks(range(len(grid.index)), [f"{value / 1000:.0f}" for value in grid.index])
        axis.set_xlabel("Baseline (mm)")
    axes[0].set_ylabel("Distance (m)")
    colorbar = fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), shrink=0.85)
    colorbar.set_label("Depth RMSE (mm)")
    fig.suptitle("Baseline-distance depth error under disparity noise")
    fig.savefig(output, dpi=150)
    plt.close(fig)
