from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stereo_research.accuracy_policy import decide_measurement_policy

from .precision_planner import TargetAccuracySpec, build_precision_plan


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
    policy_csv_path: Path


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
    policy_csv_path = output / "target_accuracy_policy_ablation.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _plot_heatmaps(table, config, plot_path)
    _policy_ablation_table(config).to_csv(
        policy_csv_path, index=False, encoding="utf-8-sig"
    )
    return BaselineExperimentResult(csv_path, plot_path, policy_csv_path)


def _policy_ablation_table(config: BaselineExperimentConfig) -> pd.DataFrame:
    """Analytic policy ablation; GT errors are joined only after decisions."""

    rng = np.random.default_rng(config.random_seed + 1)
    distance_mm = 3000.0
    baseline_mm = min(config.baselines_mm, key=lambda value: abs(value - 120.0))
    estimated_sigma_d = min(config.disparity_noises_px, key=lambda value: abs(value - 0.05))
    minimum_sigma_d = min(config.disparity_noises_px)
    true_disparity = config.focal_length_px * baseline_mm / distance_mm
    measured_disparity = true_disparity + rng.normal(
        0.0, estimated_sigma_d, config.samples_per_condition
    )
    valid = np.isfinite(measured_disparity) & (measured_disparity > 0.0)
    measured_depth = np.full(measured_disparity.shape, np.nan, dtype=float)
    measured_depth[valid] = (
        config.focal_length_px * baseline_mm / measured_disparity[valid]
    )
    causal_depth_m = float(np.nanmedian(measured_depth) / 1000.0)

    groups = (
        ("A0_FIXED", False, "HIGH", 8),
        ("A1_VISION_ONLY", False, "LOW", 16),
        ("A2_ACCURACY_ONLY", True, "HIGH", 8),
        ("A3_FULL", True, "MEDIUM", 12),
    )
    targets = (
        ("LOOSE", 10.0),
        ("MEDIUM", 3.0),
        ("STRICT", 2.5),
        ("INFEASIBLE", 0.1),
    )
    rows: list[dict[str, object]] = []
    errors = measured_depth[valid] - distance_mm
    for group, accuracy_enabled, vision_state, search_radius in groups:
        for level, target_z_mm in targets:
            started = time.perf_counter()
            if accuracy_enabled:
                plan = build_precision_plan(
                    TargetAccuracySpec(target_sigma_z_mm=target_z_mm),
                    current_depth_m=causal_depth_m,
                    focal_length_px=config.focal_length_px,
                    baseline_mm=baseline_mm,
                    estimated_sigma_disparity_px=estimated_sigma_d,
                    minimum_achievable_sigma_disparity_px=minimum_sigma_d,
                )
                decision = decide_measurement_policy(
                    plan,
                    base_search_radius_px=search_radius,
                    vision_state=vision_state,
                    max_precision_retry=1,
                    max_refinement_level=2,
                )
                retry_count = 1 if decision.precision_status == "RETRY_STRONGER" else 0
                if retry_count:
                    decision = decide_measurement_policy(
                        plan,
                        base_search_radius_px=search_radius,
                        vision_state=vision_state,
                        max_precision_retry=1,
                        max_refinement_level=2,
                        precision_retry_count=retry_count,
                    )
                required_sigma_d = plan.required_sigma_disparity_px
                estimated_sigma_z = plan.estimated_sigma_z_mm
                refinement_level = decision.refinement_level
                status = decision.precision_status
                estimated_attainment = float(status == "MET")
            else:
                required_sigma_d = None
                estimated_sigma_z = (
                    causal_depth_m * 1000.0
                ) ** 2 * estimated_sigma_d / (config.focal_length_px * baseline_mm)
                refinement_level = 0
                retry_count = 0
                status = "LEGACY_DISABLED"
                estimated_attainment = np.nan
            runtime_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "ablation_group": group,
                    "target_level": level,
                    "target_metric": "sigma",
                    "target_sigma_z_mm": target_z_mm,
                    "required_sigma_d_px": required_sigma_d,
                    "estimated_sigma_d_px": estimated_sigma_d,
                    "estimated_sigma_x_mm": None,
                    "estimated_sigma_y_mm": None,
                    "estimated_sigma_z_mm": estimated_sigma_z,
                    "vision_state": vision_state,
                    "search_radius_px": search_radius,
                    "refinement_level": refinement_level,
                    "precision_retry_count": retry_count,
                    "precision_status": status,
                    "runtime_ms": runtime_ms,
                    "valid_measurement_rate": float(np.mean(valid)),
                    "estimated_target_attainment_rate": estimated_attainment,
                    "gt_depth_rmse_mm": float(np.sqrt(np.mean(errors**2))),
                    "gt_depth_mae_mm": float(np.mean(np.abs(errors))),
                    "gt_target_attainment_rate": float(
                        np.mean(np.abs(errors) <= target_z_mm)
                    ),
                }
            )
    return pd.DataFrame(rows)


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
