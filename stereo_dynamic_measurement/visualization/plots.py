from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectories(samples: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for point_id, group in samples.groupby("point_id", sort=True):
        for axis, coordinate in zip(axes, ("X", "Y", "Z"), strict=True):
            axis.plot(group["time_s"], group[f"{coordinate}_gt_mm"], label=point_id)
            axis.set_ylabel(f"{coordinate} (mm)")
            axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Ground-truth four-point trajectories")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_error_components(samples: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for point_id, group in samples.groupby("point_id", sort=True):
        for axis, coordinate in zip(axes, ("x", "y", "z"), strict=True):
            axis.plot(group["time_s"], group[f"error_{coordinate}_mm"], label=point_id)
            axis.set_ylabel(f"Error {coordinate.upper()} (mm)")
            axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Stereo reconstruction error components")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
