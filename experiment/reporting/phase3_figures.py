from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_phase3_figures(
    *,
    output_dir: str | Path,
    innovation1_csv: Path,
    innovation2_csv: Path,
    innovation3_confusion: Path,
    occlusion_csv: Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    i1 = pd.read_csv(innovation1_csv)
    fig, left = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(i1))
    left.bar(x - 0.18, i1.coverage, width=0.36, label="Coverage")
    left.set_ylabel("Coverage")
    right = left.twinx()
    right.bar(x + 0.18, i1.valid_rmse, width=0.36, color="#d95f02", label="RMSE")
    right.set_ylabel("Valid RMSE (px)")
    left.set_xticks(x, i1.method)
    left.set_title("Innovation1 Stateful Ablation (CONTROLLED)")
    fig.tight_layout()
    paths["innovation1_ablation"] = output / "innovation1_ablation.png"
    fig.savefig(paths["innovation1_ablation"], dpi=180)
    plt.close(fig)

    i2 = pd.read_csv(innovation2_csv)
    chosen = i2[(i2.scenario == "single_jump") & (i2.point_id == "P2")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(chosen.frame, chosen.raw_Y_mm, label="Raw", alpha=0.75)
    ax.plot(chosen.frame, chosen.corrected_Y_mm, label="Corrected candidate")
    ax.plot(chosen.frame, chosen.Y_gt_mm, label="GT (offline)", linestyle="--")
    ax.set_title("Innovation2 Single Jump (CONTROLLED)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Y (mm)"); ax.legend()
    fig.tight_layout()
    paths["innovation2_single_jump"] = output / "innovation2_single_jump.png"
    fig.savefig(paths["innovation2_single_jump"], dpi=180)
    plt.close(fig)

    confusion = pd.read_csv(innovation3_confusion, index_col=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(confusion.to_numpy(float), cmap="Blues")
    ax.set_xticks(np.arange(len(confusion.columns)), confusion.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(confusion.index)), confusion.index)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Injected"); ax.set_title("Innovation3 Confusion Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    paths["innovation3_confusion"] = output / "innovation3_confusion_matrix.png"
    fig.savefig(paths["innovation3_confusion"], dpi=180)
    plt.close(fig)

    occ = pd.read_csv(occlusion_csv)
    valid = occ.dropna(subset=["disparity_error"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for point_id, group in valid.groupby("point_id"):
        ax.plot(group.sequence_frame_index, group.disparity_error, marker=".", label=point_id)
    ax.axhline(10.0, color="red", linestyle="--", label="CER@10 threshold")
    ax.set_title("Occlusion / Discontinuity Error (CONTROLLED)")
    ax.set_xlabel("Sequence frame"); ax.set_ylabel("Absolute disparity error (px)"); ax.legend(fontsize=8)
    fig.tight_layout()
    paths["occlusion_errors"] = output / "occlusion_discontinuity_errors.png"
    fig.savefig(paths["occlusion_errors"], dpi=180)
    plt.close(fig)
    return paths
