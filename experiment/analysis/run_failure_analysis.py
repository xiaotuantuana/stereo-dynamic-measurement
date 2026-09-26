from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stereo_research.global_matching import GlobalStereoMatcher
from stereo_research.local_matching import LocalMatcher
from stereo_research.models import MatcherConfig

from ..config import DatasetConfig
from ..datasets.index import iter_index
from ..datasets.io import read_image, read_pfm
from .failure_analysis import (
    CostCurveDiagnostics,
    calibrated_initial_confidence,
    classify_rejection,
    summarize_cost_curve,
)
from .thresholds import confidence_threshold_curve


DIAGNOSTIC_FIELDS = [
    "dataset", "sequence", "frame_id", "point_id", "pixel_x", "pixel_y",
    "gt_disparity", "raw_disparity", "refined_disparity", "final_disparity",
    "absolute_error", "potential_error", "baseline_confidence", "calibrated_confidence",
    "texture_std", "texture_score", "gradient_magnitude", "match_cost",
    "correct_match_cost", "best_candidate_disparity", "best_candidate_cost",
    "second_best_disparity", "second_best_cost", "uniqueness_margin",
    "uniqueness_score", "lr_consistency_error", "fb_error", "subpixel_offset",
    "search_min", "search_max", "candidate_count", "predicted_best_disagreement",
    "search_boundary", "gt_local_std", "brightness_delta", "reject_thresholds",
    "baseline_valid", "improved_valid", "rejection_quality", "failure_reason",
    "feature_low_texture", "feature_repetitive_texture", "feature_edge",
    "feature_disparity_discontinuity", "feature_search_boundary", "feature_lr_inconsistent",
    "root_cause", "sample_id", "left_path", "right_path",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cost_curve(
    matcher: LocalMatcher,
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    x: float,
    y: float,
    predicted: float,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, CostCurveDiagnostics, float]:
    left_patch = matcher._extract_patch(left_gray, (x, y))
    if left_patch is None:
        raise ValueError("Left analysis patch is out of bounds")
    disparities: list[float] = []
    costs: list[float] = []
    for disparity in range(1, maximum + 1):
        right_patch = matcher._extract_patch(right_gray, (x - disparity, y))
        if right_patch is None:
            continue
        disparities.append(float(disparity))
        costs.append(float(matcher._photo_cost(left_patch, right_patch)))
    disparity_array = np.asarray(disparities, dtype=np.float64)
    cost_array = np.asarray(costs, dtype=np.float64)
    return disparity_array, cost_array, summarize_cost_curve(
        disparity_array, cost_array, predicted
    ), float(np.std(left_patch))


def _sample_gt(gt: np.ndarray, x: float, y: float) -> float | None:
    ix, iy = int(round(x)), int(round(y))
    if not (0 <= iy < gt.shape[0] and 0 <= ix < gt.shape[1]):
        return None
    value = float(gt[iy, ix])
    return value if np.isfinite(value) and value > 0 else None


def _forced_disparity(base: np.ndarray, expanded: np.ndarray, x: float, y: float, base_limit: int) -> float | None:
    ix, iy = int(round(x)), int(round(y))
    value = float(base[iy, ix])
    if not np.isfinite(value) or value <= 0 or value >= base_limit - 1.5:
        value = float(expanded[iy, ix])
    return value if np.isfinite(value) and value > 0 else None


def _root_cause(row: dict[str, Any]) -> str:
    causes: list[str] = []
    if row["feature_disparity_discontinuity"]:
        causes.append("DISPARITY_DISCONTINUITY_OR_OCCLUSION")
    if row["feature_low_texture"]:
        causes.append("WEAK_TEXTURE")
    if row["feature_repetitive_texture"]:
        causes.append("AMBIGUOUS_REPETITIVE_MATCH")
    if row["feature_search_boundary"]:
        causes.append("SEARCH_BOUNDARY")
    if row["feature_lr_inconsistent"]:
        causes.append("LR_INCONSISTENT")
    return "+".join(causes) or "NO_SINGLE_RUNTIME_SIGNATURE"


def _diagnostic_figure(
    row: dict[str, Any],
    left: np.ndarray,
    right: np.ndarray,
    gt: np.ndarray,
    disparities: np.ndarray,
    costs: np.ndarray,
    output: Path,
) -> None:
    x, y = float(row["pixel_x"]), float(row["pixel_y"])
    predicted, truth = float(row["final_disparity"]), float(row["gt_disparity"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(left, cmap="gray")
    axes[0, 0].scatter([x], [y], c="yellow", s=45)
    axes[0, 0].set_title("Left target")
    axes[0, 1].imshow(right, cmap="gray")
    axes[0, 1].scatter([x - truth], [y], c="lime", s=45, label="GT match")
    axes[0, 1].scatter([x - predicted], [y], c="red", marker="x", s=60, label="Predicted")
    axes[0, 1].axhline(y, color="cyan", linewidth=0.5)
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].set_title("Right: correct vs predicted")
    finite_gt = np.where(np.isfinite(gt) & (gt > 0), gt, np.nan)
    image = axes[0, 2].imshow(finite_gt, cmap="viridis")
    axes[0, 2].scatter([x], [y], c="red", s=35)
    axes[0, 2].set_title("GT disparity")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046)
    radius = 45
    xi, yi = int(round(x)), int(round(y))
    axes[1, 0].imshow(left[max(0, yi-radius):yi+radius, max(0, xi-radius):xi+radius], cmap="gray")
    axes[1, 0].set_title("Left context")
    center = int(round(x - predicted))
    axes[1, 1].imshow(right[max(0, yi-radius):yi+radius, max(0, center-radius):center+radius], cmap="gray")
    axes[1, 1].set_title("Predicted right context")
    axes[1, 2].plot(disparities, costs, color="black", linewidth=1)
    axes[1, 2].axvline(truth, color="green", label=f"GT {truth:.2f}")
    axes[1, 2].axvline(predicted, color="red", label=f"Pred {predicted:.2f}")
    axes[1, 2].set_xlabel("Candidate disparity (px)")
    axes[1, 2].set_ylabel("Existing photo cost")
    axes[1, 2].set_title(f"Cost curve; margin={row['uniqueness_margin']:.4f}")
    axes[1, 2].legend(fontsize=8)
    for axis in axes.ravel()[:5]:
        axis.set_axis_off()
    fig.suptitle(
        f"{row['sample_id']} {row['point_id']} | error={row['absolute_error']:.3f}px | {row['root_cause']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_thresholds(curve: list[dict[str, float | int]], figures: Path) -> None:
    thresholds = np.asarray([row["confidence_threshold"] for row in curve], dtype=float)
    valid = np.asarray([row["valid_rate"] for row in curve], dtype=float)
    rmse = np.asarray([row["rmse"] for row in curve], dtype=float)
    bad3 = np.asarray([row["bad_3"] for row in curve], dtype=float)
    for name, values, ylabel in (
        ("confidence_vs_valid_rate", valid, "Valid rate"),
        ("confidence_vs_rmse", rmse, "RMSE (px)"),
        ("confidence_vs_bad3", bad3, "bad-3"),
    ):
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(thresholds, values, marker="o", markersize=3)
        axis.set_xlabel("Confidence threshold")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=150)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(valid, rmse, marker="o", markersize=3)
    axis.set_xlabel("Valid rate")
    axis.set_ylabel("RMSE (px)")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "valid_rate_vs_rmse.png", dpi=150)
    plt.close(fig)


def run_failure_analysis(
    config: DatasetConfig,
    baseline_dir: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    baseline = Path(baseline_dir)
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (baseline / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    samples = {sample.sample_id: sample for sample in iter_index(config.index_path, config.dataset_root)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_rows:
        grouped[row["sample_id"]].append(row)
    matcher_config = MatcherConfig()
    global_matcher = GlobalStereoMatcher(matcher_config)
    local_matcher = LocalMatcher(matcher_config)
    diagnostics: list[dict[str, Any]] = []
    figure_payloads: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sample_id, rows in grouped.items():
        sample = samples[sample_id]
        left = read_image(sample.left_path, cv2.IMREAD_GRAYSCALE)
        right = read_image(sample.right_path, cv2.IMREAD_GRAYSCALE)
        gt = read_pfm(sample.disparity_gt_path) * sample.disparity_scale
        base = global_matcher.compute(left, right, matcher_config.num_disparities)
        expanded = global_matcher.compute(left, right, matcher_config.expanded_num_disparities)
        gradient_x = cv2.Sobel(left, cv2.CV_32F, 1, 0)
        gradient_y = cv2.Sobel(left, cv2.CV_32F, 0, 1)
        for baseline_row in rows:
            x, y = float(baseline_row["left_x"]), float(baseline_row["left_y"])
            gt_disparity = _sample_gt(gt, x, y)
            baseline_prediction = (
                float(baseline_row["predicted_disparity"])
                if baseline_row["predicted_disparity"] else None
            )
            forced = baseline_prediction or _forced_disparity(
                base.left_disparity, expanded.left_disparity, x, y, base.num_disparities
            )
            if forced is None or gt_disparity is None:
                sparse = {field: None for field in DIAGNOSTIC_FIELDS}
                sparse.update({
                    "dataset": sample.dataset_name, "sequence": sample.sequence_name,
                    "frame_id": sample.frame_id, "point_id": baseline_row["point_id"],
                    "pixel_x": x, "pixel_y": y, "gt_disparity": gt_disparity,
                    "baseline_confidence": float(baseline_row["confidence"] or 0.0),
                    "baseline_valid": baseline_row["valid"].lower() == "true",
                    "improved_valid": False, "rejection_quality": "Unassessable",
                    "failure_reason": baseline_row["failure_reason"] or "no_runtime_candidate",
                    "root_cause": "NO_RUNTIME_CANDIDATE", "sample_id": sample_id,
                    "left_path": str(sample.left_path), "right_path": str(sample.right_path),
                })
                diagnostics.append(sparse)
                continue
            curve_d, curve_c, curve, texture_std = _cost_curve(
                local_matcher, left, right, x, y, forced,
                min(matcher_config.expanded_num_disparities - 1, int(x - matcher_config.patch_size // 2 - 1)),
            )
            global_sample = global_matcher.sample_initial(expanded, (x, y))
            lr_error = global_sample.lr_error_px
            confidence = calibrated_initial_confidence(
                texture_std=texture_std,
                lr_error_px=lr_error,
                diagnostics=curve,
                texture_reference=matcher_config.uncertainty_texture_reference,
                margin_reference=matcher_config.uniqueness_margin,
                lr_reference=matcher_config.lr_threshold,
                max_photo_cost=matcher_config.max_photo_cost,
            )
            potential_error = abs(forced - gt_disparity)
            ix, iy = int(round(x)), int(round(y))
            gt_patch = gt[max(0, iy-3):iy+4, max(0, ix-3):ix+4]
            gt_local_std = float(np.nanstd(gt_patch))
            gradient = float(np.hypot(gradient_x[iy, ix], gradient_y[iy, ix]))
            correct_index = int(np.argmin(np.abs(curve_d - gt_disparity)))
            predicted_right = local_matcher._extract_patch(right, (x - forced, y))
            left_patch = local_matcher._extract_patch(left, (x, y))
            brightness_delta = (
                None if predicted_right is None or left_patch is None
                else abs(float(np.mean(left_patch)) - float(np.mean(predicted_right)))
            )
            baseline_valid = baseline_row["valid"].lower() == "true"
            improved_valid = (
                baseline_valid
                and confidence >= matcher_config.confidence_medium_threshold
                and curve.uniqueness_margin is not None
                and curve.uniqueness_margin >= 0.03
            )
            diagnostic: dict[str, Any] = {
                "dataset": sample.dataset_name, "sequence": sample.sequence_name,
                "frame_id": sample.frame_id, "point_id": baseline_row["point_id"],
                "pixel_x": x, "pixel_y": y, "gt_disparity": gt_disparity,
                "raw_disparity": forced, "refined_disparity": None,
                "final_disparity": forced, "absolute_error": potential_error if baseline_valid else None,
                "potential_error": potential_error,
                "baseline_confidence": float(baseline_row["confidence"] or 0.0),
                "calibrated_confidence": confidence, "texture_std": texture_std,
                "texture_score": min(texture_std / matcher_config.uncertainty_texture_reference, 1.0),
                "gradient_magnitude": gradient, "match_cost": curve.predicted_cost,
                "correct_match_cost": float(curve_c[correct_index]),
                "best_candidate_disparity": curve.best_disparity,
                "best_candidate_cost": curve.best_cost,
                "second_best_disparity": curve.second_best_disparity,
                "second_best_cost": curve.second_best_cost,
                "uniqueness_margin": curve.uniqueness_margin,
                "uniqueness_score": min((curve.uniqueness_margin or 0.0) / matcher_config.uniqueness_margin, 1.0),
                "lr_consistency_error": lr_error, "fb_error": None,
                "subpixel_offset": forced - round(forced), "search_min": int(curve_d.min()),
                "search_max": int(curve_d.max()), "candidate_count": curve.candidate_count,
                "predicted_best_disagreement": curve.predicted_best_disagreement,
                "search_boundary": min(abs(forced-curve_d.min()), abs(forced-curve_d.max())),
                "gt_local_std": gt_local_std, "brightness_delta": brightness_delta,
                "reject_thresholds": json.dumps({
                    "min_texture_std": matcher_config.min_texture_std,
                    "uniqueness_margin": matcher_config.uniqueness_margin,
                    "lr_threshold": matcher_config.lr_threshold,
                    "confidence_medium_threshold": matcher_config.confidence_medium_threshold,
                    "analysis_margin_gate": 0.03,
                }),
                "baseline_valid": baseline_valid, "improved_valid": improved_valid,
                "rejection_quality": "Accepted" if baseline_valid else classify_rejection(potential_error),
                "failure_reason": baseline_row["failure_reason"],
                "feature_low_texture": texture_std < matcher_config.min_texture_std or gradient < 5.0,
                "feature_repetitive_texture": (curve.uniqueness_margin or 0.0) < matcher_config.uniqueness_margin,
                "feature_edge": gradient > 30.0,
                "feature_disparity_discontinuity": gt_local_std > 2.0,
                "feature_search_boundary": min(abs(forced-curve_d.min()), abs(forced-curve_d.max())) <= 1.5,
                "feature_lr_inconsistent": lr_error is None or lr_error > matcher_config.lr_threshold,
                "sample_id": sample_id, "left_path": str(sample.left_path), "right_path": str(sample.right_path),
            }
            diagnostic["root_cause"] = _root_cause(diagnostic)
            diagnostics.append(diagnostic)
            if baseline_valid:
                figure_payloads[(sample_id, baseline_row["point_id"])] = (left, right, gt, curve_d, curve_c)
    diagnostics.sort(key=lambda row: float(row.get("absolute_error") or -1.0), reverse=True)
    _write_csv(output / "diagnostics.csv", diagnostics, DIAGNOSTIC_FIELDS)
    catastrophic = [row for row in diagnostics if row["baseline_valid"] and float(row["potential_error"]) > 3.0]
    _write_csv(output / "catastrophic_failure_analysis.csv", catastrophic, DIAGNOSTIC_FIELDS)
    rejections = [row for row in diagnostics if not row["baseline_valid"]]
    _write_csv(output / "rejection_analysis.csv", rejections, DIAGNOSTIC_FIELDS)
    curve = confidence_threshold_curve(diagnostics)
    _write_csv(output / "confidence_threshold_curve.csv", curve, list(curve[0]))
    _plot_thresholds(curve, figures)
    for rank, row in enumerate([item for item in diagnostics if item["baseline_valid"]][:top_n], start=1):
        payload = figure_payloads[(row["sample_id"], row["point_id"])]
        _diagnostic_figure(row, *payload, figures / f"top_{rank:02d}_{row['frame_id']}_{row['point_id']}.png")
    causes = Counter(row["root_cause"] for row in catastrophic)
    rejection_counts = Counter(row["rejection_quality"] for row in rejections)
    summary = {
        "diagnosed_points": len(diagnostics),
        "baseline_valid_points": sum(bool(row["baseline_valid"]) for row in diagnostics),
        "catastrophic_gt_3": len(catastrophic),
        "catastrophic_gt_10": sum(float(row["potential_error"]) > 10 for row in catastrophic),
        "root_causes": dict(causes),
        "rejection_quality": dict(rejection_counts),
        "confidence_source_baseline": "hardcoded 1.0 during TemporalStereoPipeline.initialize",
        "confidence_source_analysis": "runtime-only conservative minimum of texture/photo/uniqueness/LR/global-local agreement",
        "ground_truth_used_for_decision": False,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
