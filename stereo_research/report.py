from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import evaluate_rows, frame_runtime_values


def evaluate_experiment(
    experiment: str | Path,
    output: str | Path,
    ground_truth: str | Path | None = None,
) -> Path:
    experiment_path = Path(experiment).resolve()
    results_dir = experiment_path / "results"
    if not results_dir.exists():
        results_dir = experiment_path
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    gt_path = _discover_ground_truth(experiment_path, ground_truth)
    gt_rows = _read_csv(gt_path) if gt_path is not None else []
    methods: dict[str, dict[str, Any]] = {}
    raw_rows: dict[str, list[dict[str, str]]] = {}
    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name == "summary.csv":
            continue
        rows = _read_csv(csv_path)
        if not rows:
            continue
        method = str(rows[0].get("method") or csv_path.stem)
        methods[method] = evaluate_rows(rows, gt_rows)
        raw_rows[method] = rows
    if not methods:
        raise RuntimeError(f"No method CSV files found in {results_dir}")
    sgbm_median = (
        methods.get("sgbm", {}).get("runtime_median_ms")
        if "sgbm" in methods
        else methods.get("sgbm_fixed", {}).get("runtime_median_ms")
    )
    for method_summary in methods.values():
        method_median = method_summary.get("runtime_median_ms")
        method_summary["speedup_vs_sgbm"] = (
            float(sgbm_median) / float(method_median)
            if sgbm_median is not None and method_median not in (None, 0)
            else None
        )
    summary = {
        "experiment": str(experiment_path),
        "ground_truth": None if gt_path is None else str(gt_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
    }
    summary_json = output_path / "summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(output_path / "summary.csv", methods)
    _write_plots(output_path, raw_rows, methods, gt_rows)
    return summary_json


def _discover_ground_truth(
    experiment_path: Path,
    explicit: str | Path | None,
) -> Path | None:
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Ground truth does not exist: {path}")
        return path
    manifest_path = experiment_path / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = payload.get("ground_truth")
        if value:
            path = Path(value)
            path = path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
            if path.exists():
                return path
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_summary_csv(path: Path, methods: dict[str, dict[str, Any]]) -> None:
    keys: list[str] = []
    for summary in methods.values():
        for key, value in summary.items():
            if key == "status_counts" or isinstance(value, dict):
                continue
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for method, summary in methods.items():
            row = {key: summary.get(key) for key in keys}
            row["method"] = method
            writer.writerow(row)


def trajectory_series(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[list[int], list[float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if str(row.get("repeat", "0")) != "0" or row.get("status") != "valid":
            continue
        if row.get("Z_m") in (None, ""):
            continue
        point_id = str(row.get("point_id"))
        grouped.setdefault(point_id, []).append(
            (int(row["frame"]), float(row["Z_m"]))
        )
    series: dict[str, tuple[list[int], list[float]]] = {}
    for point_id, values in grouped.items():
        values.sort(key=lambda item: item[0])
        series[point_id] = (
            [item[0] for item in values],
            [item[1] for item in values],
        )
    return series


def _write_plots(
    output_dir: Path,
    rows_by_method: dict[str, list[dict[str, str]]],
    summaries: dict[str, dict[str, Any]],
    ground_truth: list[dict[str, str]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (output_dir / "PLOTS_SKIPPED.txt").write_text(
            "Install matplotlib to generate trajectory and runtime figures.\n",
            encoding="utf-8",
        )
        return
    (output_dir / "PLOTS_SKIPPED.txt").unlink(missing_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "savefig.bbox": "tight",
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

    def save(figure: Any, name: str) -> None:
        figure.savefig(output_dir / f"{name}.png", dpi=300)
        figure.savefig(output_dir / f"{name}.pdf")
        plt.close(figure)

    methods = list(summaries)
    runtimes = [frame_runtime_values(rows_by_method[method]) for method in methods]
    if any(runtimes):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.boxplot(runtimes, tick_labels=methods, showfliers=False)
        axis.set_ylabel("Algorithm time per frame (ms)")
        axis.set_title("Runtime comparison")
        axis.grid(axis="y", alpha=0.3)
        save(figure, "runtime_boxplot")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    has_trajectory = False
    for method, rows in rows_by_method.items():
        for point_id, (frames, depths) in trajectory_series(rows).items():
            has_trajectory = True
            axis.plot(
                frames,
                depths,
                label=f"{method}:{point_id}",
                linewidth=1.0,
            )
    axis.set_xlabel("Frame")
    axis.set_ylabel("Z (m)")
    axis.set_title("Depth trajectories")
    axis.grid(alpha=0.3)
    if has_trajectory:
        axis.legend()
    save(figure, "depth_trajectories")

    for axis_key, label, name in (
        ("delta_X_mm", r"$\Delta X$ (mm)", "delta_x_trajectories"),
        ("delta_Y_mm", r"$\Delta Y$ (mm)", "delta_y_trajectories"),
        ("delta_Z_mm", r"$\Delta Z$ (mm)", "delta_z_trajectories"),
    ):
        figure, axis = plt.subplots(figsize=(6.75, 2.8))
        plotted = False
        for method_index, (method, rows) in enumerate(rows_by_method.items()):
            grouped: dict[str, list[tuple[int, float]]] = {}
            for row in rows:
                if row.get("repeat", "0") != "0" or row.get("status") != "valid":
                    continue
                value = _float_or_none(row.get(axis_key))
                if value is None:
                    continue
                grouped.setdefault(str(row.get("point_id")), []).append((int(row["frame"]), value))
            for point_id, values in grouped.items():
                values.sort()
                axis.plot(
                    [value[0] for value in values],
                    [value[1] for value in values],
                    label=f"{method}:{point_id}",
                    color=colors[method_index % len(colors)],
                    linewidth=1.3,
                )
                plotted = True
        axis.set_xlabel("Frame")
        axis.set_ylabel(label)
        axis.set_title(f"{label} trajectories")
        if plotted:
            axis.legend(ncol=2)
        else:
            axis.text(0.5, 0.5, "No valid displacement data", ha="center", va="center", transform=axis.transAxes)
        save(figure, name)

    figure, axis = plt.subplots(figsize=(6.75, 2.8))
    plotted = False
    for method_index, (method, rows) in enumerate(rows_by_method.items()):
        valid = [row for row in rows if row.get("repeat", "0") == "0" and row.get("status") == "valid"]
        valid.sort(key=lambda row: (str(row.get("point_id")), int(row.get("frame", 0))))
        for point_id in sorted({str(row.get("point_id")) for row in valid}):
            point_rows = [row for row in valid if str(row.get("point_id")) == point_id]
            measured = [_float_or_none(row.get("measured_Z_m")) for row in point_rows]
            estimated = [_float_or_none(row.get("estimated_Z_m") or row.get("Z_m")) for row in point_rows]
            if not measured or measured[0] is None or not estimated or estimated[0] is None:
                continue
            frames = [int(row["frame"]) for row in point_rows]
            raw_delta = [(value - measured[0]) * 1000.0 for value in measured if value is not None]
            estimated_delta = [(value - estimated[0]) * 1000.0 for value in estimated if value is not None]
            if len(raw_delta) != len(frames) or len(estimated_delta) != len(frames):
                continue
            color = colors[method_index % len(colors)]
            axis.plot(frames, raw_delta, color=color, alpha=0.45, linewidth=1.0, label=f"{method}:{point_id} raw")
            axis.plot(frames, estimated_delta, color=color, linewidth=1.5, linestyle="--", label=f"{method}:{point_id} filtered")
            plotted = True
    axis.set_xlabel("Frame")
    axis.set_ylabel(r"$\Delta Z$ (mm)")
    axis.set_title("Raw vs. temporally estimated displacement")
    if plotted:
        axis.legend(ncol=2)
    else:
        axis.text(0.5, 0.5, "No paired raw/filtered data", ha="center", va="center", transform=axis.transAxes)
    save(figure, "raw_vs_filtered_displacement")

    _bar_plot(plt, save, summaries, "coverage_pct", "Coverage (%)", "coverage_comparison", colors)
    _bar_plot(plt, save, summaries, "false_match_rate_pct", "False matches (%)", "false_match_comparison", colors)
    _bar_plot(plt, save, summaries, "recovery_success_rate_pct", "Recovery success (%)", "recovery_comparison", colors)

    for field, ylabel, title, filename in (
        ("cycle_error_px", "Cycle error (px)", "Four-view cycle consistency", "cycle_error_over_time"),
        ("disparity_variance_px2", r"$\sigma_d^2$ (px$^2$)", "Adaptive disparity measurement variance", "kalman_measurement_variance"),
        ("kalman_gain_disparity", "Disparity gain", "Adaptive Kalman disparity gain", "kalman_gain_over_time"),
    ):
        figure, axis = plt.subplots(figsize=(6.75, 2.8))
        plotted = False
        for method_index, (method, rows) in enumerate(rows_by_method.items()):
            values = [
                (int(row["frame"]), value)
                for row in rows
                if str(row.get("repeat", "0")) == "0"
                and (value := _float_or_none(row.get(field))) is not None
            ]
            if values:
                values.sort()
                axis.plot(
                    [item[0] for item in values],
                    [item[1] for item in values],
                    label=method,
                    color=colors[method_index % len(colors)],
                    linewidth=1.2,
                )
                plotted = True
        axis.set_xlabel("Frame")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No applicable data", ha="center", va="center", transform=axis.transAxes)
        save(figure, filename)

    figure, axis = plt.subplots(figsize=(6.75, 2.8))
    plotted = False
    for method_index, (method, rows) in enumerate(rows_by_method.items()):
        point_rows = [
            row for row in rows
            if str(row.get("repeat", "0")) == "0" and row.get("status") == "valid"
        ]
        point_rows.sort(key=lambda row: int(row.get("frame", 0)))
        frames = [int(row["frame"]) for row in point_rows]
        measured = [_float_or_none(row.get("measured_disparity")) for row in point_rows]
        estimated = [_float_or_none(row.get("estimated_disparity")) for row in point_rows]
        if frames and all(value is not None for value in measured + estimated):
            color = colors[method_index % len(colors)]
            axis.plot(frames, measured, color=color, alpha=0.4, label=f"{method} measured")
            axis.plot(frames, estimated, color=color, linestyle="--", label=f"{method} filtered")
            plotted = True
    axis.set_xlabel("Frame")
    axis.set_ylabel("Disparity (px)")
    axis.set_title("Measured vs. adaptive-filtered disparity")
    if plotted:
        axis.legend(ncol=2)
    else:
        axis.text(0.5, 0.5, "No paired disparity data", ha="center", va="center", transform=axis.transAxes)
    save(figure, "measured_vs_filtered_disparity")

    figure, axis = plt.subplots(figsize=(6.75, 2.8))
    plotted = False
    for method_index, (method, rows) in enumerate(rows_by_method.items()):
        valid = [row for row in rows if str(row.get("repeat", "0")) == "0" and row.get("status") == "valid"]
        frames = [int(row["frame"]) for row in valid]
        raw = [_float_or_none(row.get("raw_delta_Z_mm")) for row in valid]
        compensated = [_float_or_none(row.get("compensated_delta_Z_mm")) for row in valid]
        color = colors[method_index % len(colors)]
        if frames and all(value is not None for value in raw):
            axis.plot(frames, raw, color=color, alpha=0.4, label=f"{method} raw")
            plotted = True
        if frames and all(value is not None for value in compensated):
            axis.plot(frames, compensated, color=color, linestyle="--", label=f"{method} compensated")
            plotted = True
    axis.set_xlabel("Frame")
    axis.set_ylabel(r"$\Delta Z$ (mm)")
    axis.set_title("Camera compensation before and after")
    if plotted:
        axis.legend(ncol=2)
    else:
        axis.text(0.5, 0.5, "No compensation data", ha="center", va="center", transform=axis.transAxes)
    save(figure, "camera_compensation_before_after")

    figure, axis = plt.subplots(figsize=(6.75, 2.8))
    plotted = False
    for method_index, (method, rows) in enumerate(rows_by_method.items()):
        values: list[tuple[int, float]] = []
        for row in rows:
            if str(row.get("repeat", "0")) != "0" or str(row.get("point_role")) != "reference":
                continue
            vector = [_float_or_none(row.get(f"compensated_delta_{axis_name}_mm")) for axis_name in ("X", "Y", "Z")]
            if all(value is not None for value in vector):
                values.append((int(row["frame"]), float(np.linalg.norm(vector))))
        if values:
            values.sort()
            axis.plot([item[0] for item in values], [item[1] for item in values], label=method, color=colors[method_index % len(colors)])
            plotted = True
    axis.set_xlabel("Frame")
    axis.set_ylabel("Reference residual (mm)")
    axis.set_title("Static reference residual after compensation")
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No reference compensation data", ha="center", va="center", transform=axis.transAxes)
    save(figure, "reference_compensation_residual")

    _bar_plot(plt, save, summaries, "icgn_disparity_rmse_px", "IC-GN disparity RMSE (px)", "ablation_disparity_rmse", colors)
    _bar_plot(plt, save, summaries, "estimated_xyz_rmse_mm", "Estimated 3D RMSE (mm)", "ablation_3d_rmse", colors)
    _bar_plot(plt, save, summaries, "jump_rate_pct", "Jump rate (%)", "ablation_jump_rate", colors)
    _bar_plot(plt, save, summaries, "effective_tracking_rate_pct", "Effective tracking (%)", "ablation_tracking_rate", colors)
    _bar_plot(plt, save, summaries, "runtime_median_ms", "Median runtime (ms)", "ablation_runtime", colors)
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    method_names = list(summaries)
    x_positions = np.arange(len(method_names), dtype=float)
    metric_specs = (
        ("integer_disparity_rmse_px", "integer"),
        ("subpixel_disparity_rmse_px", "original subpixel"),
        ("icgn_disparity_rmse_px", "IC-GN"),
    )
    width = 0.24
    for metric_index, (metric, label) in enumerate(metric_specs):
        values = [summaries[method].get(metric) for method in method_names]
        numeric = [0.0 if value is None else float(value) for value in values]
        axis.bar(
            x_positions + (metric_index - 1) * width,
            numeric,
            width=width,
            label=label,
            color=colors[metric_index % len(colors)],
        )
    axis.set_xticks(x_positions, method_names, rotation=20)
    axis.set_ylabel("Disparity RMSE (px)")
    axis.set_title("Integer, original subpixel, and IC-GN comparison")
    axis.legend()
    save(figure, "integer_subpixel_icgn_comparison")

    gt_map = {
        (int(row["frame"]), str(row["point_id"])): row
        for row in ground_truth
        if row.get("frame") not in (None, "")
    }
    figure, axis = plt.subplots(figsize=(6.75, 2.8))
    plotted = False
    for method_index, (method, rows) in enumerate(rows_by_method.items()):
        values: list[tuple[int, float]] = []
        for row in rows:
            if row.get("repeat", "0") != "0" or row.get("status") != "valid":
                continue
            gt = gt_map.get((int(row["frame"]), str(row["point_id"])))
            if gt is None:
                continue
            predicted = _float_or_none(row.get("measured_disparity") or row.get("disparity"))
            truth = _float_or_none(gt.get("gt_disparity"))
            if truth is None:
                left = _float_or_none(gt.get("left_x"))
                right = _float_or_none(gt.get("right_x"))
                truth = None if left is None or right is None else left - right
            if predicted is not None and truth is not None:
                values.append((int(row["frame"]), predicted - truth))
        if values:
            values.sort()
            axis.plot([value[0] for value in values], [value[1] for value in values], label=method, color=colors[method_index % len(colors)])
            plotted = True
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set_xlabel("Frame")
    axis.set_ylabel("Disparity error (px)")
    axis.set_title("Measured disparity error")
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Ground truth unavailable", ha="center", va="center", transform=axis.transAxes)
    save(figure, "disparity_error")

    if gt_map:
        figure, axis = plt.subplots(figsize=(6.75, 2.8))
        for method_index, (method, rows) in enumerate(rows_by_method.items()):
            pairs: list[tuple[int, float, float]] = []
            for row in rows:
                if row.get("repeat", "0") != "0" or row.get("status") != "valid":
                    continue
                gt = gt_map.get((int(row["frame"]), str(row["point_id"])))
                if gt is None:
                    continue
                predicted_z = _float_or_none(row.get("measured_Z_m") or row.get("Z_m"))
                truth_z = _float_or_none(gt.get("Z_m"))
                if predicted_z is not None and truth_z is not None:
                    pairs.append((int(row["frame"]), predicted_z, truth_z))
            if pairs:
                pairs.sort()
                color = colors[method_index % len(colors)]
                axis.plot([item[0] for item in pairs], [item[1] for item in pairs], color=color, label=f"{method} predicted")
                axis.plot([item[0] for item in pairs], [item[2] for item in pairs], color=color, linestyle="--", alpha=0.6, label=f"{method} GT")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Z (m)")
        axis.set_title("Predicted vs. ground-truth depth")
        axis.legend(ncol=2)
        save(figure, "predicted_vs_ground_truth")

        figure, axis = plt.subplots(figsize=(6.75, 2.8))
        for method_index, (method, rows) in enumerate(rows_by_method.items()):
            groups: dict[str, list[tuple[int, list[float], list[float]]]] = {}
            for row in rows:
                if row.get("repeat", "0") != "0" or row.get("status") != "valid":
                    continue
                gt = gt_map.get((int(row["frame"]), str(row["point_id"])))
                predicted = [_float_or_none(row.get(key)) for key in ("X_m", "Y_m", "Z_m")]
                truth = [] if gt is None else [_float_or_none(gt.get(key)) for key in ("X_m", "Y_m", "Z_m")]
                if gt is None or any(value is None for value in predicted + truth):
                    continue
                groups.setdefault(str(row["point_id"]), []).append((int(row["frame"]), predicted, truth))
            for point_id, values in groups.items():
                values.sort(key=lambda item: item[0])
                pred_ref = np.asarray(values[0][1], dtype=float)
                truth_ref = np.asarray(values[0][2], dtype=float)
                errors = [float(np.linalg.norm((np.asarray(pred) - pred_ref) - (np.asarray(truth) - truth_ref)) * 1000.0) for _frame, pred, truth in values]
                axis.plot([item[0] for item in values], errors, color=colors[method_index % len(colors)], label=f"{method}:{point_id}")
        axis.set_xlabel("Frame")
        axis.set_ylabel("3D displacement error (mm)")
        axis.set_title("Relative displacement error")
        axis.legend(ncol=2)
        save(figure, "displacement_error_trajectories")


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bar_plot(
    plt: Any,
    save: Any,
    summaries: dict[str, dict[str, Any]],
    metric: str,
    ylabel: str,
    name: str,
    colors: list[str],
) -> None:
    figure, axis = plt.subplots(figsize=(6.75, 2.8))
    methods = list(summaries)
    values = [summaries[method].get(metric) for method in methods]
    numeric = [0.0 if value is None else float(value) for value in values]
    bars = axis.bar(methods, numeric, color=[colors[index % len(colors)] for index in range(len(methods))])
    for bar, value in zip(bars, values):
        label = "N/A" if value is None else f"{float(value):.1f}"
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=7)
    axis.set_ylabel(ylabel)
    axis.set_title(ylabel)
    axis.tick_params(axis="x", rotation=20)
    save(figure, name)
