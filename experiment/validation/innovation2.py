from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stereo_dynamic_measurement.innovation2.physics_validation import run_runtime_physics_analysis

from ..simulation.temporal_scenarios import TemporalScenario, build_temporal_scenarios


def _error_jitter(group: pd.DataFrame, prefix: str) -> float:
    residual = group[f"{prefix}_Y_mm"].to_numpy(float) - group["Y_gt_mm"].to_numpy(float)
    return float(np.std(np.diff(residual))) if len(residual) > 1 else 0.0


def _lag_frames(group: pd.DataFrame, prefix: str, maximum: int = 5) -> int:
    truth = group["Y_gt_mm"].to_numpy(float)
    measured = group[f"{prefix}_Y_mm"].to_numpy(float)
    if len(truth) < 4 or np.std(truth) < 1e-9:
        return 0
    candidates: list[tuple[float, int]] = []
    for lag in range(-maximum, maximum + 1):
        first = measured[max(0, lag):len(measured) + min(0, lag)]
        second = truth[max(0, -lag):len(truth) - max(0, lag)]
        candidates.append((float(np.mean((first - second) ** 2)), lag))
    return min(candidates)[1]


def _evaluate_scenario(
    scenario: TemporalScenario,
    *,
    output_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runtime_input = scenario.raw_measurements[[
        "frame", "timestamp_s", "point_id", "raw_X_mm", "raw_Y_mm", "raw_Z_mm",
        "flow_dx_mm", "flow_dy_mm", "flow_dz_mm",
    ]].copy()
    started = time.perf_counter()
    runtime = run_runtime_physics_analysis(runtime_input, fs_hz=30.0, threshold=0.65)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    merged = runtime.trajectories.merge(
        scenario.ground_truth, on=["frame", "point_id"], how="inner", validate="one_to_one"
    )
    raw = merged[["raw_X_mm", "raw_Y_mm", "raw_Z_mm"]].to_numpy(float)
    corrected = merged[["corrected_X_mm", "corrected_Y_mm", "corrected_Z_mm"]].to_numpy(float)
    truth = merged[["X_gt_mm", "Y_gt_mm", "Z_gt_mm"]].to_numpy(float)
    raw_error = np.linalg.norm(raw - truth, axis=1)
    corrected_error = np.linalg.norm(corrected - truth, axis=1)
    merged["raw_error_mm"] = raw_error
    merged["corrected_error_mm"] = corrected_error
    merged["is_injected_fault"] = [
        (int(row.frame), str(row.point_id)) in scenario.fault_keys
        for row in merged.itertuples(index=False)
    ]
    nonfault = ~merged["is_injected_fault"]
    fault = merged["is_injected_fault"]
    raw_fault_total = float(merged.loc[fault, "raw_error_mm"].sum())
    corrected_fault_total = float(merged.loc[fault, "corrected_error_mm"].sum())
    jump_suppression = (
        None if raw_fault_total <= 0 else 1.0 - corrected_fault_total / raw_fault_total
    )
    raw_jump_count = int((merged["raw_error_mm"] > 10.0).sum())
    corrected_jump_count = int((merged["corrected_error_mm"] > 10.0).sum())
    jump_count_suppression = (
        None if raw_jump_count == 0
        else (raw_jump_count - corrected_jump_count) / raw_jump_count
    )
    recovery_values: list[int] = []
    for point_id in sorted({point for _, point in scenario.fault_keys}):
        final_fault_frame = max(frame for frame, point in scenario.fault_keys if point == point_id)
        later = merged[(merged.point_id == point_id) & (merged.frame > final_fault_frame)].sort_values("frame")
        recovered = later[later.corrected_error_mm <= 3.0]
        if not recovered.empty:
            recovery_values.append(int(recovered.iloc[0].frame) - final_fault_frame)
    false_corrections = merged[nonfault & merged["correction_applied"].astype(bool)].copy()
    false_correction_rate = float(len(false_corrections) / max(int(nonfault.sum()), 1))
    raw_jitter = float(np.mean([
        _error_jitter(group.sort_values("frame"), "raw") for _, group in merged.groupby("point_id")
    ]))
    corrected_jitter = float(np.mean([
        _error_jitter(group.sort_values("frame"), "corrected") for _, group in merged.groupby("point_id")
    ]))
    corrected_lag = float(np.mean([
        abs(_lag_frames(group.sort_values("frame"), "corrected")) for _, group in merged.groupby("point_id")
    ]))
    confidence = runtime.confidence.copy()
    confidence.insert(0, "scenario", scenario.name)
    merged.insert(0, "scenario", scenario.name)
    false_corrections.insert(0, "scenario", scenario.name)
    metrics: dict[str, object] = {
        "scenario": scenario.name,
        "provenance": scenario.provenance,
        "expected_rows": scenario.expected_rows,
        "observed_rows": len(merged),
        "coverage": len(merged) / scenario.expected_rows,
        "raw_mae_mm": float(np.mean(raw_error)),
        "corrected_mae_mm": float(np.mean(corrected_error)),
        "raw_rmse_mm": float(np.sqrt(np.mean(raw_error ** 2))),
        "corrected_rmse_mm": float(np.sqrt(np.mean(corrected_error ** 2))),
        "raw_jitter_mm": raw_jitter,
        "corrected_jitter_mm": corrected_jitter,
        "jitter_improvement_mm": raw_jitter - corrected_jitter,
        "jump_suppression_rate": jump_suppression,
        "raw_jump_count": raw_jump_count,
        "corrected_jump_count": corrected_jump_count,
        "jump_count_suppression_rate": jump_count_suppression,
        "recovery_frames": float(np.mean(recovery_values)) if recovery_values else None,
        "false_correction_rate": false_correction_rate,
        "correction_rate": float(merged["correction_applied"].astype(bool).mean()),
        "lag_frames": corrected_lag,
        "c_phy_valid_rate": float(confidence["C_phy_valid"].astype(bool).mean()),
        "runtime_ms": runtime_ms,
        "legitimate_motion": scenario.legitimate_motion,
    }
    return metrics, merged, confidence, false_corrections


def run_innovation2_controlled_validation(
    output_dir: str | Path,
    *,
    frame_count: int = 96,
    seed: int = 20260827,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    trajectories: list[pd.DataFrame] = []
    confidence_rows: list[pd.DataFrame] = []
    false_cases: list[pd.DataFrame] = []
    for scenario in build_temporal_scenarios(frame_count=frame_count, seed=seed).values():
        metrics, trajectory, confidence, false_correction = _evaluate_scenario(scenario, output_dir=output)
        metrics_rows.append(metrics)
        trajectories.append(trajectory)
        confidence_rows.append(confidence)
        if not false_correction.empty:
            false_cases.append(false_correction)
    metrics_table = pd.DataFrame(metrics_rows)
    trajectory_table = pd.concat(trajectories, ignore_index=True)
    confidence_table = pd.concat(confidence_rows, ignore_index=True)
    false_table = pd.concat(false_cases, ignore_index=True) if false_cases else pd.DataFrame(columns=["scenario", "frame", "point_id"])
    metrics_table.to_csv(output / "scenario_metrics.csv", index=False, encoding="utf-8-sig")
    trajectory_table.to_csv(output / "raw_vs_corrected.csv", index=False, encoding="utf-8-sig")
    confidence_table.to_csv(output / "physics_evidence.csv", index=False, encoding="utf-8-sig")
    false_table.to_csv(output / "false_correction_cases.csv", index=False, encoding="utf-8-sig")
    applied = trajectory_table[trajectory_table["correction_applied"].astype(bool)].copy()
    applied["error_improvement_mm"] = applied["raw_error_mm"] - applied["corrected_error_mm"]
    applied[applied.error_improvement_mm > 0].sort_values("error_improvement_mm", ascending=False).head(20).to_csv(
        output / "successful_corrections.csv", index=False, encoding="utf-8-sig"
    )
    applied[applied.error_improvement_mm <= 0].sort_values("error_improvement_mm").head(20).to_csv(
        output / "failed_corrections.csv", index=False, encoding="utf-8-sig"
    )
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    metadata = {
        "experiment": "Innovation2 CONTROLLED temporal validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame_count": frame_count, "seed": seed, "git_commit": git_commit,
        "git_dirty": git_dirty, "python": sys.version, "platform": platform.platform(),
        "runtime_function": "run_runtime_physics_analysis",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary: dict[str, object] = {
        "scenario_count": len(metrics_table),
        "provenance": "CONTROLLED",
        "aggregate_raw_rmse_mm": float(metrics_table["raw_rmse_mm"].mean()),
        "aggregate_corrected_rmse_mm": float(metrics_table["corrected_rmse_mm"].mean()),
        "aggregate_false_correction_rate": float(metrics_table["false_correction_rate"].mean()),
        "aggregate_jitter_improvement_mm": float(metrics_table["jitter_improvement_mm"].mean()),
        "ground_truth_online_access": False,
        "available_runtime_evidence": ["flow_3d", "temporal", "spatial", "spectral"],
        "unavailable_runtime_evidence": ["phase", "coherence"],
        "final_pipeline_write_enabled": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
