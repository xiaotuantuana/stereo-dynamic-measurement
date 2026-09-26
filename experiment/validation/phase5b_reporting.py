"""Read-only reporting for independent static-stereo Phase 5B runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path


def static_mode_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for method, group in rows.groupby("method", sort=True):
        errors = group["disparity_error"].dropna().to_numpy(float)
        runtimes = group["runtime_ms"].dropna().to_numpy(float)
        output.append({
            "method": method,
            "points": len(group),
            "coverage": float(group["valid"].astype(bool).mean()),
            "disparity_mae": float(np.mean(errors)) if len(errors) else np.nan,
            "disparity_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.nan,
            "cer_3": float(np.mean(errors > 3.0)) if len(errors) else np.nan,
            "cer_5": float(np.mean(errors > 5.0)) if len(errors) else np.nan,
            "cer_10": float(np.mean(errors > 10.0)) if len(errors) else np.nan,
            "runtime_ms_per_sample": float(np.mean(runtimes)) if len(runtimes) else np.nan,
            "depth_xyz_3d": "N/A:no_calibration_or_depth_gt",
            "distance_error": "N/A:no_calibration_or_depth_gt",
        })
    return pd.DataFrame(output)


def write_phase5b_report(output_dir: str | Path, *, tests_passed: bool) -> dict[str, object]:
    output = Path(output_dir)
    frames = []
    for label, directory in (("M0", "M0"), ("M1", "M1_INNOVATION1")):
        path = output / directory / "metrics.jsonl"
        frame = pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        frame["method"] = label
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    summary = static_mode_summary(rows)
    ranges = rows.loc[rows["gt_disparity"].notna()].copy()
    ranges["disparity_range_px"] = pd.cut(ranges["gt_disparity"], [0, 5, 10, 20, np.inf], right=False)
    range_summary = ranges.groupby(["method", "disparity_range_px"], observed=False).agg(
        points=("sample_id", "size"), coverage=("valid", "mean"),
        disparity_mae=("disparity_error", "mean"),
        disparity_rmse=("disparity_error", lambda x: float(np.sqrt(np.mean(np.square(x.dropna())))) if x.notna().any() else np.nan),
    ).reset_index()
    failures = rows.groupby(["method", "status", "failure_reason"], dropna=False).size().reset_index(name="count")
    quality = rows.groupby("method").agg(
        confidence_mean=("confidence", "mean"), texture_mean=("texture_std", "mean"),
        match_cost_mean=("match_cost", "mean"), uniqueness_mean=("uniqueness_margin", "mean"),
    ).reset_index()
    m0, m1 = summary.set_index("method").loc["M0"], summary.set_index("method").loc["M1"]
    exceptions = int((rows["status"] == "exception").sum())
    nonfinite = int((~np.isfinite(rows.loc[rows["predicted_disparity"].notna(), "predicted_disparity"].astype(float))).sum())
    gates = pd.DataFrame([
        {"gate": "regression", "pass": tests_passed, "observed": "passed" if tests_passed else "not supplied"},
        {"gate": "dataset_gt_complete", "pass": bool(rows["gt_disparity"].notna().all()), "observed": int(rows["gt_disparity"].notna().sum())},
        {"gate": "no_crash", "pass": exceptions == 0, "observed": exceptions},
        {"gate": "no_nan_prediction", "pass": nonfinite == 0, "observed": nonfinite},
        {"gate": "m1_coverage_no_regression", "pass": m1.coverage >= m0.coverage, "observed": float(m1.coverage - m0.coverage)},
        {"gate": "m1_rmse_no_regression", "pass": m1.disparity_rmse <= m0.disparity_rmse, "observed": float(m1.disparity_rmse - m0.disparity_rmse)},
    ])
    allow = bool(gates["pass"].all())
    summary.to_csv(output / "mode_comparison.csv", index=False, encoding="utf-8-sig")
    range_summary.to_csv(output / "disparity_range_summary.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output / "failure_rejection_distribution.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(output / "matching_confidence_quality.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(output / "gate_summary.csv", index=False, encoding="utf-8-sig")
    payload = {"ALLOW_FULL_STATIC_BENCHMARK": "YES" if allow else "NO", "DEPTH_XYZ_DISTANCE": "N/A:no calibration/depth GT", "STATIC_TEMPORAL_CLAIMS": "PROHIBITED"}
    (output / "PHASE5B_5000_STATIC_BENCHMARK_REPORT.md").write_text(
        "# Phase 5B — 5000-Sample Large-Scale Stereo Benchmark\n\n```json\n"
        + json.dumps(payload, indent=2) + "\n```\n\n"
        + "All samples were independently initialized; no temporal state was retained. Depth, XYZ, 3D, and distance metrics are N/A because the static source supplies disparity GT but no calibration or depth GT.\n",
        encoding="utf-8",
    )
    return payload
