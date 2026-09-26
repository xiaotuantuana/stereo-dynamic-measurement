from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def disparity_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if not np.any(valid):
        return {}
    error = np.abs(pred[valid] - gt[valid])
    return {
        "disparity_mae": float(np.mean(error)),
        "disparity_rmse": float(np.sqrt(np.mean(error ** 2))),
        "bad_1": float(np.mean(error > 1.0)),
        "bad_2": float(np.mean(error > 2.0)),
        "bad_3": float(np.mean(error > 3.0)),
    }


def depth_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if not np.any(valid):
        return {}
    error = np.abs(pred[valid] - gt[valid])
    return {
        "depth_mae": float(np.mean(error)),
        "depth_rmse": float(np.sqrt(np.mean(error ** 2))),
        "depth_median_ae": float(np.median(error)),
        "depth_p90": float(np.percentile(error, 90)),
        "depth_p95": float(np.percentile(error, 95)),
        "depth_max_error": float(np.max(error)),
        "depth_relative_error": float(np.mean(error / gt[valid])),
    }


def summarize_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, float | int]:
    values = list(rows)
    total = len(values)
    valid_count = sum(bool(row.get("valid")) for row in values)

    def numeric(field: str) -> np.ndarray:
        return np.asarray([
            float(row[field]) for row in values
            if row.get(field) is not None and np.isfinite(float(row[field]))
        ], dtype=np.float64)

    errors = numeric("disparity_error")
    runtimes = numeric("runtime_ms")
    summary: dict[str, float | int] = {
        "rows": total,
        "valid_rows": valid_count,
        "invalid_rows": total - valid_count,
        "valid_rate": float(valid_count / total) if total else 0.0,
        "failure_rate": float((total - valid_count) / total) if total else 0.0,
        "coverage": float(valid_count / total) if total else 0.0,
    }
    if errors.size:
        summary.update({
            "disparity_mae": float(np.mean(errors)),
            "disparity_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "valid_mae": float(np.mean(errors)),
            "valid_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "bad_1": float(np.mean(errors > 1.0)),
            "bad_2": float(np.mean(errors > 2.0)),
            "bad_3": float(np.mean(errors > 3.0)),
            "cer_3": float(np.mean(errors > 3.0)),
            "cer_5": float(np.mean(errors > 5.0)),
            "cer_10": float(np.mean(errors > 10.0)),
        })
    if runtimes.size:
        summary.update({
            "runtime_average_ms": float(np.mean(runtimes)),
            "runtime_median_ms": float(np.median(runtimes)),
            "fps": float(1000.0 / np.mean(runtimes)) if np.mean(runtimes) > 0 else 0.0,
        })
    return summary
