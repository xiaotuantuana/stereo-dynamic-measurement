from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np


def summarize_sequence_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, float | int]:
    values = list(rows)
    total = len(values)
    valid = sum(bool(row.get("valid")) for row in values)
    errors = np.asarray([
        float(row["disparity_error"])
        for row in values
        if row.get("evaluable") and row.get("disparity_error") is not None
    ], dtype=float)
    runtimes = np.asarray([
        float(row["runtime_ms"])
        for row in values if row.get("runtime_ms") is not None
    ], dtype=float)
    trajectories: dict[str, list[float]] = defaultdict(list)
    for row in values:
        disparity = row.get("final_disparity")
        if row.get("valid") and disparity is not None:
            trajectories[str(row.get("point_id", "__all__"))].append(float(disparity))
    jumps = [
        abs(current - previous)
        for trajectory in trajectories.values()
        for previous, current in zip(trajectory, trajectory[1:])
    ]
    result: dict[str, float | int] = {
        "rows": total,
        "valid_rows": valid,
        "coverage": valid / total if total else 0.0,
        "tracking_loss_rate": (
            sum(row.get("status") == "lost" for row in values) / total if total else 0.0
        ),
        "frame_to_frame_jitter_px": float(np.mean(jumps)) if jumps else 0.0,
    }
    if errors.size:
        result.update({
            "valid_mae": float(np.mean(errors)),
            "valid_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "median_error": float(np.median(errors)),
            "p90_error": float(np.percentile(errors, 90)),
            "p95_error": float(np.percentile(errors, 95)),
            "bad_1": float(np.mean(errors > 1)),
            "bad_2": float(np.mean(errors > 2)),
            "bad_3": float(np.mean(errors > 3)),
            "cer_3": float(np.mean(errors > 3)),
            "cer_5": float(np.mean(errors > 5)),
            "cer_10": float(np.mean(errors > 10)),
        })
    if runtimes.size:
        result.update({
            "runtime_average_ms": float(np.mean(runtimes)),
            "runtime_median_ms": float(np.median(runtimes)),
        })
    return result
