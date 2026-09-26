from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def _number(row: Mapping[str, object], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_gate(
    rows: Iterable[Mapping[str, object]],
    *,
    uniqueness_margin: float,
    confidence_threshold: float,
) -> dict[str, float | int]:
    values = list(rows)
    accepted_errors: list[float] = []
    for row in values:
        if row.get("confidence_source") != "initial_runtime_evidence":
            continue
        confidence = _number(row, "confidence")
        margin = _number(row, "uniqueness_margin")
        error = _number(row, "disparity_error")
        if confidence is None or margin is None or error is None:
            continue
        if confidence >= confidence_threshold and margin >= uniqueness_margin:
            accepted_errors.append(error)
    errors = np.asarray(accepted_errors, dtype=np.float64)
    total = len(values)
    result: dict[str, float | int] = {
        "uniqueness_margin": uniqueness_margin,
        "confidence_threshold": confidence_threshold,
        "rows": total,
        "accepted": int(errors.size),
        "coverage": float(errors.size / total) if total else 0.0,
    }
    if errors.size:
        result.update({
            "mae": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "bad_1": float(np.mean(errors > 1.0)),
            "bad_3": float(np.mean(errors > 3.0)),
            "cer_3": float(np.mean(errors > 3.0)),
            "cer_5": float(np.mean(errors > 5.0)),
            "cer_10": float(np.mean(errors > 10.0)),
        })
    return result


def sweep_gates(
    rows: Iterable[Mapping[str, object]],
    *,
    uniqueness_margins: Sequence[float],
    confidence_thresholds: Sequence[float],
) -> list[dict[str, float | int]]:
    values = list(rows)
    return [
        evaluate_gate(values, uniqueness_margin=margin, confidence_threshold=confidence)
        for margin in uniqueness_margins
        for confidence in confidence_thresholds
    ]
