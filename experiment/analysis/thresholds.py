from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def confidence_threshold_curve(
    rows: Iterable[Mapping[str, object]],
    *,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float | int]]:
    values = list(rows)
    levels = list(thresholds) if thresholds is not None else [round(value, 2) for value in np.linspace(0, 1, 21)]
    curve: list[dict[str, float | int]] = []
    for threshold in levels:
        accepted = [
            row for row in values
            if row.get("calibrated_confidence") is not None
            and float(row["calibrated_confidence"]) >= threshold
            and row.get("potential_error") is not None
            and np.isfinite(float(row["potential_error"]))
        ]
        errors = np.asarray([float(row["potential_error"]) for row in accepted], dtype=np.float64)
        count = len(accepted)
        curve.append({
            "confidence_threshold": float(threshold),
            "accepted": count,
            "total": len(values),
            "valid_rate": float(count / len(values)) if values else 0.0,
            "mae": float(np.mean(errors)) if count else float("nan"),
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if count else float("nan"),
            "bad_1": float(np.mean(errors > 1.0)) if count else float("nan"),
            "bad_3": float(np.mean(errors > 3.0)) if count else float("nan"),
            "cer_3": float(np.mean(errors > 3.0)) if count else float("nan"),
            "cer_5": float(np.mean(errors > 5.0)) if count else float("nan"),
            "cer_10": float(np.mean(errors > 10.0)) if count else float("nan"),
        })
    return curve

