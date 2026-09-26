from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from stereo_research.models import MethodName, PointSpec

from ..simulation.controlled_stereo import create_controlled_stereo_sequence
from .runner import run_sequences


def _controlled_q() -> np.ndarray:
    return np.asarray([
        [1.0, 0.0, 0.0, -160.0],
        [0.0, 1.0, 0.0, -90.0],
        [0.0, 0.0, 0.0, 100.0],
        [0.0, 0.0, 10.0, 0.0],
    ], dtype=float)


def run_controlled_ablation(
    output_dir: str | Path,
    *,
    frame_count: int = 12,
    methods: tuple[MethodName, ...] = ("M0", "M1", "M2", "M3"),
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequence = create_controlled_stereo_sequence(output / "controlled_sequence", frame_count=frame_count)
    points = (
        PointSpec("P1", (160.0, 60.0)),
        PointSpec("P2", (200.0, 90.0)),
        PointSpec("P3", (240.0, 120.0)),
    )
    summaries: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for method in methods:
        summary = run_sequences(
            [sequence], output / method, method=method, q=_controlled_q(),
            calibration_unit="m", points=points,
        )
        summaries[method] = summary
        rows.append({
            "method": method,
            "provenance": "CONTROLLED",
            "frames": frame_count,
            "coverage": summary.get("coverage"),
            "valid_mae": summary.get("valid_mae"),
            "valid_rmse": summary.get("valid_rmse"),
            "bad_1": summary.get("bad_1"),
            "bad_2": summary.get("bad_2"),
            "bad_3": summary.get("bad_3"),
            "cer_3": summary.get("cer_3"),
            "cer_5": summary.get("cer_5"),
            "cer_10": summary.get("cer_10"),
            "median_error": summary.get("median_error"),
            "p90_error": summary.get("p90_error"),
            "p95_error": summary.get("p95_error"),
            "jitter_px": summary.get("frame_to_frame_jitter_px"),
            "tracking_loss_rate": summary.get("tracking_loss_rate"),
            "runtime_median_ms": summary.get("runtime_median_ms"),
        })
    with (output / "ablation_innovation1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lifecycle = {method: summary["lifecycle"] for method, summary in summaries.items()}
    payload: dict[str, object] = {
        "methods": list(methods),
        "provenance": "CONTROLLED",
        "stateful_protocol": all(
            item["initialize_calls"] == 1 and item["step_calls"] == frame_count - 1
            for item in lifecycle.values()
        ),
        "lifecycle": lifecycle,
        "ground_truth_online_access": False,
    }
    (output / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
