from __future__ import annotations

import csv
import json
from pathlib import Path

from stereo_research.models import PointSpec

from ..simulation.controlled_stereo import create_occlusion_discontinuity_sequence
from .ablation import _controlled_q
from .runner import run_sequences


def run_occlusion_diagnostics(
    output_dir: str | Path,
    *,
    frame_count: int = 18,
) -> dict[str, object]:
    output = Path(output_dir)
    sequence = create_occlusion_discontinuity_sequence(output / "controlled_sequence", frame_count=frame_count)
    points = (
        PointSpec("background", (100.0, 90.0)),
        PointSpec("edge_background", (158.0, 90.0)),
        PointSpec("foreground", (190.0, 90.0)),
        PointSpec("edge_foreground", (228.0, 90.0)),
    )
    run = output / "M3"
    summary = run_sequences(
        [sequence], run, method="M3", q=_controlled_q(), calibration_unit="m", points=points,
    )
    with (run / "frame_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    diagnostics: list[dict[str, object]] = []
    for row in source:
        error = None if row["disparity_error"] == "" else float(row["disparity_error"])
        diagnostics.append({
            **row,
            "near_discontinuity_offline_label": row["point_id"].startswith("edge_"),
            "catastrophic_gt_3": None if error is None else error > 3.0,
            "catastrophic_gt_10": None if error is None else error > 10.0,
        })
    diagnostics.sort(
        key=lambda row: -1.0 if row["disparity_error"] == "" else float(row["disparity_error"]),
        reverse=True,
    )
    fields = list(diagnostics[0])
    with (output / "occlusion_discontinuity_diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    result: dict[str, object] = {
        "provenance": "CONTROLLED",
        "frames": frame_count,
        "lifecycle": summary["lifecycle"],
        "coverage": summary["coverage"],
        "cer_3": summary.get("cer_3"),
        "cer_10": summary.get("cer_10"),
        "ground_truth_online_access": False,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
