from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3", required=True)
    parser.add_argument("--m0", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    m3_dir, m0_dir, output = Path(args.m3), Path(args.m0), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    m3, m0 = _rows(m3_dir / "metrics.csv"), _rows(m0_dir / "metrics.csv")
    s3, s0 = _summary(m3_dir / "summary.json"), _summary(m0_dir / "summary.json")
    same_predictions = all(
        a["status"] == b["status"] and a["predicted_disparity"] == b["predicted_disparity"]
        for a, b in zip(m3, m0, strict=True)
    )
    ablation = []
    for method, summary in (("M0", s0), ("M3", s3)):
        ablation.append({
            "method": method,
            "coverage": summary.get("coverage", summary.get("valid_rate")),
            "disparity_mae": summary.get("disparity_mae"),
            "disparity_rmse": summary.get("disparity_rmse"),
            "bad_3": summary.get("bad_3"),
            "cer_10": summary.get("cer_10", summary.get("bad_3")),
            "initialization_only": True,
            "innovation1_identifiable": False,
            "prediction_rows_identical": same_predictions,
            "interpretation": "Static benchmark calls initialize() only; temporal/local M3 components are inactive.",
        })
    _write(output / "ablation_innovation1.csv", ablation)

    valid = [r for r in m3 if r.get("disparity_error") not in (None, "")]
    gt = np.asarray([float(r["gt_disparity"]) for r in valid])
    sub = np.asarray([float(r["predicted_disparity"]) for r in valid])
    integer = np.rint(sub)
    variants = []
    for name, pred in (("rounded_integer", integer), ("sgbm_internal_subpixel", sub)):
        error = np.abs(pred - gt)
        variants.append({
            "variant": name,
            "points": int(error.size),
            "mae": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bad_1": float(np.mean(error > 1)),
            "bad_3": float(np.mean(error > 3)),
            "cer_10": float(np.mean(error > 10)),
            "innovation1_subpixel_identifiable": False,
            "interpretation": "This isolates OpenCV SGBM fractional output, not LocalMatcher temporal subpixel refinement.",
        })
    _write(output / "subpixel_analysis.csv", variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
