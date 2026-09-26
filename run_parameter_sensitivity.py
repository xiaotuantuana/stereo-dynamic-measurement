from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiment.optimization.sensitivity import sweep_gates


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bin_metrics(rows: list[dict[str, str]], field: str, edges: list[float]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for low, high in zip(edges, edges[1:]):
        bucket = []
        for row in rows:
            if row.get("confidence_source") != "initial_runtime_evidence":
                continue
            try:
                value = float(row[field])
                error = float(row["disparity_error"])
            except (KeyError, TypeError, ValueError):
                continue
            if low <= value < high:
                bucket.append(error)
        if bucket:
            import numpy as np
            values = np.asarray(bucket)
            output.append({
                "field": field, "bin_low": low, "bin_high": high, "points": len(bucket),
                "mae": float(np.mean(values)), "rmse": float(np.sqrt(np.mean(values ** 2))),
                "bad_3": float(np.mean(values > 3)), "cer_10": float(np.mean(values > 10)),
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline gate sweep over runtime-only evidence.")
    parser.add_argument("--development", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    margins = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]
    confidences = [0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
    output = Path(args.output_dir)
    development_rows = _read(Path(args.development))
    validation_rows = _read(Path(args.validation))
    dev = sweep_gates(development_rows, uniqueness_margins=margins, confidence_thresholds=confidences)
    val = sweep_gates(validation_rows, uniqueness_margins=margins, confidence_thresholds=confidences)
    val_lookup = {(r["uniqueness_margin"], r["confidence_threshold"]): r for r in val}
    combined: list[dict[str, object]] = []
    for row in dev:
        other = val_lookup[(row["uniqueness_margin"], row["confidence_threshold"])]
        combined.append({**{f"development_{k}": v for k, v in row.items()}, **{f"validation_{k}": v for k, v in other.items()}})
    _write(output / "parameter_sensitivity.csv", combined)
    safe = [r for r in combined if r.get("development_cer_10") == 0.0]
    recommended = (
        max(safe, key=lambda r: (float(r["development_coverage"]), -float(r.get("development_rmse", 1e9))))
        if safe else min(
            combined,
            key=lambda r: (float(r.get("development_cer_10", 1.0)), -float(r["development_coverage"])),
        )
    )
    recommended["selection_rule"] = (
        "maximum development coverage subject to development CER@10px=0"
        if safe else "minimum development CER@10px, then maximum development coverage"
    )
    (output / "recommended_operating_point.json").write_text(
        json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dev_disparity = _bin_metrics(development_rows, "gt_disparity", [0, 16, 32, 64, 96, 128, 256, 1e9])
    val_disparity = _bin_metrics(validation_rows, "gt_disparity", [0, 16, 32, 64, 96, 128, 256, 1e9])
    dev_texture = _bin_metrics(development_rows, "texture_std", [0, 5, 10, 20, 40, 80, 1e9])
    val_texture = _bin_metrics(validation_rows, "texture_std", [0, 5, 10, 20, 40, 80, 1e9])
    _write(output / "development_metrics_by_disparity.csv", dev_disparity)
    _write(output / "validation_metrics_by_disparity.csv", val_disparity)
    _write(output / "development_metrics_by_texture.csv", dev_texture)
    _write(output / "validation_metrics_by_texture.csv", val_texture)
    _write(output / "metrics_by_disparity.csv", [{"split": split, **row} for split, rows in (("development", dev_disparity), ("validation", val_disparity)) for row in rows])
    _write(output / "metrics_by_texture.csv", [{"split": split, **row} for split, rows in (("development", dev_texture), ("validation", val_texture)) for row in rows])
    print(json.dumps(recommended, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
