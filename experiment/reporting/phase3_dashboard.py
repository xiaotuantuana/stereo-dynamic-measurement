from __future__ import annotations

import csv
import json
from pathlib import Path


FIELDS = [
    "mode", "status", "evidence", "provenance", "unit", "coverage", "mae", "rmse",
    "bad_1", "bad_3", "cer_10", "jitter", "jump_suppression", "false_correction",
    "diagnosis_precision", "diagnosis_recall", "diagnosis_f1", "runtime", "notes",
]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], field: str) -> float | str:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return sum(values) / len(values) if values else ""


def build_phase3_dashboard(
    *,
    output_path: str | Path,
    innovation1_csv: Path,
    innovation2_csv: Path,
    innovation3_summary: Path,
) -> list[dict[str, object]]:
    i1 = {row["method"]: row for row in _csv(innovation1_csv)}
    i2 = _csv(innovation2_csv)
    i3 = json.loads(innovation3_summary.read_text(encoding="utf-8"))

    def i1_row(mode: str, method: str, status: str) -> dict[str, object]:
        source = i1[method]
        return {
            "mode": mode, "status": status, "evidence": "stateful image sequence",
            "provenance": source["provenance"], "unit": "px",
            "coverage": source["coverage"], "mae": source["valid_mae"], "rmse": source["valid_rmse"],
            "bad_1": source["bad_1"], "bad_3": source["bad_3"], "cer_10": source["cer_10"],
            "jitter": source["jitter_px"], "runtime": source["runtime_median_ms"],
            "notes": f"Controlled {method}; final production behavior unchanged.",
        }

    rows: list[dict[str, object]] = [
        i1_row("BASELINE", "M0", "FUNCTIONAL"),
        i1_row("INNOVATION1", "M3", "PRELIMINARY_VALIDATED"),
        {
            "mode": "INNOVATION1+INNOVATION2", "status": "FUNCTIONAL_COMPONENT_ONLY",
            "evidence": "controlled temporal candidate correction", "provenance": "CONTROLLED", "unit": "mm",
            "coverage": _mean(i2, "coverage"), "mae": _mean(i2, "corrected_mae_mm"),
            "rmse": _mean(i2, "corrected_rmse_mm"), "jitter": _mean(i2, "corrected_jitter_mm"),
            "jump_suppression": _mean([row for row in i2 if row.get("jump_suppression_rate") not in (None, "")], "jump_suppression_rate"),
            "false_correction": _mean(i2, "false_correction_rate"), "runtime": _mean(i2, "runtime_ms"),
            "notes": "Not an end-to-end I1+I2 run; candidate correction remains outside final output authority.",
        },
        {
            "mode": "FULL_ENHANCED", "status": "BLOCKED", "evidence": "controlled diagnosis component only",
            "provenance": "CONTROLLED", "unit": "N/A", "coverage": "", "mae": "", "rmse": "",
            "diagnosis_precision": i3.get("precision", ""), "diagnosis_recall": i3.get("recall", ""),
            "diagnosis_f1": i3.get("f1", ""),
            "notes": "No authorized end-to-end final-result write; accuracy/coverage are intentionally N/A.",
        },
    ]
    normalized = [{field: row.get(field, "") for field in FIELDS} for row in rows]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    output.with_suffix(".json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return normalized
