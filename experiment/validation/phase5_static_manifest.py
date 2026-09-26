"""Deterministic, non-temporal manifest freezer for Phase 5B."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def select_frozen_static_records(records: Iterable[dict[str, Any]], *, target_count: int) -> list[dict[str, Any]]:
    """Select static samples solely by SHA-256(sample_id), never by results."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    static = [dict(record) for record in records if not bool(record.get("is_sequence", False))]
    unique = {str(record["sample_id"]): record for record in static}
    return sorted(unique.values(), key=lambda record: sha256(str(record["sample_id"]).encode()).hexdigest())[:target_count]


def freeze_phase5b_static_manifest(
    index_path: str | Path, dataset_root: str | Path, output_dir: str | Path, *, target_count: int = 5000,
) -> dict[str, object]:
    index, root, output = Path(index_path), Path(dataset_root), Path(output_dir)
    records = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    selected = select_frozen_static_records(records, target_count=target_count)
    if len(selected) != target_count:
        raise ValueError(f"only {len(selected)} unique static samples available; expected {target_count}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "phase5b_5000_static_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    audit_rows: list[dict[str, object]] = []
    for record in selected:
        left = (root / Path(str(record["left_path"]))).resolve()
        right = (root / Path(str(record["right_path"]))).resolve()
        gt = (root / Path(str(record["disparity_gt_path"]))).resolve()
        if not left.is_file() or not right.is_file() or not gt.is_file():
            raise FileNotFoundError(f"incomplete static source for {record['sample_id']}")
        audit_rows.append({
            "sample_id": record["sample_id"], "left_path": str(left), "right_path": str(right),
            "gt_path": str(gt), "left_sha256": _file_sha256(left),
            "right_sha256": _file_sha256(right), "gt_sha256": _file_sha256(gt),
        })
    audit_path = output / "phase5b_5000_static_manifest_audit.csv"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    digest = _file_sha256(manifest_path)
    metadata = {
        "selection_rule": "ascending SHA-256(sample_id); static rows only; no result-dependent selection",
        "static_pool_count": len({str(record["sample_id"]) for record in records if not bool(record.get("is_sequence", False))}),
        "selected_count": len(selected), "manifest_sha256": digest,
        "calibration_available": False, "temporal_continuity_used": False,
    }
    (output / "phase5b_manifest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(freeze_phase5b_static_manifest(
        "cache/dataset_index.jsonl", "E:/数据集", "results/phase5b_5000_static",
    ), indent=2), flush=True)
