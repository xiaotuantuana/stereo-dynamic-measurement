"""Read-only source inventory and deterministic manifest freezing for Phase 5."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Iterable
import csv
import json
import argparse


@dataclass(frozen=True)
class SourceSample:
    source: str
    sample_id: str
    frame_id: str
    timestamp: float | None
    left_path: Path
    right_path: Path
    gt_path: Path | None
    is_sequence: bool = False


@dataclass(frozen=True)
class DeduplicationReport:
    total_manifest_rows: int
    unique_samples: tuple[SourceSample, ...]
    duplicate_samples: int
    overlap_counts: dict[tuple[str, str], int]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def deduplicate_source_samples(samples: Iterable[SourceSample]) -> DeduplicationReport:
    """Collapse duplicate stereo observations by path, metadata, or image content.

    A pair is considered the same observation when it shares a left/right source
    pair, full source identity (sample/frame/timestamp/GT), or the same
    left/right image SHA-256 *with the same GT and timestamp identity*.  Image
    content by itself cannot merge two temporal observations with distinct time
    or GT identity.  The union rather than a single composite key preserves every
    declared identity channel in the audit.
    """

    values = list(samples)
    union_find = _UnionFind(len(values))
    first_for: dict[tuple[object, ...], int] = {}
    hashes: dict[Path, str] = {}

    def register(key: tuple[object, ...], index: int) -> None:
        existing = first_for.setdefault(key, index)
        union_find.union(existing, index)

    for index, sample in enumerate(values):
        left, right = sample.left_path.resolve(), sample.right_path.resolve()
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"source image is missing for {sample.sample_id}")
        register(("paths", str(left), str(right), sample.timestamp,
                  None if sample.gt_path is None else str(sample.gt_path.resolve())), index)
        register(("metadata", sample.sample_id, sample.frame_id, sample.timestamp,
                  None if sample.gt_path is None else str(sample.gt_path.resolve())), index)
        hashes.setdefault(left, _file_sha256(left))
        hashes.setdefault(right, _file_sha256(right))
        register(("image_sha256", hashes[left], hashes[right], sample.timestamp,
                  None if sample.gt_path is None else str(sample.gt_path.resolve())), index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(values)):
        groups[union_find.find(index)].append(index)
    unique_indices = [min(indices) for _, indices in sorted(groups.items(), key=lambda item: min(item[1]))]
    overlap_counts: dict[tuple[str, str], int] = defaultdict(int)
    for indices in groups.values():
        present = sorted({values[index].source for index in indices})
        for pair in combinations(present, 2):
            overlap_counts[pair] += 1
    return DeduplicationReport(
        total_manifest_rows=len(values),
        unique_samples=tuple(values[index] for index in unique_indices),
        duplicate_samples=len(values) - len(unique_indices),
        overlap_counts=dict(sorted(overlap_counts.items())),
    )


def _from_record(source: str, record: dict[str, object], root: Path) -> SourceSample:
    def absolute(name: str) -> Path | None:
        value = record.get(name)
        return None if value is None else (root / Path(str(value))).resolve()

    return SourceSample(
        source=source,
        sample_id=str(record["sample_id"]),
        frame_id=str(record["frame_id"]),
        timestamp=None if record.get("timestamp") is None else float(record["timestamp"]),
        left_path=absolute("left_path"),  # type: ignore[arg-type]
        right_path=absolute("right_path"),  # type: ignore[arg-type]
        gt_path=absolute("disparity_gt_path"),
        is_sequence=bool(record.get("is_sequence", False)),
    )


def _read_jsonl(source: str, path: Path, root: Path) -> list[SourceSample]:
    return [_from_record(source, json.loads(line), root) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_phase3_manifest(path: Path, root: Path) -> list[SourceSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: list[SourceSample] = []
    for sequence in payload["sequences"]:
        sequence_id = str(sequence["sequence_id"])
        for frame in sequence["frames"]:
            samples.append(SourceSample(
                source="phase3_500",
                sample_id=f"{sequence_id}/{frame['frame_id']}",
                frame_id=str(frame["frame_id"]),
                timestamp=float(frame["timestamp"]),
                left_path=(root / Path(frame["left_path"])).resolve(),
                right_path=(root / Path(frame["right_path"])).resolve(),
                gt_path=(root / Path(frame["disparity_gt_path"])).resolve(),
                is_sequence=True,
            ))
    return samples


def inventory_current_sources(project_root: str | Path, dataset_root: str | Path) -> DeduplicationReport:
    root, data_root = Path(project_root).resolve(), Path(dataset_root).resolve()
    samples = [
        *_read_phase3_manifest(root / "results/phase3/stateful_innovation1_500/M3/sequence_manifest.json", root),
        *_read_jsonl("development_manifest", root / "configs/manifests/development_manifest.jsonl", data_root),
        *_read_jsonl("validation_manifest", root / "configs/manifests/validation_manifest.jsonl", data_root),
        *_read_jsonl("dataset_index", root / "cache/dataset_index.jsonl", data_root),
    ]
    return deduplicate_source_samples(samples)


def freeze_deterministic_manifest(
    samples: Iterable[SourceSample], path: str | Path, *, target_count: int,
) -> tuple[list[dict[str, object]], str]:
    """Freeze at most ``target_count`` unique observations in declared source order.

    Controlled temporal observations remain first so their stateful semantics are
    retained; all other samples follow by source and stable sample identity.
    """

    priority = {"phase3_500": 0, "development_manifest": 1, "validation_manifest": 2, "dataset_index": 3}
    ordered = sorted(samples, key=lambda item: (priority.get(item.source, 99), item.sample_id, item.frame_id))
    rows = [{
        "source": item.source,
        "sample_id": item.sample_id,
        "frame_id": item.frame_id,
        "timestamp": item.timestamp,
        "left_path": str(item.left_path),
        "right_path": str(item.right_path),
        "gt_path": "" if item.gt_path is None else str(item.gt_path),
        "is_sequence": item.is_sequence,
        "left_sha256": _file_sha256(item.left_path),
        "right_sha256": _file_sha256(item.right_path),
    } for item in ordered[:target_count]]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["source", "sample_id", "frame_id", "timestamp", "left_path", "right_path", "gt_path", "is_sequence", "left_sha256", "right_sha256"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, _file_sha256(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--target-count", type=int, default=2000)
    options = parser.parse_args()
    report = inventory_current_sources(Path.cwd(), Path("E:/数据集"))
    frozen_sha256 = None
    frozen_count = None
    if options.freeze is not None:
        rows, frozen_sha256 = freeze_deterministic_manifest(
            report.unique_samples, options.freeze, target_count=options.target_count,
        )
        frozen_count = len(rows)
    print(json.dumps({
        "TOTAL_MANIFEST_ROWS": report.total_manifest_rows,
        "UNIQUE_STEREO_SAMPLES": len(report.unique_samples),
        "DUPLICATE_SAMPLES": report.duplicate_samples,
        "OVERLAP_COUNTS": {" | ".join(key): value for key, value in report.overlap_counts.items()},
        "FROZEN_MANIFEST_ROWS": frozen_count,
        "FROZEN_MANIFEST_SHA256": frozen_sha256,
    }, ensure_ascii=False, indent=2), flush=True)
