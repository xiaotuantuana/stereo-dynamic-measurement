from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import DatasetSample


@dataclass(frozen=True)
class ManifestPaths:
    development_path: Path
    validation_path: Path
    metadata_path: Path


def _select_fixed(items: list[DatasetSample], count: int, seed: int) -> list[DatasetSample]:
    ordered = sorted(items, key=lambda item: item.sample_id)
    if len(ordered) <= count:
        return ordered
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda item: item.sample_id)


def _record(sample: DatasetSample) -> dict[str, object]:
    record = asdict(sample)
    for field_name in ("left_path", "right_path", "disparity_gt_path", "depth_gt_path", "calibration_path"):
        value = record[field_name]
        record[field_name] = None if value is None else str(Path(value).resolve())
    record["sample_id"] = sample.sample_id
    return record


def _write(path: Path, samples: list[DatasetSample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(_record(sample), ensure_ascii=False) + "\n")


def create_fixed_manifests(
    samples: Iterable[DatasetSample],
    output_dir: str | Path,
    *,
    development_count: int = 500,
    validation_count: int = 500,
    seed: int = 20260827,
) -> ManifestPaths:
    if development_count < 1 or validation_count < 1:
        raise ValueError("Manifest sample counts must be positive")
    values = list(samples)
    development = _select_fixed([item for item in values if item.official_split == "train"], development_count, seed)
    validation = _select_fixed([item for item in values if item.official_split == "val"], validation_count, seed + 1)
    if not development or not validation:
        raise ValueError("Both train and val samples are required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    development_path = output / "development_manifest.jsonl"
    validation_path = output / "validation_manifest.jsonl"
    metadata_path = output / "manifest_metadata.json"
    _write(development_path, development)
    _write(validation_path, validation)
    metadata_path.write_text(json.dumps({
        "seed": seed,
        "development_source_split": "train",
        "development_count": len(development),
        "validation_source_split": "val",
        "validation_count": len(validation),
        "independent_test_available": False,
        "scene_disjoint": False,
        "scene_disjoint_reason": "source subset is flattened and contains no scene identity",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return ManifestPaths(development_path, validation_path, metadata_path)


def read_manifest(path: str | Path, dataset_root: str | Path) -> Iterator[DatasetSample]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield DatasetSample.from_record(json.loads(line), Path(dataset_root))

