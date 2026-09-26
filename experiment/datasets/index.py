from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .adapters import FlyingThings3DSubsetAdapter, GenericStereoAdapter
from .models import DatasetIndexMetadata, DatasetSample


def build_index(dataset_root: str | Path, index_path: str | Path) -> DatasetIndexMetadata:
    root = Path(dataset_root).resolve()
    output = Path(index_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    flying = FlyingThings3DSubsetAdapter()
    generic = GenericStereoAdapter()
    excluded = sorted(root.glob("*FlyingThings3D_subset*.tar"))
    adapters_used: set[str] = set()
    datasets: set[str] = set()
    sample_count = 0
    seen: set[str] = set()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        sources = (
            (flying.name, flying.discover(root)),
            (generic.name, generic.discover(root, excluded_roots=excluded)),
        )
        for adapter_name, samples in sources:
            for sample in samples:
                if sample.sample_id in seen:
                    continue
                seen.add(sample.sample_id)
                adapters_used.add(adapter_name)
                datasets.add(sample.dataset_name)
                handle.write(json.dumps(sample.to_record(root), ensure_ascii=False) + "\n")
                sample_count += 1
    temporary.replace(output)
    metadata = DatasetIndexMetadata(
        dataset_root=str(root),
        index_path=str(output),
        generated_at=datetime.now(timezone.utc).isoformat(),
        sample_count=sample_count,
        dataset_count=len(datasets),
        adapter_names=tuple(sorted(adapters_used)),
    )
    output.with_suffix(".meta.json").write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def iter_index(index_path: str | Path, dataset_root: str | Path) -> Iterator[DatasetSample]:
    path = Path(index_path)
    root = Path(dataset_root)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid dataset index record at line {line_number}") from exc
            yield DatasetSample.from_record(record, root)

