from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    dataset_root: Path
    index_path: Path
    results_root: Path
    smoke_per_dataset: int = 10
    development_fraction: float = 0.05
    development_max_frames: int = 500
    default_method: str = "M3"


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_root = config_path.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()

    smoke = int(payload.get("smoke_per_dataset", 10))
    fraction = float(payload.get("development_fraction", 0.05))
    maximum = int(payload.get("development_max_frames", 500))
    if smoke < 1 or maximum < 1 or not 0 < fraction <= 1:
        raise ValueError("Dataset sampling limits must be positive and fraction must be in (0, 1]")
    return DatasetConfig(
        dataset_root=resolve(str(payload["dataset_root"])),
        index_path=resolve(str(payload.get("index_path", "cache/dataset_index.jsonl"))),
        results_root=resolve(str(payload.get("results_root", "results"))),
        smoke_per_dataset=smoke,
        development_fraction=fraction,
        development_max_frames=maximum,
        default_method=str(payload.get("default_method", "M3")),
    )

