from __future__ import annotations

from pathlib import Path

from experiment.datasets.manifests import create_fixed_manifests, read_manifest
from experiment.datasets.models import DatasetSample


def _sample(split: str, frame: int) -> DatasetSample:
    return DatasetSample(
        dataset_name="FlyingThings3D_subset",
        sequence_name=split,
        frame_id=f"{frame:07d}",
        left_path=Path(f"{split}/left/{frame}.png"),
        right_path=Path(f"{split}/right/{frame}.png"),
        official_split=split,
    )


def test_fixed_manifests_keep_train_and_val_separate_and_are_reproducible(tmp_path: Path) -> None:
    samples = [_sample("train", index) for index in range(20)] + [
        _sample("val", index) for index in range(20)
    ]

    first = create_fixed_manifests(samples, tmp_path, development_count=7, validation_count=5, seed=42)
    first_dev = [item.sample_id for item in read_manifest(first.development_path, tmp_path)]
    first_val = [item.sample_id for item in read_manifest(first.validation_path, tmp_path)]
    second = create_fixed_manifests(samples, tmp_path, development_count=7, validation_count=5, seed=42)

    assert all("/train/" in sample_id for sample_id in first_dev)
    assert all("/val/" in sample_id for sample_id in first_val)
    assert first_dev == [item.sample_id for item in read_manifest(second.development_path, tmp_path)]
    assert first_val == [item.sample_id for item in read_manifest(second.validation_path, tmp_path)]

