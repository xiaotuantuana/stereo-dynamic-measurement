from __future__ import annotations

from pathlib import Path

from experiment.validation.phase5_data_inventory import (
    SourceSample,
    deduplicate_source_samples,
    freeze_deterministic_manifest,
)


def _sample(source: str, left: Path, right: Path, *, frame_id: str, gt: Path) -> SourceSample:
    return SourceSample(
        source=source,
        sample_id=f"dataset/sequence/{frame_id}",
        frame_id=frame_id,
        timestamp=None,
        left_path=left,
        right_path=right,
        gt_path=gt,
    )


def test_deduplicate_source_samples_collapses_manifest_copies_by_paths_and_hashes(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    gt = tmp_path / "gt.pfm"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    gt.write_bytes(b"gt")
    copied_left = tmp_path / "copied-left.png"
    copied_right = tmp_path / "copied-right.png"
    copied_left.write_bytes(b"left")
    copied_right.write_bytes(b"right")

    report = deduplicate_source_samples([
        _sample("phase3", left, right, frame_id="0001", gt=gt),
        _sample("derived_copy", left, right, frame_id="0001", gt=gt),
        _sample("content_copy", copied_left, copied_right, frame_id="other", gt=gt),
    ])

    assert report.total_manifest_rows == 3
    assert len(report.unique_samples) == 1
    assert report.duplicate_samples == 2
    assert report.overlap_counts[("content_copy", "derived_copy")] == 1
    assert report.overlap_counts[("derived_copy", "phase3")] == 1


def test_freeze_deterministic_manifest_prioritizes_temporal_source_and_is_sha256_bound(tmp_path: Path) -> None:
    paths = []
    for name in ("a-left", "a-right", "b-left", "b-right"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)
    samples = (
        _sample("dataset_index", paths[2], paths[3], frame_id="0002", gt=paths[0]),
        SourceSample("phase3_500", "controlled/0001", "0001", 0.0, paths[0], paths[1], paths[2], True),
    )

    manifest, digest = freeze_deterministic_manifest(samples, tmp_path / "frozen.csv", target_count=2)

    assert [row["source"] for row in manifest] == ["phase3_500", "dataset_index"]
    assert len(digest) == 64
    assert (tmp_path / "frozen.csv").read_text(encoding="utf-8-sig").count("\n") == 3


def test_deduplication_preserves_identical_images_at_distinct_timestamps(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    gt = tmp_path / "gt.pfm"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    gt.write_bytes(b"gt")

    report = deduplicate_source_samples([
        SourceSample("phase3_500", "controlled/0001", "0001", 0.0, left, right, gt, True),
        SourceSample("phase3_500", "controlled/0002", "0002", 1.0, left, right, gt, True),
    ])

    assert len(report.unique_samples) == 2
    assert report.duplicate_samples == 0
