from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from experiment.datasets.index import build_index, iter_index
from experiment.datasets.audit import write_audit


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((12, 16, 3), dtype=np.uint8))


def test_flyingthings_split_roots_are_paired_without_loading_all_images(tmp_path: Path) -> None:
    image_root = tmp_path / "FlyingThings3D_subset_image_clean.tar" / "FlyingThings3D_subset"
    disparity_root = tmp_path / "FlyingThings3D_subset_disparity.tar" / "FlyingThings3D_subset"
    for side in ("left", "right"):
        _write_image(image_root / "train" / "image_clean" / side / "0000000.png")
    gt = disparity_root / "train" / "disparity" / "left" / "0000000.pfm"
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_bytes(b"Pf\n1 1\n-1.0\n\x00\x00\x80?")

    index_path = tmp_path / "cache" / "dataset_index.jsonl"
    metadata = build_index(tmp_path, index_path)
    samples = list(iter_index(index_path, tmp_path))

    assert metadata.sample_count == 1
    assert metadata.dataset_count == 1
    assert samples[0].sample_id == "FlyingThings3D_subset/train/0000000"
    assert samples[0].left_path.name == "0000000.png"
    assert samples[0].right_path.name == "0000000.png"
    assert samples[0].disparity_gt_path == gt
    assert samples[0].disparity_scale == -1.0
    assert samples[0].width == 16
    assert samples[0].height == 12
    assert samples[0].is_sequence is False


def test_iter_index_reuses_cached_records(tmp_path: Path) -> None:
    image_root = tmp_path / "sample" / "left"
    _write_image(image_root / "frame.png")
    _write_image(tmp_path / "sample" / "right" / "frame.png")
    index_path = tmp_path / "cache" / "dataset_index.jsonl"
    build_index(tmp_path, index_path)

    (image_root / "later.png").write_bytes(b"not scanned")

    assert [sample.frame_id for sample in iter_index(index_path, tmp_path)] == ["frame"]


def test_audit_reports_gt_and_static_suitability(tmp_path: Path) -> None:
    sample = __import__("experiment.datasets.models", fromlist=["DatasetSample"]).DatasetSample(
        dataset_name="set-a",
        sequence_name="train",
        frame_id="1",
        left_path=tmp_path / "left.png",
        right_path=tmp_path / "right.png",
        disparity_gt_path=tmp_path / "disp.pfm",
        width=960,
        height=540,
        is_sequence=False,
    )

    csv_path, markdown_path = write_audit([sample], tmp_path / "results")

    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "set-a,1,Yes,disparity,No,No,960x540,Innovation1" in csv_text
    assert "静态视差精度" in markdown_path.read_text(encoding="utf-8")
