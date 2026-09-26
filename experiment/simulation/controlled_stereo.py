from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..datasets.models import DatasetSample
from ..sequence.models import DatasetSequence


def _write_pfm(path: Path, image: np.ndarray) -> None:
    values = np.asarray(image, dtype="<f4")
    if values.ndim != 2:
        raise ValueError("controlled disparity must be single-channel")
    with path.open("wb") as handle:
        handle.write(f"Pf\n{values.shape[1]} {values.shape[0]}\n-1.0\n".encode("ascii"))
        np.flipud(values).tofile(handle)


def create_controlled_stereo_sequence(
    output_dir: str | Path,
    *,
    frame_count: int = 12,
    seed: int = 20260827,
    width: int = 320,
    height: int = 180,
) -> DatasetSequence:
    if frame_count < 2:
        raise ValueError("controlled sequence requires at least two frames")
    if width < 192 or height < 96:
        raise ValueError("controlled image dimensions are too small for SGBM")
    output = Path(output_dir)
    left_dir, right_dir, gt_dir = output / "left", output / "right", output / "disparity"
    for directory in (left_dir, right_dir, gt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (5, 5), 0.8)
    cv2.rectangle(base, (width // 3, height // 3), (2 * width // 3, 2 * height // 3), 220, 2)
    frames: list[DatasetSample] = []
    disparities: list[float] = []
    for index in range(frame_count):
        motion_x = 3.0 * np.sin(2.0 * np.pi * index / 45.0)
        disparity = 12.0 + 2.0 * np.sin(2.0 * np.pi * index / 60.0)
        left = cv2.warpAffine(
            base,
            np.asarray([[1.0, 0.0, motion_x], [0.0, 1.0, 0.0]], dtype=np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        right = cv2.warpAffine(
            left,
            np.asarray([[1.0, 0.0, -disparity], [0.0, 1.0, 0.0]], dtype=np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        name = f"{index:06d}"
        left_path, right_path, gt_path = left_dir / f"{name}.png", right_dir / f"{name}.png", gt_dir / f"{name}.pfm"
        if not cv2.imwrite(str(left_path), left) or not cv2.imwrite(str(right_path), right):
            raise OSError("failed to write controlled stereo images")
        _write_pfm(gt_path, np.full((height, width), disparity, dtype=np.float32))
        disparities.append(disparity)
        frames.append(DatasetSample(
            dataset_name="CONTROLLED",
            sequence_name="stateful_stereo",
            frame_id=name,
            left_path=left_path,
            right_path=right_path,
            disparity_gt_path=gt_path,
            timestamp=index / 30.0,
            official_split="controlled",
            width=width,
            height=height,
            is_sequence=True,
        ))
    return DatasetSequence.from_frames(
        frames,
        provenance="CONTROLLED",
        metadata={
            "generator": "controlled_stereo_v1",
            "seed": seed,
            "frame_count": frame_count,
            "disparity_start_px": disparities[0],
            "disparity_end_px": disparities[-1],
            "ground_truth_online_access": False,
        },
    )


def create_occlusion_discontinuity_sequence(
    output_dir: str | Path,
    *,
    frame_count: int = 18,
    seed: int = 20260828,
    width: int = 320,
    height: int = 180,
) -> DatasetSequence:
    if frame_count < 2:
        raise ValueError("controlled sequence requires at least two frames")
    output = Path(output_dir)
    left_dir, right_dir, gt_dir = output / "left", output / "right", output / "disparity"
    for directory in (left_dir, right_dir, gt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    background = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (5, 5), 0.8)
    foreground = cv2.GaussianBlur(rng.integers(40, 230, (80, 70), dtype=np.uint8), (3, 3), 0.5)
    frames: list[DatasetSample] = []
    for index in range(frame_count):
        x0 = 160 + int(round(3.0 * np.sin(2.0 * np.pi * index / 18.0)))
        y0, patch_h, patch_w = 50, foreground.shape[0], foreground.shape[1]
        left = background.copy()
        left[y0:y0 + patch_h, x0:x0 + patch_w] = foreground
        right = cv2.warpAffine(
            background,
            np.asarray([[1.0, 0.0, -12.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (width, height), borderMode=cv2.BORDER_REFLECT101,
        )
        right_x = x0 - 28
        right[y0:y0 + patch_h, right_x:right_x + patch_w] = foreground
        disparity = np.full((height, width), 12.0, dtype=np.float32)
        disparity[y0:y0 + patch_h, x0:x0 + patch_w] = 28.0
        name = f"{index:06d}"
        left_path, right_path, gt_path = left_dir / f"{name}.png", right_dir / f"{name}.png", gt_dir / f"{name}.pfm"
        if not cv2.imwrite(str(left_path), left) or not cv2.imwrite(str(right_path), right):
            raise OSError("failed to write controlled occlusion images")
        _write_pfm(gt_path, disparity)
        frames.append(DatasetSample(
            dataset_name="CONTROLLED", sequence_name="occlusion_discontinuity",
            frame_id=name, left_path=left_path, right_path=right_path,
            disparity_gt_path=gt_path, timestamp=index / 30.0,
            official_split="controlled", width=width, height=height, is_sequence=True,
        ))
    return DatasetSequence.from_frames(
        frames, provenance="CONTROLLED",
        metadata={
            "generator": "controlled_occlusion_v1", "scenario": "occlusion_discontinuity",
            "seed": seed, "ground_truth_online_access": False,
        },
    )
