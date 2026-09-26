from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .io import image_size
from .models import DatasetSample


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class FlyingThings3DSubsetAdapter:
    name = "flyingthings3d_subset"

    def discover(self, dataset_root: Path) -> Iterator[DatasetSample]:
        image_roots = sorted(dataset_root.glob("*image_clean.tar/FlyingThings3D_subset"))
        for image_root in image_roots:
            disparity_root = dataset_root / image_root.parent.name.replace(
                "image_clean", "disparity"
            ) / image_root.name
            for split in ("train", "val", "test"):
                left_dir = image_root / split / "image_clean" / "left"
                right_dir = image_root / split / "image_clean" / "right"
                if not left_dir.is_dir() or not right_dir.is_dir():
                    continue
                width: int | None = None
                height: int | None = None
                for left_path in sorted(path for path in left_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
                    right_path = right_dir / left_path.name
                    if not right_path.is_file():
                        continue
                    if width is None or height is None:
                        width, height = image_size(left_path)
                    gt_path = disparity_root / split / "disparity" / "left" / f"{left_path.stem}.pfm"
                    yield DatasetSample(
                        dataset_name="FlyingThings3D_subset",
                        sequence_name=split,
                        frame_id=left_path.stem,
                        left_path=left_path,
                        right_path=right_path,
                        disparity_gt_path=gt_path if gt_path.is_file() else None,
                        disparity_scale=-1.0,
                        official_split=split,
                        width=width,
                        height=height,
                        is_sequence=False,
                    )


class GenericStereoAdapter:
    name = "generic_stereo"
    _pairs = {
        "left": "right",
        "image_0": "image_1",
        "image_2": "image_3",
        "cam0": "cam1",
        "l": "r",
    }

    def discover(self, dataset_root: Path, excluded_roots: Iterable[Path] = ()) -> Iterator[DatasetSample]:
        excluded = tuple(path.resolve() for path in excluded_roots)
        for left_dir in sorted(path for path in dataset_root.rglob("*") if path.is_dir()):
            if any(root == left_dir.resolve() or root in left_dir.resolve().parents for root in excluded):
                continue
            right_name = self._pairs.get(left_dir.name.lower())
            if right_name is None:
                continue
            right_dir = self._case_insensitive_child(left_dir.parent, right_name)
            if right_dir is None:
                continue
            dataset_name = self._dataset_name(dataset_root, left_dir)
            sequence_name = left_dir.parent.name or "static"
            width: int | None = None
            height: int | None = None
            for left_path in sorted(path for path in left_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
                relative = left_path.relative_to(left_dir)
                right_path = right_dir / relative
                if not right_path.is_file():
                    continue
                if width is None or height is None:
                    width, height = image_size(left_path)
                frame_id = relative.with_suffix("").as_posix()
                yield DatasetSample(
                    dataset_name=dataset_name,
                    sequence_name=sequence_name,
                    frame_id=frame_id,
                    left_path=left_path,
                    right_path=right_path,
                    width=width,
                    height=height,
                    is_sequence=False,
                )

    @staticmethod
    def _case_insensitive_child(parent: Path, expected: str) -> Path | None:
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == expected:
                return child
        return None

    @staticmethod
    def _dataset_name(dataset_root: Path, left_dir: Path) -> str:
        relative = left_dir.relative_to(dataset_root)
        return relative.parts[0] if len(relative.parts) > 1 else left_dir.parent.name
