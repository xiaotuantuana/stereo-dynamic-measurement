from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSample:
    dataset_name: str
    sequence_name: str
    frame_id: str
    left_path: Path
    right_path: Path
    disparity_gt_path: Path | None = None
    disparity_scale: float = 1.0
    depth_gt_path: Path | None = None
    calibration_path: Path | None = None
    timestamp: float | None = None
    official_split: str | None = None
    width: int | None = None
    height: int | None = None
    is_sequence: bool = False

    @property
    def sample_id(self) -> str:
        return f"{self.dataset_name}/{self.sequence_name}/{self.frame_id}"

    def to_record(self, dataset_root: Path) -> dict[str, Any]:
        root = dataset_root.resolve()

        def relative(path: Path | None) -> str | None:
            if path is None:
                return None
            return path.resolve().relative_to(root).as_posix()

        record = asdict(self)
        for field_name in (
            "left_path", "right_path", "disparity_gt_path", "depth_gt_path", "calibration_path"
        ):
            record[field_name] = relative(getattr(self, field_name))
        record["sample_id"] = self.sample_id
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any], dataset_root: Path) -> "DatasetSample":
        root = dataset_root.resolve()

        def absolute(value: str | None) -> Path | None:
            return None if value is None else root / Path(value)

        return cls(
            dataset_name=str(record["dataset_name"]),
            sequence_name=str(record["sequence_name"]),
            frame_id=str(record["frame_id"]),
            left_path=absolute(record["left_path"]),  # type: ignore[arg-type]
            right_path=absolute(record["right_path"]),  # type: ignore[arg-type]
            disparity_gt_path=absolute(record.get("disparity_gt_path")),
            disparity_scale=float(record.get("disparity_scale", 1.0)),
            depth_gt_path=absolute(record.get("depth_gt_path")),
            calibration_path=absolute(record.get("calibration_path")),
            timestamp=None if record.get("timestamp") is None else float(record["timestamp"]),
            official_split=record.get("official_split"),
            width=None if record.get("width") is None else int(record["width"]),
            height=None if record.get("height") is None else int(record["height"]),
            is_sequence=bool(record.get("is_sequence", False)),
        )


@dataclass(frozen=True)
class DatasetIndexMetadata:
    dataset_root: str
    index_path: str
    generated_at: str
    sample_count: int
    dataset_count: int
    adapter_names: tuple[str, ...]
