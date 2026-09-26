from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..datasets.models import DatasetSample


SequenceProvenance = Literal["REAL_DATASET", "CONTROLLED"]


def _frame_key(sample: DatasetSample) -> tuple[int, int | str]:
    try:
        return 0, int(sample.frame_id)
    except ValueError:
        return 1, sample.frame_id


@dataclass(frozen=True)
class DatasetSequence:
    dataset_name: str
    sequence_name: str
    frames: tuple[DatasetSample, ...]
    provenance: SequenceProvenance
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_frames(
        cls,
        frames: list[DatasetSample] | tuple[DatasetSample, ...],
        *,
        provenance: SequenceProvenance,
        metadata: dict[str, object] | None = None,
    ) -> "DatasetSequence":
        if not frames:
            raise ValueError("sequence must contain at least one frame")
        first = frames[0]
        if any(
            frame.dataset_name != first.dataset_name
            or frame.sequence_name != first.sequence_name
            for frame in frames
        ):
            raise ValueError("all frames must belong to the same dataset and sequence")
        ordered = tuple(sorted(frames, key=_frame_key))
        ids = [frame.frame_id for frame in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate frame_id in sequence")
        timestamps = [frame.timestamp for frame in ordered if frame.timestamp is not None]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("sequence timestamps must be strictly increasing")
        if provenance == "REAL_DATASET" and not all(frame.is_sequence for frame in ordered):
            raise ValueError("REAL_DATASET sequence requires adapter-verified sequence frames")
        return cls(
            dataset_name=first.dataset_name,
            sequence_name=first.sequence_name,
            frames=ordered,
            provenance=provenance,
            metadata=dict(metadata or {}),
        )

    @property
    def sequence_id(self) -> str:
        return f"{self.dataset_name}/{self.sequence_name}"
