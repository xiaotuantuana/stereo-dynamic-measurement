from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiment.datasets.models import DatasetSample
from experiment.sequence.models import DatasetSequence
from experiment.sequence.runner import run_sequences
from stereo_research.models import FramePointResult, PointSpec


def _sample(frame: str, *, sequence: str = "controlled_a") -> DatasetSample:
    return DatasetSample(
        dataset_name="CONTROLLED",
        sequence_name=sequence,
        frame_id=frame,
        left_path=Path(f"left_{frame}.png"),
        right_path=Path(f"right_{frame}.png"),
        timestamp=float(frame),
        is_sequence=True,
    )


def test_dataset_sequence_orders_numeric_frames_and_rejects_duplicates() -> None:
    sequence = DatasetSequence.from_frames([_sample("2"), _sample("0"), _sample("1")], provenance="CONTROLLED")
    assert [frame.frame_id for frame in sequence.frames] == ["0", "1", "2"]
    with pytest.raises(ValueError, match="duplicate"):
        DatasetSequence.from_frames([_sample("0"), _sample("0")], provenance="CONTROLLED")


def test_stateful_runner_initializes_once_steps_remaining_frames_and_isolates_sequences(tmp_path: Path) -> None:
    instances: list[FakePipeline] = []

    def factory(*_args, **_kwargs):
        pipeline = FakePipeline()
        instances.append(pipeline)
        return pipeline

    sequences = [
        DatasetSequence.from_frames([_sample("0"), _sample("1"), _sample("2")], provenance="CONTROLLED"),
        DatasetSequence.from_frames([_sample("0", sequence="controlled_b"), _sample("1", sequence="controlled_b")], provenance="CONTROLLED"),
    ]
    summary = run_sequences(
        sequences,
        tmp_path,
        method="M3",
        q=np.eye(4),
        calibration_unit="m",
        points=(PointSpec("P1", (20.0, 20.0)),),
        pipeline_factory=factory,
        image_loader=lambda _path: np.zeros((40, 80), dtype=np.uint8),
    )

    assert len(instances) == 2
    assert [(item.initialize_calls, item.step_calls) for item in instances] == [(1, 2), (1, 1)]
    assert summary["lifecycle"] == {"sequences": 2, "initialize_calls": 2, "step_calls": 3, "reset_equivalent_calls": 2}
    lifecycle = json.loads((tmp_path / "lifecycle.json").read_text(encoding="utf-8"))
    assert lifecycle["reset_strategy"] == "new_pipeline_per_sequence"
    assert (tmp_path / "run_metadata.json").is_file()
    assert (tmp_path / "sequence_manifest.json").is_file()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["method"] == "M3"
    assert "generated_at" in metadata and "git_commit" in metadata


class FakePipeline:
    def __init__(self) -> None:
        self.initialize_calls = 0
        self.step_calls = 0

    def initialize(self, _left, _right, points, frame):
        self.initialize_calls += 1
        return [self._result(point.point_id, frame) for point in points]

    def step(self, _left, _right, frame):
        self.step_calls += 1
        return [self._result("P1", frame)]

    @staticmethod
    def _result(point_id: str, frame: int) -> FramePointResult:
        return FramePointResult(
            method="M3", frame=frame, point_id=point_id, status="valid",
            left_x=20.0, left_y=20.0, disparity=8.0, measured_disparity=8.0,
            confidence=0.8,
        )
