from pathlib import Path

from experiment.simulation.controlled_stereo import create_controlled_stereo_sequence


def test_controlled_stereo_sequence_is_explicitly_labeled_and_has_gt(tmp_path: Path) -> None:
    sequence = create_controlled_stereo_sequence(tmp_path, frame_count=4, seed=7)
    assert sequence.provenance == "CONTROLLED"
    assert len(sequence.frames) == 4
    assert all(frame.is_sequence for frame in sequence.frames)
    assert all(frame.left_path.is_file() and frame.right_path.is_file() for frame in sequence.frames)
    assert all(frame.disparity_gt_path and frame.disparity_gt_path.is_file() for frame in sequence.frames)
    assert sequence.metadata["ground_truth_online_access"] is False
