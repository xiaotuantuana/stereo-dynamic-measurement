from pathlib import Path

import numpy as np

from experiment.datasets.io import read_pfm
from experiment.sequence.occlusion_diagnostics import run_occlusion_diagnostics
from experiment.simulation.controlled_stereo import create_occlusion_discontinuity_sequence


def test_occlusion_sequence_contains_real_disparity_discontinuity(tmp_path: Path) -> None:
    sequence = create_occlusion_discontinuity_sequence(tmp_path, frame_count=4)
    gt = read_pfm(sequence.frames[0].disparity_gt_path)
    assert set(np.unique(gt)) == {12.0, 28.0}
    assert sequence.metadata["scenario"] == "occlusion_discontinuity"


def test_occlusion_diagnostics_run_through_stateful_pipeline(tmp_path: Path) -> None:
    summary = run_occlusion_diagnostics(tmp_path, frame_count=6)
    assert summary["provenance"] == "CONTROLLED"
    assert summary["lifecycle"]["initialize_calls"] == 1
    assert summary["lifecycle"]["step_calls"] == 5
    assert (tmp_path / "occlusion_discontinuity_diagnostics.csv").is_file()
