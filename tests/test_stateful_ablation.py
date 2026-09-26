from pathlib import Path

from experiment.sequence.ablation import run_controlled_ablation


def test_controlled_ablation_runs_real_m0_and_m3_stateful_paths(tmp_path: Path) -> None:
    summary = run_controlled_ablation(tmp_path, frame_count=4, methods=("M0", "M3"))
    assert set(summary["methods"]) == {"M0", "M3"}
    assert summary["lifecycle"]["M3"]["initialize_calls"] == 1
    assert summary["lifecycle"]["M3"]["step_calls"] == 3
    assert summary["stateful_protocol"] is True
    assert (tmp_path / "ablation_innovation1.csv").is_file()
