from __future__ import annotations

from pathlib import Path

from experiment.benchmark.result_writer import ResultWriter


def test_result_writer_resumes_completed_samples_without_duplicate_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with ResultWriter(run_dir, resume=False) as writer:
        writer.write({"sample_id": "set/train/1", "valid": True})

    with ResultWriter(run_dir, resume=True) as writer:
        assert writer.is_completed("set/train/1")
        assert not writer.is_completed("set/train/2")
        writer.write({"sample_id": "set/train/2", "valid": False})

    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

