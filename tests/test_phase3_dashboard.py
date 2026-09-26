import csv
from pathlib import Path

from experiment.reporting.phase3_dashboard import build_phase3_dashboard


def test_dashboard_keeps_incomparable_controlled_metrics_as_na(tmp_path: Path) -> None:
    rows = build_phase3_dashboard(
        output_path=tmp_path / "experiment_summary.csv",
        innovation1_csv=Path("results/phase3/stateful_innovation1_smoke/ablation_innovation1.csv"),
        innovation2_csv=Path("results/phase3/innovation2_controlled/scenario_metrics.csv"),
        innovation3_summary=Path("results/phase3/innovation3_controlled/summary.json"),
    )
    assert [row["mode"] for row in rows] == ["BASELINE", "INNOVATION1", "INNOVATION1+INNOVATION2", "FULL_ENHANCED"]
    assert rows[-1]["status"] == "BLOCKED"
    assert rows[-1]["mae"] == ""
    with (tmp_path / "experiment_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4
