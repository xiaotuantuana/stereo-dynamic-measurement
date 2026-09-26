from __future__ import annotations

import argparse
from pathlib import Path

from experiment.analysis.run_failure_analysis import run_failure_analysis
from experiment.config import load_dataset_config


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Phase 1 smoke failures with runtime diagnostics.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--baseline", default=str(PROJECT_ROOT / "results" / "phase1_5_baseline_snapshot"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "phase1_5_failure_analysis"))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    summary = run_failure_analysis(
        load_dataset_config(args.config), args.baseline, args.output, top_n=args.top_n
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

