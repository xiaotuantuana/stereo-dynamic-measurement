from __future__ import annotations

import argparse

from .report import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporal stereo experiment outputs")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-truth")
    args = parser.parse_args()
    summary = evaluate_experiment(
        args.experiment,
        args.output,
        ground_truth=args.ground_truth,
    )
    print(summary)


if __name__ == "__main__":
    main()
