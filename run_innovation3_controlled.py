from __future__ import annotations

import argparse
import json

from experiment.validation.innovation3 import run_innovation3_controlled_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run severity-aware CONTROLLED Innovation3 validation.")
    parser.add_argument("--output-dir", default="results/phase3/innovation3_controlled")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run_innovation3_controlled_validation(args.output_dir, seeds=args.seeds), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
