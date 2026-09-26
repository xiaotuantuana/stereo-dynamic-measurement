from __future__ import annotations

import argparse
import json

from experiment.validation.innovation2 import run_innovation2_controlled_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CONTROLLED Innovation2 temporal scenarios.")
    parser.add_argument("--output-dir", default="results/phase3/innovation2_controlled")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    print(json.dumps(run_innovation2_controlled_validation(args.output_dir, frame_count=args.frames, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
