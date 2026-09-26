from __future__ import annotations

import argparse
import json

from experiment.sequence.ablation import run_controlled_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CONTROLLED stateful Innovation1 ablation.")
    parser.add_argument("--output-dir", default="results/phase3/stateful_innovation1_smoke")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run_controlled_ablation(args.output_dir, frame_count=args.frames), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
