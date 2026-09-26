from __future__ import annotations

import argparse

from stereo_research.real_experiment import create_synthetic_demo, run_synthetic_dry_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and execute the labelled synthetic real-experiment dry-run")
    parser.add_argument("--root", default="experiments/real_gt/synthetic_demo")
    args = parser.parse_args()
    root = create_synthetic_demo(args.root)
    result = run_synthetic_dry_run(root)
    print(result.status)
    print("SYNTHETIC ONLY - NOT A REAL EXPERIMENT RESULT")
    for path in result.output_paths:
        print(path)
    return 1 if result.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
