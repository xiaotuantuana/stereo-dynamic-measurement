from __future__ import annotations

import argparse

from stereo_research.real_experiment import evaluate_e1_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline E1 distance/baseline/target evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_e1_manifest(args.manifest, args.output)
    print(result.status)
    for path in result.output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
