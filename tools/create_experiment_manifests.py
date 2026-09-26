from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.config import load_dataset_config
from experiment.datasets.index import iter_index
from experiment.datasets.manifests import create_fixed_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Development and Validation sample manifests.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "configs" / "manifests"))
    parser.add_argument("--development-count", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    paths = create_fixed_manifests(
        iter_index(config.index_path, config.dataset_root), args.output,
        development_count=args.development_count,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

