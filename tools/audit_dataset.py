from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.config import load_dataset_config
from experiment.datasets.audit import write_audit
from experiment.datasets.index import build_index, iter_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or reuse a stereo dataset index and audit it.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    if args.rebuild_index or not config.index_path.is_file():
        metadata = build_index(config.dataset_root, config.index_path)
        print(f"Indexed {metadata.sample_count} stereo pairs with {', '.join(metadata.adapter_names)}")
    else:
        print(f"Reusing index: {config.index_path}")
    csv_path, markdown_path = write_audit(
        iter_index(config.index_path, config.dataset_root), config.results_root
    )
    print(f"Audit CSV: {csv_path}")
    print(f"Summary: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

