from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .innovation2.physics_validation import PhysicsValidationConfig, run_physics_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run innovation-2 four-point physics-confidence validation.")
    parser.add_argument("--config", default="configs/innovation2_validation.yaml")
    args = parser.parse_args()
    source = Path(args.config).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    values = dict(payload.get("physics_validation", payload))
    output = Path(values.pop("output_dir", "../outputs/innovation2"))
    if not output.is_absolute(): output = source.parent / output
    result = run_physics_validation(PhysicsValidationConfig(**values), output)
    for label, path in result.paths.items(): print(f"{label}: {path}")


if __name__ == "__main__": main()
