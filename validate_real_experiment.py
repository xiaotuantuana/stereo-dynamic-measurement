from __future__ import annotations

import argparse
import json

from stereo_research.real_experiment import validate_real_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an E0/E1 real-experiment manifest without modifying data")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", help="Optional JSON validation report")
    args = parser.parse_args()
    result = validate_real_experiment(args.manifest)
    payload = result.as_dict()
    print(result.status)
    for issue in result.issues:
        print(f"{issue.level}: {issue.code}: {issue.message}")
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if result.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
