"""Audit a SCADA file before training or live shadow deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = PROJECT_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from scada_adapter import prepare_scada_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/raw/scada_pipeline.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/titas_scada.yml")
    args = parser.parse_args()

    _, report = prepare_scada_file(args.data, args.config, require_labels=False)
    print(f"Input rows: {report['input_rows']}")
    print(f"Accepted rows: {report['output_rows']}")
    print(f"Invalid rows: {report['invalid_rows_removed']}")
    print(f"Exact duplicates removed: {report['duplicate_rows_removed']}")
    print(f"Timestamp collision groups: {report['timestamp_collision_groups']}")
    print(f"Timestamp collision rows: {report['timestamp_collision_rows']}")
    print(f"Invalid rate: {report['invalid_rate']:.2%}")
    if report["timestamp_collision_rows"]:
        print("STATUS: REVIEW REQUIRED - chronology is ambiguous")
    elif report["invalid_rows_removed"]:
        print("STATUS: REVIEW REQUIRED - invalid rows were removed")
    else:
        print("STATUS: READY FOR TRAINING REVIEW")


if __name__ == "__main__":
    main()