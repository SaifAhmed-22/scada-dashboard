"""Repair ambiguous timestamps in the checked-in synthetic sample.

This is only for the sample dataset. Real Titas data must use a historian
sequence/ingestion ID or an approved source timestamp instead of this fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "scada_pipeline.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "scada_pipeline_repaired.csv"


def repair_timestamps(input_path: Path, output_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    frame["original_timestamp"] = frame["timestamp"]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["_source_order"] = range(len(frame))
    frame = frame.sort_values(
        ["segment_id", "timestamp", "_source_order"], kind="stable"
    ).reset_index(drop=True)
    frame["reading_sequence"] = frame.groupby("segment_id").cumcount()
    frame["collision_sequence"] = frame.groupby(
        ["segment_id", "timestamp"]
    ).cumcount()
    frame["timestamp"] = frame["timestamp"] + pd.to_timedelta(
        frame["collision_sequence"], unit="s"
    )
    frame = frame.drop(columns=["_source_order"])
    frame = frame.sort_values(["timestamp", "segment_id", "reading_sequence"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    repaired = repair_timestamps(args.input, args.output)
    collisions = repaired.duplicated(["segment_id", "timestamp"], keep=False).sum()
    print(f"Rows preserved: {len(repaired)}")
    print(f"Remaining timestamp collisions: {collisions}")
    print(f"Saved: {args.output}")
    print("Assumption: colliding sample rows retain source order and are offset by seconds.")


if __name__ == "__main__":
    main()
