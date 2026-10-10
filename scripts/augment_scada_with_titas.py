"""Add Titas historical/context features to a SCADA CSV.

Expected input contains a well_id column with TT-1 ... TT-11. This script
does not guess a mapping from existing demo segment IDs to Titas wells.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

from titas_context import add_titas_context_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--well-id-col", default="well_id")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    augmented = add_titas_context_features(
        frame,
        well_id_col=args.well_id_col,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output, index=False)
    print(f"Input rows: {len(frame)}")
    print(f"Output columns: {len(augmented.columns)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
