"""Normalize and validate SCADA data before it reaches the risk model.

The adapter deliberately keeps quality metadata outside the model feature frame.
That lets operators see bad or imputed readings without silently teaching the model
that a communication problem is a leak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CONTINUOUS_COLUMNS = [
    "pressure",
    "flow_rate",
    "temperature",
    "pump_speed",
    "energy_consumption",
]
STATE_COLUMNS = [
    "valve_status",
    "pump_state",
    "compressor_state",
    "alarm_triggered",
]
LABEL_COLUMNS = ["event_type", "target"]


class ScadaValidationError(ValueError):
    """Raised when SCADA data cannot safely be normalized."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not config.get("columns"):
        raise ScadaValidationError("SCADA config must define a columns mapping")
    return config


def _convert_units(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for column, unit_spec in config.get("units", {}).items():
        if column not in result.columns:
            continue
        source = str(unit_spec.get("source", "")).lower()
        target = str(unit_spec.get("target", "")).lower()
        if source == target:
            continue
        if column == "pressure" and source == "psi" and target == "bar":
            result[column] = result[column] * 0.0689475729
        elif column == "pressure" and source == "kpa" and target == "bar":
            result[column] = result[column] * 0.01
        elif column == "flow_rate" and source in {"m3/h", "m³/h"} and target in {"m3/s", "m³/s"}:
            result[column] = result[column] / 3600
        elif column == "temperature" and source in {"f", "fahrenheit"} and target in {"c", "celsius"}:
            result[column] = (result[column] - 32) * 5 / 9
        else:
            raise ScadaValidationError(
                f"Unsupported unit conversion for {column}: {source} -> {target}"
            )
    return result


def _required_columns(config: dict[str, Any], require_labels: bool) -> list[str]:
    key = "required_for_training" if require_labels else "required_for_inference"
    return list(config.get(key, []))


def normalize_scada_frame(
    source_frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    require_labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map, convert, validate, and sort a SCADA frame.

    Invalid rows are removed from the returned model frame and counted in the
    report. Live callers should reject a batch when its invalid-rate is too high.
    """
    if source_frame.empty:
        raise ScadaValidationError("SCADA input is empty")

    mapping = config.get("columns", {})
    missing_sources = [
        source_name for source_name in _required_columns(config, require_labels)
        if mapping.get(source_name) not in source_frame.columns
    ]
    if missing_sources:
        raise ScadaValidationError(
            "Missing configured SCADA columns: " + ", ".join(missing_sources)
        )

    rename_map = {
        source_name: canonical_name
        for canonical_name, source_name in mapping.items()
        if source_name in source_frame.columns and source_name != canonical_name
    }
    frame = source_frame.rename(columns=rename_map).copy()
    required = _required_columns(config, require_labels)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ScadaValidationError("Missing canonical columns: " + ", ".join(missing))

    frame = _convert_units(frame, config)
    report: dict[str, Any] = {
        "input_rows": int(len(frame)),
        "duplicate_rows_removed": 0,
        "timestamp_collision_rows": 0,
        "invalid_rows_removed": 0,
        "out_of_range_counts": {},
        "missing_counts": {},
        "unit_conversions": [],
    }

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    before_duplicates = len(frame)
    frame = frame.sort_values(["segment_id", "timestamp"])
    frame = frame.drop_duplicates(keep="last")
    report["duplicate_rows_removed"] = int(before_duplicates - len(frame))
    report["timestamp_collision_rows"] = int(
        frame.duplicated(["segment_id", "timestamp"], keep=False).sum()
    )

    for column in CONTINUOUS_COLUMNS + STATE_COLUMNS:
        if column not in frame.columns:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        report["missing_counts"][column] = int(frame[column].isna().sum())

    invalid = frame["timestamp"].isna() | frame["segment_id"].isna()
    for column, spec in config.get("ranges", {}).items():
        if column not in frame.columns:
            continue
        if "min" in spec:
            bad = frame[column] < spec["min"]
            report["out_of_range_counts"][column] = int(bad.sum())
            invalid |= bad
        if "max" in spec:
            bad = frame[column] > spec["max"]
            report["out_of_range_counts"][column] = report["out_of_range_counts"].get(column, 0) + int(bad.sum())
            invalid |= bad
        if "allowed" in spec:
            bad = ~frame[column].isin(spec["allowed"]) & frame[column].notna()
            report["out_of_range_counts"][column] = int(bad.sum())
            invalid |= bad

    invalid |= frame[required].isna().any(axis=1)
    report["invalid_rows_removed"] = int(invalid.sum())
    frame = frame.loc[~invalid].sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
    report["output_rows"] = int(len(frame))
    report["invalid_rate"] = round(report["invalid_rows_removed"] / report["input_rows"], 6)
    report["timezone"] = config.get("timezone", "UTC")
    report["expected_frequency"] = config.get("expected_frequency")

    if frame.empty:
        raise ScadaValidationError("No valid SCADA rows remain after validation")
    return frame, report


def prepare_scada_file(
    input_path: str | Path,
    config_path: str | Path,
    *,
    require_labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config(config_path)
    source = pd.read_csv(input_path)
    return normalize_scada_frame(source, config, require_labels=require_labels)
