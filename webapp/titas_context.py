"""Titas-specific context and physics-informed features.

This module is intentionally non-invasive: it adds features only when the
input SCADA dataframe contains a `well_id` column (or when the caller
provides a well mapping). It never invents a segment->well mapping.

Historical Titas values in this module come from the 1999 BUET thesis and
must be treated as historical/reference context rather than current operating
setpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = PACKAGE_ROOT / "data" / "titas" / "titas_well_context.csv"
SENSITIVITY_PATH = PACKAGE_ROOT / "data" / "titas" / "titas_sensitivity_curves.csv"


def load_titas_context(path: str | Path = CONTEXT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_titas_sensitivity(path: str | Path = SENSITIVITY_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def add_titas_context_features(
    df: pd.DataFrame,
    *,
    well_id_col: str = "well_id",
    context_path: str | Path = CONTEXT_PATH,
) -> pd.DataFrame:
    """Join static Titas well context and derive normalized static features.

    Required input:
        `well_id` identifying TT-1 ... TT-11.

    Derived features include geometry ratios that do not require an assumption
    about current operating units:
        - tubing_area_min/max_in2
        - flowline_area_min/max_in2
        - depth_to_perforation_ratio
        - perforation_fraction_of_depth
        - flowline_length_to_diameter_ratio
        - tubing_length_to_diameter_ratio

    Also adds historical baseline comparison fields when matching columns exist:
        - whp_vs_historical_ratio
        - wht_delta_from_historical_f
        - flow_vs_historical_ratio
    """
    if well_id_col not in df.columns:
        raise ValueError(
            f"Missing '{well_id_col}'. Do not infer a segment-to-well mapping; "
            "supply an approved mapping first."
        )

    context = load_titas_context(context_path)
    join_cols = [
        "well_id", "location", "sales_line", "sand_group", "well_type",
        "completion_year", "production_start_year", "total_depth_ft",
        "perforation_top_ft", "perforation_bottom_ft",
        "effective_perforation_length_ft", "casing_id_in",
        "tubing_id_min_in", "tubing_id_max_in", "total_tubing_length_ft",
        "flowline_id_min_in", "flowline_id_max_in", "total_flowline_length_ft",
        "safety_valve_depth_ft", "reservoir_pressure_psia",
        "reservoir_temperature_f", "backpressure_C", "backpressure_n",
        "observed_whp_psia", "observed_wht_f", "observed_flow_mmscfd",
    ]
    context = context[join_cols].drop_duplicates("well_id")
    out = df.copy()
    out[well_id_col] = out[well_id_col].astype(str)
    context["well_id"] = context["well_id"].astype(str)
    out = out.merge(
        context,
        how="left",
        left_on=well_id_col,
        right_on="well_id",
        validate="many_to_one",
        suffixes=("", "_titas"),
    )
    if well_id_col != "well_id":
        out = out.drop(columns=["well_id"])

    # Geometry features.
    out["tubing_area_min_in2"] = np.pi * (out["tubing_id_min_in"] ** 2) / 4.0
    out["tubing_area_max_in2"] = np.pi * (out["tubing_id_max_in"] ** 2) / 4.0
    out["flowline_area_min_in2"] = np.pi * (out["flowline_id_min_in"] ** 2) / 4.0
    out["flowline_area_max_in2"] = np.pi * (out["flowline_id_max_in"] ** 2) / 4.0
    out["depth_to_perforation_ratio"] = (
        out["perforation_top_ft"] / out["total_depth_ft"]
    )
    out["perforation_fraction_of_depth"] = (
        out["effective_perforation_length_ft"] / out["total_depth_ft"]
    )
    out["flowline_length_to_diameter_ratio"] = (
        out["total_flowline_length_ft"] / out["flowline_id_min_in"].replace(0, np.nan)
    )
    out["tubing_length_to_diameter_ratio"] = (
        out["total_tubing_length_ft"] / out["tubing_id_min_in"].replace(0, np.nan)
    )

    # Historical baseline features; only computed for columns explicitly supplied by caller.
    if "wellhead_pressure_psia" in out.columns:
        out["whp_vs_historical_ratio"] = (
            out["wellhead_pressure_psia"] / out["observed_whp_psia"].replace(0, np.nan)
        )
        out["whp_delta_from_historical_psia"] = (
            out["wellhead_pressure_psia"] - out["observed_whp_psia"]
        )
    if "wellhead_temperature_f" in out.columns:
        out["wht_delta_from_historical_f"] = (
            out["wellhead_temperature_f"] - out["observed_wht_f"]
        )
    if "flow_rate_mmscfd" in out.columns:
        out["flow_vs_historical_ratio"] = (
            out["flow_rate_mmscfd"] / out["observed_flow_mmscfd"].replace(0, np.nan)
        )
        out["flow_delta_from_historical_mmscfd"] = (
            out["flow_rate_mmscfd"] - out["observed_flow_mmscfd"]
        )

    return out


def backpressure_expected_flow_mmscfd(
    reservoir_pressure_psia,
    flowing_bottomhole_pressure_psia,
    C,
    n,
):
    """Historical back-pressure equation from the thesis:
       Qsc = C * (PR^2 - Pwf^2)^n

    This is only valid when the pressures are compatible with the equation's
    definitions. Do not substitute wellhead pressure for flowing bottomhole
    pressure without a separate pressure-drop model.
    """
    pr = np.asarray(reservoir_pressure_psia, dtype=float)
    pwf = np.asarray(flowing_bottomhole_pressure_psia, dtype=float)
    c = np.asarray(C, dtype=float)
    exponent = np.asarray(n, dtype=float)

    drawdown_term = np.maximum(pr**2 - pwf**2, 0.0)
    return c * np.power(drawdown_term, exponent)


def add_backpressure_features(
    df: pd.DataFrame,
    *,
    reservoir_pressure_col: str = "reservoir_pressure_psia",
    flowing_bottomhole_pressure_col: str = "flowing_bottomhole_pressure_psia",
    C_col: str = "backpressure_C",
    n_col: str = "backpressure_n",
    observed_flow_col: str = "flow_rate_mmscfd",
) -> pd.DataFrame:
    """Add physics residuals only when bottomhole pressure is available."""
    required = [
        reservoir_pressure_col,
        flowing_bottomhole_pressure_col,
        C_col,
        n_col,
        observed_flow_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Back-pressure features require: " + ", ".join(missing)
        )

    out = df.copy()
    out["expected_flow_mmscfd_backpressure"] = backpressure_expected_flow_mmscfd(
        out[reservoir_pressure_col],
        out[flowing_bottomhole_pressure_col],
        out[C_col],
        out[n_col],
    )
    expected = out["expected_flow_mmscfd_backpressure"].replace(0, np.nan)
    out["flow_residual_mmscfd"] = (
        out[observed_flow_col] - out["expected_flow_mmscfd_backpressure"]
    )
    out["flow_residual_ratio"] = (
        out["flow_residual_mmscfd"] / expected
    )
    out["absolute_flow_residual_ratio"] = out["flow_residual_ratio"].abs()
    return out


def expected_flow_from_sensitivity(
    df: pd.DataFrame,
    *,
    scenario_type: str,
    scenario_value_col: str,
    well_id_col: str = "well_id",
    target_flow_col: str = "flow_rate_mmscfd",
    sensitivity_path: str | Path = SENSITIVITY_PATH,
) -> pd.DataFrame:
    """Attach the nearest documented thesis sensitivity scenario.

    This is an empirical historical lookup, not a physics solver. The caller
    supplies the scenario variable (e.g. reservoir pressure or sales-line
    pressure). Nearest-neighbour matching avoids inventing interpolation.
    """
    for col in [well_id_col, scenario_value_col]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}'")
    sens = load_titas_sensitivity(sensitivity_path)
    sens = sens[sens["scenario_type"] == scenario_type].copy()
    if sens.empty:
        raise ValueError(f"No sensitivity data for scenario_type={scenario_type!r}")

    left = df.copy()
    left["_row_id"] = np.arange(len(left))
    right = sens[["well_id", "scenario_value", "flow_mmscfd", "whp_psia", "wht_f"]].copy()
    left[well_id_col] = left[well_id_col].astype(str)
    right["well_id"] = right["well_id"].astype(str)

    rows = []
    for _, row in left.iterrows():
        candidates = right[right["well_id"] == str(row[well_id_col])]
        if candidates.empty or pd.isna(row[scenario_value_col]):
            rows.append({
                "_row_id": row["_row_id"],
                "historical_expected_flow_mmscfd": np.nan,
                "historical_expected_whp_psia": np.nan,
                "historical_expected_wht_f": np.nan,
                "historical_scenario_value_used": np.nan,
            })
            continue
        idx = (candidates["scenario_value"] - float(row[scenario_value_col])).abs().idxmin()
        best = candidates.loc[idx]
        rows.append({
            "_row_id": row["_row_id"],
            "historical_expected_flow_mmscfd": best["flow_mmscfd"],
            "historical_expected_whp_psia": best["whp_psia"],
            "historical_expected_wht_f": best["wht_f"],
            "historical_scenario_value_used": best["scenario_value"],
        })

    match = pd.DataFrame(rows)
    out = left.merge(match, on="_row_id", how="left").drop(columns=["_row_id"])
    if target_flow_col in out.columns:
        expected = out["historical_expected_flow_mmscfd"].replace(0, np.nan)
        out["historical_flow_residual_mmscfd"] = (
            out[target_flow_col] - out["historical_expected_flow_mmscfd"]
        )
        out["historical_flow_residual_ratio"] = (
            out["historical_flow_residual_mmscfd"] / expected
        )
    return out
