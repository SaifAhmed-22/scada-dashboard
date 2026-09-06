"""Titas-specific historical context and physics-informed features.

Historical Titas values in this module come from the 1999 BUET thesis
"Production System Analysis of the Titas Gas Field" and are reference
context, not current operating setpoints.

The module requires an explicit `well_id` (TT-1 ... TT-11) and never infers a
segment-to-well mapping. The historical back-pressure equation is exposed only
for cases where flowing bottomhole pressure is explicitly available.
"""
from __future__ import annotations

from pathlib import Path

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
    """Join documented well context and derive normalized geometry/baseline features."""
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
        "simulated_whp_psia", "simulated_wht_f", "simulated_flow_mmscfd",
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
        out["total_flowline_length_ft"] /
        out["flowline_id_min_in"].replace(0, np.nan)
    )
    out["tubing_length_to_diameter_ratio"] = (
        out["total_tubing_length_ft"] /
        out["tubing_id_min_in"].replace(0, np.nan)
    )

    if "wellhead_pressure_psia" in out.columns:
        out["whp_vs_historical_ratio"] = (
            out["wellhead_pressure_psia"] /
            out["observed_whp_psia"].replace(0, np.nan)
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
            out["flow_rate_mmscfd"] /
            out["observed_flow_mmscfd"].replace(0, np.nan)
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
    """Historical thesis equation: Qsc = C * (PR^2 - Pwf^2)^n."""
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
    """Add physics residuals when all required pressure/flow terms exist."""
    required = [
        reservoir_pressure_col,
        flowing_bottomhole_pressure_col,
        C_col,
        n_col,
        observed_flow_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Back-pressure features require: " + ", ".join(missing))

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
    out["flow_residual_ratio"] = out["flow_residual_mmscfd"] / expected
    out["absolute_flow_residual_ratio"] = out["flow_residual_ratio"].abs()
    return out
