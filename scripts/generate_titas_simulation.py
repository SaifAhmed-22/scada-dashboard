"""Generate a reproducible, Titas-specific synthetic SCADA research dataset.

This is a simulation, not field telemetry. It uses the historical well-context
records already stored in data/titas/titas_well_context.csv only to parameterize
well-specific baseline behavior and the historical back-pressure relation.

Each generated segment is explicitly paired with one Titas well by construction:
SEG-001 -> TT-1, ..., SEG-011 -> TT-11. This mapping is valid only for this
synthetic research dataset and must not be transferred to live field data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = PROJECT_ROOT / "data" / "titas" / "titas_well_context.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "titas_synthetic_scada.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "titas" / "simulation_manifest.yml"
MINUTES = 2 * 24 * 60
SEED = 20260906
PSIA_PER_BAR = 14.5037738
M3S_PER_MMSCFD = 1e6 * 0.028316846592 / 86400


def _context() -> pd.DataFrame:
    ctx = pd.read_csv(CONTEXT_PATH)
    required = {
        "well_id", "location", "reservoir_pressure_psia", "reservoir_temperature_f",
        "backpressure_C", "backpressure_n", "observed_whp_psia", "observed_wht_f",
        "observed_flow_mmscfd",
    }
    missing = sorted(required - set(ctx.columns))
    if missing:
        raise ValueError(f"Titas context missing required columns: {missing}")
    ctx["pwf_psia"] = np.sqrt(
        np.maximum(
            ctx["reservoir_pressure_psia"] ** 2
            - (ctx["observed_flow_mmscfd"] / ctx["backpressure_C"]) ** (1 / ctx["backpressure_n"]),
            0,
        )
    )
    ctx["baseline_pressure_bar"] = ctx["observed_whp_psia"] / PSIA_PER_BAR
    ctx["baseline_temperature_c"] = (ctx["observed_wht_f"] - 32) * 5 / 9
    ctx["baseline_flow_m3s"] = ctx["observed_flow_mmscfd"] * M3S_PER_MMSCFD
    return ctx


def generate(output: Path = DEFAULT_OUTPUT, *, seed: int = SEED, days: int = 2) -> pd.DataFrame:
    ctx = _context()
    minutes = days * 24 * 60
    rng = np.random.default_rng(seed)
    times = pd.date_range(
        start="2025-01-01 00:00", periods=minutes, freq="min", tz="Asia/Dhaka"
    )
    records: list[dict] = []

    for w_i, w in ctx.iterrows():
        starts = np.array([
            180 + w_i * 37,
            720 + w_i * 53,
            1320 + w_i * 41,
            2200 + w_i * 29,
            2860 + w_i * 17,
        ])
        durations = np.array([
            45 + (w_i % 4) * 5,
            70 + (w_i % 3) * 7,
            55 + (w_i % 5) * 6,
            85 + (w_i % 4) * 4,
            50 + (w_i % 3) * 8,
        ])
        event = np.array(["normal"] * minutes, dtype=object)
        target = np.zeros(minutes, dtype=int)
        for st, dur, etype in zip(
            starts, durations, ["leak", "blockage", "leak", "pump_trip", "leak"]
        ):
            start_i = int(st)
            end_i = min(start_i + int(dur), minutes)
            event[start_i:end_i] = etype
            if etype in {"leak", "blockage"}:
                target[start_i:end_i] = 1

        t = np.arange(minutes)
        daily = np.sin(2 * np.pi * t / (24 * 60))
        short = np.sin(2 * np.pi * t / 180 + w_i)
        p = w.baseline_pressure_bar * (1 + 0.015 * daily + 0.008 * short) + rng.normal(0, 0.35, minutes)
        temp = w.baseline_temperature_c + 0.6 * daily + 0.2 * short + rng.normal(0, 0.18, minutes)
        flow = w.baseline_flow_m3s * (1 + 0.018 * daily + 0.006 * short) + rng.normal(0, 0.08, minutes)
        pump_state = np.ones(minutes, dtype=int)
        compressor_state = np.ones(minutes, dtype=int)
        pump_speed = 1450 + 90 * daily + rng.normal(0, 25, minutes)
        energy = 28 + 4 * daily + 0.015 * (pump_speed - 1350) + rng.normal(0, 1.2, minutes)
        valve = np.ones(minutes, dtype=int)

        leak = event == "leak"
        block = event == "blockage"
        trip = event == "pump_trip"
        p[leak] *= 0.84
        flow[leak] *= 0.72
        temp[leak] -= 1.8
        energy[leak] *= 0.92
        p[block] *= 1.07
        flow[block] *= 0.63
        temp[block] += 0.7
        energy[block] *= 1.05
        pump_state[trip] = 0
        compressor_state[trip] = 0
        pump_speed[trip] = 0
        flow[trip] *= 0.45
        energy[trip] *= 0.35
        valve[block] = 2

        p = np.clip(p, 60, 220)
        flow = np.clip(flow, 0.05, 12)
        temp = np.clip(temp, 0, 80)
        pump_speed = np.clip(pump_speed, 0, 1900)
        energy = np.clip(energy, 0.5, 80)

        whp_psia = p * PSIA_PER_BAR
        pwf = w.pwf_psia + (whp_psia - w.observed_whp_psia) * 0.65 + rng.normal(0, 7.0, minutes)
        pwf = np.clip(pwf, 500, w.reservoir_pressure_psia - 5)
        flow_mmscfd = flow / M3S_PER_MMSCFD
        alarm = ((p < w.baseline_pressure_bar * 0.86) | (flow < w.baseline_flow_m3s * 0.66)).astype(int)
        flip = rng.random(minutes) < 0.015
        alarm[flip] = 1 - alarm[flip]

        for j, ts in enumerate(times):
            records.append({
                "timestamp": ts.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S"),
                "segment_id": int(w_i + 1),
                "well_id": w.well_id,
                "pressure": round(float(p[j]), 4),
                "flow_rate": round(float(flow[j]), 5),
                "temperature": round(float(temp[j]), 4),
                "valve_status": int(valve[j]),
                "pump_state": int(pump_state[j]),
                "pump_speed": round(float(pump_speed[j]), 2),
                "compressor_state": int(compressor_state[j]),
                "energy_consumption": round(float(energy[j]), 3),
                "alarm_triggered": int(alarm[j]),
                "event_type": event[j],
                "target": int(target[j]),
                "wellhead_pressure_psia": round(float(whp_psia[j]), 3),
                "wellhead_temperature_f": round(float(temp[j] * 9 / 5 + 32), 3),
                "flow_rate_mmscfd": round(float(flow_mmscfd[j]), 5),
                "flowing_bottomhole_pressure_psia": round(float(pwf[j]), 3),
            })

    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    frame = generate(args.output, seed=args.seed, days=args.days)
    print(f"Generated rows: {len(frame)}")
    print(f"Wells: {frame['well_id'].nunique()}")
    print(f"Positive-event rate: {frame['target'].mean():.2%}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
