# Titas research / physics layer

## Purpose

This layer adds historical Titas well context to the SCADA research model. It is designed to support a field-specific, physics-informed anomaly/leak-detection study without pretending that historical thesis values are current operating setpoints.

## Included data

- `data/titas/titas_well_context.csv`: 11 historical well profiles with geometry, grouping, reservoir parameters, back-pressure coefficients/exponents, and observed/simulated baseline values.
- `data/titas/titas_sensitivity_curves.csv`: historical sensitivity scenarios from thesis Tables 8.2-8.7.
- `data/titas/titas_gas_properties.yml`: historical gas composition and gas properties.
- `webapp/titas_context.py`: context join, geometry features, historical baseline deviations, and the documented back-pressure equation.

## Model integration

`webapp/model_pipeline.py` calls the Titas layer after the existing per-segment temporal feature engineering. Titas features are used only when an explicit `well_id` column is present. No `segment_id -> TT-*` mapping is inferred.

The model can use historical baseline deviations such as WHP/flow ratios and geometry features. The back-pressure equation is enabled only when `flowing_bottomhole_pressure_psia` is explicitly supplied; ordinary pipeline pressure is not silently substituted for bottomhole pressure.

## Research protocol

For publication, compare at least:

1. SCADA-only baseline.
2. SCADA + Titas static/context features.
3. SCADA + Titas context + validated physics residuals.

Report chronological holdout F1, ROC-AUC, PR-AUC, false-positive rate, confusion matrix, and detection lead time where event timing supports it.

## Historical-data warning

The source thesis is from 1999. Its values are historical/reference information. They must be validated against current Titas historian/GIS/well records before being described as current operating conditions.

The historical sensitivity tables are empirical reference curves, not a replacement for a validated live hydraulic simulator.
