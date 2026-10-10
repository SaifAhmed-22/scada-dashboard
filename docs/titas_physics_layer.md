# Titas physics/context layer

This add-on extracts research-useful information from the BUET thesis
"Production System Analysis of the Titas Gas Field" (1999) and packages it
for the SCADA research model.

## What is included

- `data/titas/titas_well_context.csv`
  - 11 historical well profiles
  - geometry, well type, sand group, location/sales line
  - perforation and safety-valve information
  - historical reservoir pressure/temperature
  - back-pressure coefficient `C` and exponent `n`
  - observed and simulated WHP/WHT/flow baseline

- `data/titas/titas_sensitivity_curves.csv`
  - Tables 8.2-8.7
  - historical modeled responses to reservoir pressure and sales-line pressure scenarios

- `data/titas/titas_gas_properties.yml`
  - historical average gas composition
  - gas gravity
  - critical pressure and temperature

- `webapp/titas_context.py`
  - safe context join
  - geometry features
  - historical baseline deviations
  - documented back-pressure equation
  - nearest documented sensitivity scenario lookup

## Important data-governance rule

The thesis is from 1999. These values are historical/reference data.
They must not be presented as current Titas operating setpoints.

The current demo SCADA dataset uses `segment_id` and does not contain an
approved mapping from segment IDs to TT-1 ... TT-11. The add-on therefore
does not guess this mapping. A real integration should first obtain an
approved `segment_id -> well_id` mapping from field/GIS/historian metadata.

## Recommended research use

1. Add well context after the approved mapping exists.
2. Use static context as covariates.
3. Use baseline ratios/deviations as field-specific features.
4. Add the back-pressure residual only when true flowing bottomhole pressure
   is available or calculated with a separately validated pressure-drop model.
5. Treat the sensitivity curves as historical empirical envelopes, not as a
   replacement for a live hydraulic simulator.
