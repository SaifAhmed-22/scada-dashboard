# Titas SCADA Integration

This project is ready for an offline Titas data-mapping phase. It is not a live
SCADA connector and must not be connected to a control network until the tag
namespace, units, quality flags, access controls, and validation results are
approved by the system owner.

## Integration files

- `config/titas_scada.yml` defines the source-to-canonical tag mapping, units,
  expected frequency, ranges, and required fields.
- `webapp/scada_adapter.py` normalizes and validates incoming tables.
- `scripts/train_model.py` validates labeled data and saves a model artifact.
- `tests/test_scada_adapter.py` covers mapping, unit conversion, and bad rows.

## Canonical model contract

The model expects:

```text
timestamp, segment_id, pressure, flow_rate, temperature,
valve_status, pump_state, pump_speed, compressor_state,
energy_consumption, alarm_triggered, event_type, target
```

For inference, `event_type` and `target` are not required. For training, `target`
must be a verified incident label. Do not generate a leak label only because a
reading looks unusual.

## Before using Titas data

1. Replace the placeholder tag names in `config/titas_scada.yml` with the
   historian/interface specification.
2. Confirm pressure, flow, temperature, energy, and speed units.
3. Confirm timestamp timezone and sampling cadence.
4. Define sensor quality and maintenance-state semantics.
5. Remove or lag `alarm_triggered` if it is generated after the incident; using
   a future-derived alarm creates target leakage.
6. Build labels from verified leak, maintenance, sensor-fault, and operational
   event records.
7. Evaluate chronologically and on unseen segments before live shadow mode.
8. Keep the prediction service read-only with respect to the SCADA network.

## Recommended deployment flow

```text
SCADA historian export/API
        -> adapter validation and unit conversion
        -> immutable storage and quality report
        -> offline training and review
        -> versioned model artifact
        -> read-only prediction API
        -> dashboard and operator alert workflow
```

The dashboard's development fallback trains at startup for convenience. For a
controlled deployment, train with `scripts/train_model.py` and set
`PIPELINE_RISK_MODEL_PATH` so serving loads the reviewed artifact instead.