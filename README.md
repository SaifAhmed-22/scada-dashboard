# Titas SCADA Pipeline Risk Assessment

A research-oriented SCADA gas-pipeline anomaly/leak-risk system with a Flask dashboard, chronological ML evaluation, explainability, data-quality controls, and an optional historical Titas field-context/physics layer.

## Quick start

From the repository root in PowerShell:

```powershell
py -m pip install -r requirements.txt
py webapp\app.py
```

Open `http://127.0.0.1:5000` after startup training completes.

The demo application uses `data/processed/scada_pipeline_repaired.csv`. The original sample remains under `data/raw/` for provenance.

## Research architecture

```text
SCADA input
   │
   ├── data validation / unit normalization
   │
   ├── chronological per-segment feature engineering
   │       ├── rolling statistics
   │       ├── pressure rate-of-change
   │       ├── flow anomaly index
   │       └── equipment state-change flags
   │
   ├── optional Titas context (requires explicit well_id)
   │       ├── well geometry
   │       ├── historical operating baselines
   │       ├── reservoir context
   │       └── validated physics residuals when flowing BHP exists
   │
   ├── Isolation Forest anomaly score
   │
   ├── supervised classifier
   │
   └── blended 0–100 risk score + alert level + SHAP factors
```

## Repository layout

```text
data/raw/          Original SCADA sample / provenance
data/processed/    Validated/repaired demo data
data/titas/        Historical Titas well, gas, and sensitivity data
config/             SCADA/topology configuration
docs/               Project and dataset documentation
docs/research/      Research papers and supporting literature
scripts/             Data audit, timestamp repair, training, and Titas augmentation
tests/               Automated tests
artifacts/           Generated model/plot outputs (where present)
webapp/              Flask application, model pipeline, adapters, frontend
```

## Reproducible training

```powershell
py scripts\repair_sample_timestamps.py
py scripts\audit_data.py --data data\processed\scada_pipeline_repaired.csv
py scripts\train_model.py
```

Training uses a chronological holdout and time-series cross-validation. Same-segment timestamp collisions are rejected unless explicitly overridden after source review.

A trained artifact can be served without startup retraining:

```powershell
$env:PIPELINE_RISK_MODEL_PATH = "D:\path\to\pipeline_risk_model.pkl"
py webapp\app.py
```

## Titas research layer

`data/titas/` contains historical information extracted from the supplied 1999 BUET Titas production-system thesis. The model uses this information only when an explicit `well_id` is available; it never guesses a `segment_id -> TT-*` mapping.

The back-pressure relation is used only when explicit flowing bottomhole pressure is available. Pipeline pressure is not silently substituted for bottomhole pressure.

For the publication experiment, compare:

1. SCADA-only baseline.
2. SCADA + Titas static/context features.
3. SCADA + Titas context + validated physics residuals.

Report chronological F1, ROC-AUC, PR-AUC, false-positive rate, confusion matrix, and detection lead time when event timing supports it.

**Historical-data warning:** Titas thesis values are reference information from 1999, not current 2026 operating specifications. Validate against current historian/GIS/well records before operational use.

## Data contract

The current demo SCADA input contains:

```text
timestamp, segment_id, pressure, flow_rate, temperature,
valve_status, pump_state, pump_speed, compressor_state,
energy_consumption, alarm_triggered, event_type, target
```

A Titas-enabled dataset additionally requires an approved `well_id` mapping. Real field deployment should obtain this mapping from authoritative historian/GIS/asset metadata rather than infer it from the demo topology.

## Deployment

The root `Procfile` uses Gunicorn:

```text
gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

The built-in Flask server is intended for development and demonstration only.
