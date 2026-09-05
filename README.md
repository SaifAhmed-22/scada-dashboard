# Pipeline Risk Assessment

A SCADA pipeline gas-leak risk assessment project with a Flask dashboard, time-series model, research material, and reproducible data layout.

## Quick start

From the repository root in PowerShell:

```powershell
py -m pip install -r requirements.txt
py webapp\app.py
```

Open <http://127.0.0.1:5000> after the model finishes training.

The app reads the canonical dataset from `data/raw/scada_pipeline.csv` and writes dashboard plots to `webapp/static/img/` at startup.

## Repository layout

```text
data/raw/       Canonical SCADA input data
docs/data/       Dataset notes and generated documentation
docs/research/  Research papers and supporting documents
scripts/         Standalone analysis scripts
artifacts/plots/ Reference plots produced by standalone analysis
webapp/          Flask app, model module, templates, and frontend assets
```

## Standalone analysis

```powershell
py scripts\scada_risk_model.py
```

The standalone script resolves the dataset relative to the repository, so it can be run from any working directory.

## Validated model training

The Titas adapter is configured in `config/titas_scada.yml`. Replace its source
tag names and units with the official SCADA historian specification before using
live data. Train a versioned model artifact with:

```powershell
py scripts\audit_data.py
py scripts\train_model.py
```

Training stops when same-segment timestamps collide because rolling features and
pressure gradients need unambiguous chronology. Repair the source export first.
The current sample intentionally reports collisions; `--allow-collisions` exists
only for an explicitly reviewed experiment and should not be used for production
training.

To serve that artifact instead of training during Flask startup, set its path:

```powershell
$env:PIPELINE_RISK_MODEL_PATH = "D:\path\to\pipeline_risk_model.pkl"
py webapp\app.py
```

Without `PIPELINE_RISK_MODEL_PATH`, the app retains development-mode startup
training against the validated CSV.

The adapter normalizes timestamps to UTC, converts configured units, removes
duplicate and invalid rows, and reports data-quality counts. Quality flags are
kept outside the model feature matrix so sensor failures are not mistaken for
leaks.

## Deployment

The root `Procfile` uses Gunicorn with `webapp` as its working directory:

```text
gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

The built-in Flask server is intended for local development and demonstrations.

## Data contract

The input CSV must contain:

```text
timestamp, segment_id, pressure, flow_rate, temperature,
valve_status, pump_state, pump_speed, compressor_state,
energy_consumption, alarm_triggered, event_type, target
```

Replace the file under `data/raw/` and restart the app to retrain the model.
