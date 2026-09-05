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
