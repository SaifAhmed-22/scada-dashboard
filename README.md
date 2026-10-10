# Pipeline Risk Assessment

A SCADA pipeline gas-leak risk assessment project with a Flask dashboard, time-series model, research material, and reproducible data layout.

For a plain-language team overview, see [Pipeline Risk Project Guide](docs/Pipeline_Risk_Project_Guide.docx).

## Quick start

From the repository root in PowerShell:

```powershell
py -m pip install -r requirements.txt
py webapp\app.py
```

Open <http://127.0.0.1:5000> after the model finishes training.

The app reads the collision-repaired demo dataset from `data/processed/scada_pipeline_repaired.csv` and writes dashboard plots to `webapp/static/img/` at startup. The original source sample remains in `data/raw/` for provenance.

Run the test suite from the repository root with:

```powershell
py -m unittest discover -s tests -v
```

Compare model performance with and without the existing alarm flag using the
same chronological split:

```powershell
py scripts\feature_ablation.py --output artifacts\models\alarm_feature_ablation.json
```

This is a demo-data sensitivity check, not proof of field performance or
temporal leakage. The ablation report records data/config hashes, split dates,
model metrics, and package versions.

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

This is a separate exploratory implementation that generates standalone
analysis plots; it is not the model implementation used by the Flask app.
For validated model artifacts, prefer `scripts\train_model.py`, which uses the
same `webapp\model_pipeline.py` implementation as the dashboard.

## Research materials

- `docs/research/*.pdf` files preserve the source papers. The corresponding
  Markdown files are searchable text extractions; keeping both formats is
  intentional.
- `data/titas/` and `webapp/titas_context.py` contain historical research
  references and optional feature utilities. These features are not part of
  the demo dashboard model because the SCADA sample has no approved
  `well_id` mapping.
- `scripts/augment_scada_with_titas.py` requires an explicit `well_id` column.
  It augments a CSV but does not train the dashboard model or map segment IDs
  to wells.
- `scripts/generate_titas_simulation.py` creates synthetic research data using
  historical parameters. Its output is simulated data, not field telemetry.
- `scripts/create_project_guide.py` regenerates the checked-in Word guide and
  requires the `python-docx` package from `requirements.txt`.

## Validated model training

The Titas adapter is configured in `config/titas_scada.yml`. Replace its source
tag names and units with the official SCADA historian specification before using
live data. Train a versioned model artifact with:

```powershell
py scripts\repair_sample_timestamps.py
py scripts\audit_data.py --data data\processed\scada_pipeline_repaired.csv
py scripts\train_model.py
```

Training stops when same-segment timestamps collide because rolling features and
pressure gradients need unambiguous chronology. The demo sample has a documented
repair script that preserves every row and offsets collisions by seconds. Real
Titas data must use a true historian sequence or source timestamp instead.
The training script writes model metrics plus a reproducibility record containing
the data/config SHA-256 hashes, chronological holdout window, model settings,
and installed package versions. The model uses `contamination="auto"` for its
Isolation Forest and reports average precision (PR-AUC) alongside ROC-AUC.

To serve that artifact instead of training during Flask startup, set its path:

```powershell
$env:PIPELINE_RISK_MODEL_PATH = "D:\path\to\pipeline_risk_model.pkl"
py webapp\app.py
```

Without `PIPELINE_RISK_MODEL_PATH`, the app retains development-mode startup
training against the validated CSV.
When `PIPELINE_RISK_MODEL_PATH` is set, the app exits with a clear error if the
path does not point to a file; it does not silently train a replacement model.

The adapter normalizes timestamps to UTC, converts configured units, removes
duplicate and invalid rows, and reports data-quality counts. Quality flags are
kept outside the model feature matrix so sensor failures are not mistaken for
leaks.

The current demo model includes `alarm_triggered` as a predictor. Exclude it
from production training unless its timestamp and generation logic establish
that the alarm was available before the prediction time. Check the feature
ablation report and verify this against real historian semantics before making
any early-warning claim.

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
