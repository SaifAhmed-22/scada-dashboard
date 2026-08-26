# SCADA Pipeline Risk Monitor — Web App

A local web dashboard around the SCADA time-series risk model: Isolation
Forest (unsupervised anomaly score) feeding into an XGBoost classifier,
blended into a 0–100% Risk Score with SHAP feature attribution, exactly
as described in `scada_risk_model.py`. This app wraps that same model in
a small Flask API and a browser dashboard so you can *watch* it work
instead of reading console output.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser. The model trains
once at startup (a few seconds) — you'll see "Model ready" printed in
the terminal when it's done.

> Uses Flask's built-in dev server, which is fine for local/demo use.
> Don't expose this directly on the internet as-is.

## What's in the dashboard

**Live Monitor** — pick a pipeline segment on the left (sorted by how
many historical incidents it had). The center panel replays that
segment's SCADA feed reading-by-reading, exactly the way
`predict_pipeline_risk()` would score it in real time (including the
"cold start" behavior on a segment's very first couple of readings,
before rolling stats have any history to work with). Hit play, or drag
the scrubber. The pipe schematic, gauge, and "top contributing factors"
panel all update live off the actual SHAP values for that instant.

**What-If Simulator** — set sensor values by hand and click "Score this
reading." Check "borrow recent history from a real segment" to append
your hypothetical reading onto a real segment's recent trend line
first — this gives the rolling/volatility features real context instead
of the all-zero cold-start defaults, so the risk score reflects an
actual "what if this happened next" scenario rather than an isolated
snapshot.

**Model Performance** — holdout AUC, TimeSeriesSplit CV F1, the
anomaly-only AUC (how well Isolation Forest does completely unlabeled),
top feature importances, and the confusion matrix / feature-importance
plots.

## Project layout

```
app.py              Flask routes + startup training
model_pipeline.py    SCADARiskModel class — feature engineering, Isolation
                      Forest, classifier, SHAP, predict_pipeline_risk()
scada_pipeline.csv   Synthetic SCADA dataset the model trains on
templates/index.html Dashboard page
static/style.css     Dark SCADA-HMI visual theme
static/app.js        All frontend logic (fetch calls, gauge, chart, schematic)
static/img/          Confusion matrix + feature importance plots (generated
                      fresh at every startup)
```

## API endpoints (if you want to script against it directly)

| Method | Path                       | Returns                                              |
|--------|----------------------------|-------------------------------------------------------|
| GET    | `/api/meta`                | dataset summary + model metrics + top feature importances |
| GET    | `/api/segments`             | one row per pipeline segment (for the picker list)     |
| GET    | `/api/simulate/<segment_id>`| full walk-forward risk-scoring of that segment's history |
| POST   | `/api/predict`              | score a single reading (JSON body, see below)          |

`POST /api/predict` body:
```json
{
  "pressure": 76.2, "flow_rate": 4.4, "temperature": 32.0,
  "valve_status": 1, "pump_state": 1, "pump_speed": 1300,
  "compressor_state": 0, "energy_consumption": 27.0,
  "alarm_triggered": 0,

  "use_history": false,
  "segment_id": null
}
```
Set `"use_history": true` and a real `"segment_id"` to append this
reading onto that segment's actual recent history before scoring
(recommended — see "What-If Simulator" above). Leave it `false` for a
pure cold-start single-reading score.

## Swapping in real SCADA data

Replace `scada_pipeline.csv` with your own export using the same
columns (`timestamp, segment_id, pressure, flow_rate, temperature,
valve_status, pump_state, pump_speed, compressor_state,
energy_consumption, alarm_triggered, event_type, target`) and restart
the app — it retrains on whatever CSV is in `DATA_PATH` at startup.
If your SCADA system reports separate inlet/outlet pressure instead of
one sensor per segment, see the comment above `pressure_gradient` in
`model_pipeline.py`'s `engineer_features()` — it's a one-line swap.
