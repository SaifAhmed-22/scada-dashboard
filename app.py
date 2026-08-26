# ============================================================
# SCADA Pipeline Risk Dashboard - Flask backend
# Trains the model once at startup, then serves:
#   GET  /                      -> dashboard page
#   GET  /api/meta              -> dataset + model metrics
#   GET  /api/segments          -> segment picker list
#   GET  /api/simulate/<id>     -> full walk-forward simulation for a segment
#   POST /api/predict           -> manual "what-if" reading prediction
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

from model_pipeline import SCADARiskModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "scada_pipeline.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "img")

app = Flask(__name__)

print("Loading SCADA data and training model (this runs once at startup)...")
raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
model = SCADARiskModel()
model.fit(raw_df)
model.save_report_plots(PLOTS_DIR)
print(f"Model ready. Holdout AUC={model.metrics.get('holdout_auc')}, "
      f"CV F1={model.metrics.get('cv_f1_mean')}")

# Precompute the segment picker table once (cheap, 50 rows)
_segment_summary = (
    model.scored_history.groupby("segment_id")
    .agg(
        num_readings=("risk_score_pct", "count"),
        num_incidents=("target", "sum"),
        max_risk_score=("risk_score_pct", "max"),
        last_alert_level=("alert_level", "last"),
    )
    .reset_index()
    .sort_values(["num_incidents", "max_risk_score"], ascending=False)
)


def _clean(obj):
    """Make numpy/pandas scalars JSON-safe."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    cr = model.metrics.get("classification_report", {})
    payload = {
        "dataset": {
            "rows": int(len(raw_df)),
            "segments": int(raw_df["segment_id"].nunique()),
            "time_start": str(raw_df["timestamp"].min()),
            "time_end": str(raw_df["timestamp"].max()),
            "event_type_counts": raw_df["event_type"].value_counts().to_dict(),
        },
        "model": {
            "uses_xgboost": model.has_xgboost,
            "uses_shap": model.has_shap,
            "holdout_train_size": model.metrics.get("holdout_train_size"),
            "holdout_test_size": model.metrics.get("holdout_test_size"),
            "holdout_auc": model.metrics.get("holdout_auc"),
            "cv_f1_mean": model.metrics.get("cv_f1_mean"),
            "cv_folds": model.metrics.get("cv_folds"),
            "anomaly_only_auc": model.metrics.get("anomaly_only_auc"),
            "precision_anomalous": cr.get("1", {}).get("precision"),
            "recall_anomalous": cr.get("1", {}).get("recall"),
            "f1_anomalous": cr.get("1", {}).get("f1-score"),
        },
        "risk_weights": {
            "classifier_weight": model.RISK_WEIGHT_CLASSIFIER,
            "anomaly_weight": model.RISK_WEIGHT_ANOMALY,
            "alert_thresholds": model.ALERT_THRESHOLDS,
        },
        "top_features": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in model.feature_importances.head(10).items()
        ],
    }
    return jsonify(_clean(payload))


@app.route("/api/segments")
def api_segments():
    records = _segment_summary.to_dict(orient="records")
    return jsonify(_clean(records))


@app.route("/api/simulate/<int:segment_id>")
def api_simulate(segment_id):
    if segment_id not in raw_df["segment_id"].unique():
        return jsonify({"error": f"Unknown segment_id {segment_id}"}), 404
    results = model.simulate_stream(segment_id)
    return jsonify(_clean(results))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    segment_id = data.pop("segment_id", None)
    use_history = data.pop("use_history", False)

    required = ["pressure", "flow_rate", "temperature", "valve_status",
                "pump_state", "pump_speed", "compressor_state", "energy_consumption"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    data.setdefault("alarm_triggered", 0)

    try:
        if use_history and segment_id is not None:
            segment_id = int(segment_id)
            hist = raw_df[raw_df["segment_id"] == segment_id].sort_values("timestamp")
            if hist.empty:
                return jsonify({"error": f"Unknown segment_id {segment_id}"}), 404
            last_ts = pd.to_datetime(hist["timestamp"]).max()
            reading_row = pd.DataFrame([data])
            reading_row["timestamp"] = last_ts + pd.Timedelta(minutes=1)
            reading_row["segment_id"] = segment_id
            window = pd.concat([hist, reading_row], ignore_index=True)
            result = model.predict_pipeline_risk(window)
        else:
            result = model.predict_pipeline_risk(
                data, segment_id=int(segment_id) if segment_id is not None else 0
            )
    except Exception as exc:  # surface a readable error to the frontend
        return jsonify({"error": str(exc)}), 400

    return jsonify(_clean(result))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
