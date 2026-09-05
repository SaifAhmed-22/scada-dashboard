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
import yaml
from flask import Flask, Response, render_template, jsonify, request

from alert_store import AlertStore
from model_pipeline import SCADARiskModel
from scada_adapter import prepare_scada_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "scada_pipeline.csv")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "titas_scada.yml")
TOPOLOGY_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "demo_topology.yml")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "img")
MODEL_ARTIFACT_PATH = os.getenv("PIPELINE_RISK_MODEL_PATH")
ALERT_DB_PATH = os.getenv(
    "PIPELINE_RISK_ALERT_DB",
    os.path.join(PROJECT_ROOT, "data", "runtime", "alerts.db"),
)

app = Flask(__name__)
alert_store = AlertStore(ALERT_DB_PATH)
with open(TOPOLOGY_CONFIG_PATH, "r", encoding="utf-8") as topology_file:
    topology_config = yaml.safe_load(topology_file) or {}

print("Loading and validating SCADA data...")
raw_df, data_quality = prepare_scada_file(DATA_PATH, CONFIG_PATH, require_labels=True)
if MODEL_ARTIFACT_PATH and os.path.exists(MODEL_ARTIFACT_PATH):
    print(f"Loading persisted model artifact: {MODEL_ARTIFACT_PATH}")
    model, model_metadata = SCADARiskModel.load_artifact(MODEL_ARTIFACT_PATH)
    model.raw_df = raw_df
else:
    print("No model artifact configured; training development model at startup...")
    model = SCADARiskModel()
    model.fit(raw_df)
    model_metadata = {"source": "development_startup_training"}

# Older saved artifacts did not retain raw pressure/flow context in history.
# Backfill those display-only fields without changing model features.
if "pressure" not in model.scored_history.columns or "flow_rate" not in model.scored_history.columns:
    raw_history = raw_df.sort_values(["timestamp", "segment_id"], kind="stable").reset_index(drop=True)
    if len(raw_history) == len(model.scored_history):
        model.scored_history["pressure"] = raw_history["pressure"].to_numpy()
        model.scored_history["flow_rate"] = raw_history["flow_rate"].to_numpy()
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
        last_risk_score=("risk_score_pct", "last"),
        last_timestamp=("timestamp", "last"),
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


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


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
            "data_quality": data_quality,
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


@app.route("/api/operations")
def api_operations():
    scored = model.scored_history.sort_values("timestamp")
    latest = scored.iloc[-1]
    alert_counts = scored["alert_level"].value_counts().to_dict()
    detection_rows = scored[scored["alert_level"] != "Normal"].tail(50).iloc[::-1]

    detections = []
    for _, row in detection_rows.iterrows():
        detections.append({
            "timestamp": str(row["timestamp"]),
            "segment_id": int(row["segment_id"]),
            "risk_score_pct": float(row["risk_score_pct"]),
            "alert_level": row["alert_level"],
            "event_type": row.get("event_type", "unknown"),
            "pressure": float(row["pressure"]),
            "flow_rate": float(row["flow_rate"]),
        })

    detections = alert_store.enrich(detections)
    payload = {
        "summary": {
            "latest_timestamp": str(latest["timestamp"]),
            "latest_segment_id": int(latest["segment_id"]),
            "latest_risk_score": float(latest["risk_score_pct"]),
            "latest_alert_level": latest["alert_level"],
            "critical_count": int(alert_counts.get("Critical Leak Threat", 0)),
            "warning_count": int(alert_counts.get("Warning", 0)),
            "normal_count": int(alert_counts.get("Normal", 0)),
        },
        "detections": detections,
        "segments": _segment_summary.to_dict(orient="records"),
    }
    return jsonify(_clean(payload))


@app.route("/api/topology")
def api_topology():
    segments = sorted(int(value) for value in raw_df["segment_id"].unique())
    per_row = int(topology_config.get("segments_per_row", 25))
    segment_state = _segment_summary.set_index("segment_id").to_dict(orient="index")
    nodes = []
    edges = []
    for index, segment_id in enumerate(segments):
        row, column = divmod(index, per_row)
        nodes.append({
            "id": f"segment-{segment_id}",
            "segment_id": segment_id,
            "label": f"SEG-{segment_id:03d}",
            "x": 34 + column * 30,
            "y": 48 + row * 74,
            "last_risk_score": float(segment_state[segment_id]["last_risk_score"]),
            "last_alert_level": segment_state[segment_id]["last_alert_level"],
        })
        if column > 0:
            edges.append({"from": f"segment-{segments[index - 1]}", "to": f"segment-{segment_id}"})
    return jsonify({
        "source": topology_config.get("source", "demo"),
        "label": topology_config.get("label", "DEMO SCHEMATIC"),
        "viewbox": [0, 0, 780, 190],
        "nodes": nodes,
        "edges": edges,
    })


@app.route("/api/alerts/<alert_key>/<action>", methods=["POST"])
def api_alert_action(alert_key, action):
    data = request.get_json(silent=True) or {}
    try:
        result = alert_store.update(
            alert_key,
            action,
            note=str(data.get("note", "")),
            actor=str(data.get("actor", "operator")),
        )
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@app.route("/api/export/data")
def api_export_data():
    export_df = raw_df
    segment_id = request.args.get("segment_id", type=int)
    if segment_id is not None:
        export_df = raw_df[raw_df["segment_id"] == segment_id]
    response = Response(export_df.to_csv(index=False), mimetype="text/csv")
    filename = "scada_pipeline.csv" if segment_id is None else f"scada_segment_{segment_id}.csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


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
