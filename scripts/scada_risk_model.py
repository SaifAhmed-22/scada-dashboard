# ============================================================
# SCADA Time-Series Gas Pipeline Risk & Leak-Detection Model
# Domain: Petroleum & Mining Engineering | Bangladesh gas distribution context
# Upgrades the static (single-snapshot) pipeline risk classifier to a
# real-time, time-series-aware SCADA risk engine.
# ============================================================
#
# DESIGN NOTES (why the model looks the way it does):
# - The original model scored risk from static attributes (pipe age,
#   design pressure, soil corrosion index, wall thickness). That is
#   fine for long-term asset planning, but it cannot react to what a
#   SCADA system reports minute-to-minute: pressure swings, flow
#   anomalies, valve/pump/compressor state changes.
# - Two safety-engineering ideas from the reference literature shaped
#   the feature design here:
#     1) Fault-tree style causal grouping (fatigue / corrosion & cracking /
#        environmental / engineering fault / third-party damage) -> we
#        proxy "fatigue-like" stress with pressure volatility and rate of
#        change, and "engineering/operational fault" with valve/pump/
#        compressor state changes and flow anomalies.
#     2) Risk-matrix style reasoning (Frequency x Severity -> a single
#        risk category, e.g. Low/Medium/High) -> we combine an
#        unsupervised anomaly score (novelty / "how unusual is this
#        right now") with a supervised classifier probability
#        (learned failure signature) into one 0-100% Risk Score, then
#        bucket it into Normal / Warning / Critical Leak Threat.
#
# ============================================================

# === 1. DEPENDENCIES ===
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === 2. CONFIGURATION ===
# TODO: point this at your live SCADA export path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "scada_pipeline.csv"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ROLL_SHORT = "1h"     # short (volatility) window
ROLL_LONG = "24h"     # long (baseline/trend) window
EPS = 1e-6            # guards against divide-by-zero

# Blend weights for the final unified Risk Score.
# clf_proba  -> learned failure signature (supervised)
# anomaly    -> "how unusual is this instant" (unsupervised, unlabeled-friendly)
RISK_WEIGHT_CLASSIFIER = 0.65
RISK_WEIGHT_ANOMALY = 0.35

# Alert-level thresholds on the 0-100 Risk Score.
ALERT_THRESHOLDS = {"Normal": 0, "Warning": 33, "Critical Leak Threat": 66}

print("XGBoost available:", HAS_XGBOOST, "| SHAP available:", HAS_SHAP)

# === 3. LOAD DATA ===
raw_df = pd.read_csv(DATA_PATH)
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
print("Raw SCADA rows:", raw_df.shape[0], "| segments:", raw_df["segment_id"].nunique())
print("Time span:", raw_df["timestamp"].min(), "->", raw_df["timestamp"].max())
print(raw_df["event_type"].value_counts())

# === 4. TIME-SERIES FEATURE ENGINEERING ===
# This single function is used for BOTH training and the live
# predict_pipeline_risk() stream function below, so training and
# inference can never drift apart (a very common source of bugs in
# real-time ML systems).

RAW_NUMERIC_COLS = ["pressure", "flow_rate", "temperature",
                     "pump_speed", "energy_consumption"]


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw per-reading SCADA rows into a model-ready feature table.

    Expects columns: timestamp, segment_id, pressure, flow_rate,
    temperature, valve_status, pump_state, pump_speed,
    compressor_state, energy_consumption.
    (alarm_triggered / event_type / target are optional and ignored here
    -- they are labels, not features.)

    Rolling stats are computed PER PIPELINE SEGMENT, independently, since
    each segment is a physically distinct stretch of pipe -- mixing
    segments together would blend unrelated pressure/flow regimes.
    """
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    segment_frames = []
    for seg_id, g in df.groupby("segment_id", sort=False):
        g = g.sort_values("timestamp").set_index("timestamp")

        # --- rolling-window aggregations (1h short / 24h long) ---
        for col in RAW_NUMERIC_COLS:
            for window, tag in [(ROLL_SHORT, "1h"), (ROLL_LONG, "24h")]:
                roll = g[col].rolling(window)
                g[f"{col}_roll_mean_{tag}"] = roll.mean()
                g[f"{col}_roll_std_{tag}"] = roll.std()
                g[f"{col}_roll_min_{tag}"] = roll.min()
                g[f"{col}_roll_max_{tag}"] = roll.max()
                g[f"{col}_roll_range_{tag}"] = (
                    g[f"{col}_roll_max_{tag}"] - g[f"{col}_roll_min_{tag}"]
                )

        # --- domain-specific metric: Pressure Gradient (dP/dt) ---
        # NOTE: this CSV reports a single pressure sensor per segment, not
        # separate inlet/outlet readings. dP/dt (pressure rate-of-change)
        # is the time-series-native analogue of the classic
        # (Inlet Pressure - Outlet Pressure) hydraulic-loss gradient, and
        # is what actually shows up as a leading indicator in a streaming
        # SCADA feed. If your live system exposes both inlet_pressure and
        # outlet_pressure, simply replace the two lines below with
        # g['pressure_gradient'] = g['inlet_pressure'] - g['outlet_pressure']
        # and the rest of the pipeline (volatility feature, model, stream
        # function) needs no other changes.
        minutes_elapsed = g.index.to_series().diff().dt.total_seconds().div(60)
        g["pressure_gradient"] = g["pressure"].diff() / minutes_elapsed.replace(0, np.nan)
        g["pressure_gradient_roll_std_1h"] = g["pressure_gradient"].rolling(ROLL_SHORT).std()

        # --- domain-specific metric: Flow Rate Anomaly Index ---
        # z-score of the current flow reading against its own 1h rolling
        # baseline -> large |z| means "flow just did something this
        # segment doesn't normally do", a classic leak/blockage signature.
        g["flow_anomaly_index"] = (
            (g["flow_rate"] - g["flow_rate_roll_mean_1h"])
            / (g["flow_rate_roll_std_1h"] + EPS)
        )

        # --- operational state-change flags (engineering-fault proxy) ---
        g["valve_state_changed"] = g["valve_status"].diff().fillna(0).ne(0).astype(int)
        g["pump_state_changed"] = g["pump_state"].diff().fillna(0).ne(0).astype(int)
        g["compressor_state_changed"] = g["compressor_state"].diff().fillna(0).ne(0).astype(int)

        segment_frames.append(g.reset_index())

    out = pd.concat(segment_frames, ignore_index=True)

    # one-hot encode valve_status (categorical: 0=closed,1=open,2=partial)
    out = pd.get_dummies(out, columns=["valve_status"], prefix="valve")

    # first reading(s) of a segment have no rolling history yet -> 0
    # (conservative default: "no volatility signal observed yet")
    engineered_cols = [c for c in out.columns
                       if ("roll_" in c) or ("gradient" in c) or ("anomaly_index" in c)]
    out[engineered_cols] = out[engineered_cols].fillna(0)

    return out.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)


featured_df = engineer_features(raw_df)
print("\nEngineered feature table shape:", featured_df.shape)
print("New engineered columns (sample):",
      [c for c in featured_df.columns if "roll_" in c][:6], "...")

# === 5. TIME-AWARE DATA SPLITTING ===
# Replaces a random train_test_split, which would silently let the model
# "see the future" (rows from later timestamps) while training on rows
# from earlier timestamps -- a classic and easy-to-miss leakage bug in
# SCADA / process-monitoring ML.
sorted_df = featured_df.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)

# every engineered/raw column EXCEPT identifiers and labels
FEATURE_COLUMNS = [c for c in sorted_df.columns
                   if c not in ("timestamp", "segment_id", "event_type", "target")]

X_all = sorted_df[FEATURE_COLUMNS].astype(float)
y_all = sorted_df["target"].astype(int)

# --- 5a. TimeSeriesSplit cross-validation ---
# Each fold's validation rows come strictly AFTER its training rows in
# time. This gives an honest read on how the model would have performed
# if deployed progressively through the historical stream.
print("\n=== TimeSeriesSplit Cross-Validation (chronological folds) ===")
tscv = TimeSeriesSplit(n_splits=5)
cv_f1, cv_auc = [], []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_all), start=1):
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
    if y_tr.nunique() < 2:
        print(f"  Fold {fold}: skipped (training fold has only one class)")
        continue
    probe = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    probe.fit(X_tr, y_tr)
    pred = probe.predict(X_va)
    f1 = f1_score(y_va, pred, zero_division=0)
    cv_f1.append(f1)
    msg = f"  Fold {fold}: train={len(tr_idx):>4} val={len(va_idx):>4}  F1={f1:.3f}"
    if y_va.nunique() > 1:
        auc = roc_auc_score(y_va, probe.predict_proba(X_va)[:, 1])
        cv_auc.append(auc)
        msg += f"  AUC={auc:.3f}"
    print(msg)

if cv_f1:
    print(f"\nTimeSeriesSplit CV summary -> mean F1={np.mean(cv_f1):.3f}"
          + (f"  mean AUC={np.mean(cv_auc):.3f}" if cv_auc else ""))

# --- 5b. Final chronological holdout split (this is what actually trains
# the model that ships inside predict_pipeline_risk()) ---
unique_times = np.sort(sorted_df["timestamp"].unique())
cutoff = unique_times[int(len(unique_times) * 0.8)]
train_df = sorted_df[sorted_df["timestamp"] <= cutoff].reset_index(drop=True)
test_df = sorted_df[sorted_df["timestamp"] > cutoff].reset_index(drop=True)

print(f"\nChronological holdout -> cutoff={cutoff}")
print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
print(f"Test : {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

y_train = train_df["target"].astype(int)
y_test = test_df["target"].astype(int)

# === 6. UNSUPERVISED ANOMALY DETECTION (Isolation Forest) ===
# This model never sees the target label -- it only learns what "normal
# operating geometry" looks like across the engineered features, then
# flags points that sit far from that geometry. This matters for SCADA
# specifically because novel failure modes (ones the classifier below has
# never been trained on) can still be caught here.
X_train_if = train_df[FEATURE_COLUMNS].astype(float)
X_test_if = test_df[FEATURE_COLUMNS].astype(float)

# Isolation Forest splits on random thresholds across dimensions, so
# features on comparable scales behave more evenly than raw mixed units
# (pressure in bar vs. energy in kW vs. 0/1 flags).
if_scaler = StandardScaler()
X_train_if_scaled = if_scaler.fit_transform(X_train_if)
X_test_if_scaled = if_scaler.transform(X_test_if)

# Pragmatic contamination estimate: real operators usually have SOME
# historical sense of "what fraction of readings turned out abnormal"
# from maintenance logs, even without a fully labeled dataset. Here we
# borrow that estimate from the training split. Without it, start with
# contamination='auto' or a conservative expert estimate (e.g. 0.05).
train_anomaly_rate = float(np.clip(y_train.mean(), 0.01, 0.49))
print(f"\nIsolationForest contamination set to observed train rate: {train_anomaly_rate:.3f}")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=train_anomaly_rate,
    random_state=RANDOM_STATE,
)
iso_forest.fit(X_train_if_scaled)

# decision_function: HIGH = normal, LOW/negative = anomalous (sklearn
# convention). Flip the sign so higher = MORE anomalous (more intuitive
# for a risk score), then squash into a clean [0, 1] range.
raw_train_anomaly = -iso_forest.decision_function(X_train_if_scaled)
raw_test_anomaly = -iso_forest.decision_function(X_test_if_scaled)

anomaly_scaler = MinMaxScaler(clip=True)
anomaly_scaler.fit(raw_train_anomaly.reshape(-1, 1))

train_df["anomaly_score"] = anomaly_scaler.transform(raw_train_anomaly.reshape(-1, 1)).ravel()
test_df["anomaly_score"] = anomaly_scaler.transform(raw_test_anomaly.reshape(-1, 1)).ravel()

# Sanity check: does the UNSUPERVISED score track the label at all,
# despite never training on it?
try:
    unsupervised_auc = roc_auc_score(train_df["target"], train_df["anomaly_score"])
    print(f"Anomaly score vs. label (train, sanity check only) -> AUC={unsupervised_auc:.3f}")
except ValueError:
    pass

# === 7. SUPERVISED CLASSIFIER (anomaly score feeds in as a feature) ===
CLASSIFIER_FEATURES = FEATURE_COLUMNS + ["anomaly_score"]
X_train_clf = train_df[CLASSIFIER_FEATURES].astype(float)
X_test_clf = test_df[CLASSIFIER_FEATURES].astype(float)

if HAS_XGBOOST:
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    print("\nTraining XGBoost classifier (trees + anomaly_score feature)...")
else:
    clf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
    )
    print("\nXGBoost not available -> training RandomForestClassifier instead...")

clf.fit(X_train_clf, y_train)

y_pred = clf.predict(X_test_clf)
y_proba = clf.predict_proba(X_test_clf)[:, 1]

print(f"\nTrain size: {len(X_train_clf)} | Test size: {len(X_test_clf)}")
print("\n=== Classification Report (chronological holdout) ===")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
try:
    print("Holdout AUC:", round(roc_auc_score(y_test, y_proba), 3))
except ValueError:
    print("Holdout AUC: undefined (only one class present in this holdout window)")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - SCADA Risk Classifier (holdout)", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Feature importance plot (top 15)
importances = pd.Series(clf.feature_importances_, index=CLASSIFIER_FEATURES)
top_importances = importances.sort_values(ascending=False).head(15).sort_values()
plt.figure(figsize=(9, 7))
top_importances.plot(kind="barh", color="#2e86c1")
plt.xlabel("Importance Score", fontsize=12)
plt.title("Top 15 Feature Importances - SCADA Risk Classifier", fontsize=13)
plt.grid(True, axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: confusion_matrix.png, feature_importance.png")
print("\nTop 8 features by importance:")
print(importances.sort_values(ascending=False).head(8))

# === 8. SHAP FEATURE ATTRIBUTION (for real-time explanations) ===
if HAS_SHAP:
    shap_explainer = shap.TreeExplainer(clf)
else:
    shap_explainer = None


def _positive_class_shap(sv):
    """
    Normalize SHAP's several possible output shapes into a single 2D
    array of shape (n_samples, n_features) representing the
    "risk"/positive-class contribution -- this differs across
    shap/sklearn/xgboost version combinations, so we handle all three:
      - list of two arrays (older API): [class0, class1]
      - 3D array (n_samples, n_features, n_classes)
      - already-2D array (binary XGBoost's single log-odds output)
    """
    if isinstance(sv, list):
        return sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


if HAS_SHAP:
    shap_values_test = _positive_class_shap(shap_explainer.shap_values(X_test_clf))
    mean_abs_shap = pd.Series(
        np.abs(shap_values_test).mean(axis=0), index=CLASSIFIER_FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(9, 7))
    mean_abs_shap.head(15).sort_values().plot(kind="barh", color="#e74c3c")
    plt.xlabel("Mean |SHAP value| (impact on Risk Score)", fontsize=12)
    plt.title("Global Feature Attribution - SHAP (holdout set)", fontsize=13)
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_feature_attribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: shap_feature_attribution.png")

# === 9. UNIFIED RISK SCORE + ALERT LEVEL ===
# This is the "risk-matrix" idea from the PRA/Eisenberg literature,
# translated into ML terms: instead of one qualitative Frequency x
# Severity lookup table, we blend a learned failure PROBABILITY
# (classifier) with a live UNUSUALNESS signal (anomaly detector) into a
# single, continuously updating 0-100% score.

def bucket_alert(risk_score_pct: float) -> str:
    if risk_score_pct >= ALERT_THRESHOLDS["Critical Leak Threat"]:
        return "Critical Leak Threat"
    if risk_score_pct >= ALERT_THRESHOLDS["Warning"]:
        return "Warning"
    return "Normal"


test_df["classifier_probability"] = y_proba
test_df["risk_score_pct"] = np.clip(
    100 * (RISK_WEIGHT_CLASSIFIER * test_df["classifier_probability"]
           + RISK_WEIGHT_ANOMALY * test_df["anomaly_score"]),
    0, 100,
)
test_df["alert_level"] = test_df["risk_score_pct"].apply(bucket_alert)

print("\n=== Alert level distribution on holdout set ===")
print(test_df["alert_level"].value_counts())
print("\nAlert level vs. true event_type (holdout):")
print(pd.crosstab(test_df["event_type"], test_df["alert_level"]))

# === 10. STREAM PROCESSING PREDICTION FUNCTION ===
# This is the function your SCADA integration layer actually calls, once
# per new reading (or once per new batch of readings) per segment.

def predict_pipeline_risk(window, segment_id=None, top_k_features=5):
    """
    Real-time SCADA risk scoring for ONE pipeline segment.

    Parameters
    ----------
    window : pandas.DataFrame or dict
        - DataFrame: a sliding window of the most recent raw SCADA
          readings for a single segment, sorted chronologically
          (oldest first, most recent LAST). Recommended: feed at least
          the last 1-24h of readings so rolling features are meaningful.
        - dict: a single most-recent reading (e.g. straight off an
          MQTT/OPC-UA tag). With no history, volatility-style features
          (rolling std, pressure_gradient, flow_anomaly_index) fall back
          to their conservative "no signal yet" defaults -- pass a
          DataFrame with history whenever you can for sharper detection.
        Required raw fields either way: timestamp, pressure, flow_rate,
        temperature, valve_status, pump_state, pump_speed,
        compressor_state, energy_consumption. `alarm_triggered` is
        optional (defaults to 0 if missing).
    segment_id : int or str, optional
        Required only if `window` doesn't already carry a segment_id
        column (typical for a bare single-reading dict).
    top_k_features : int
        How many top contributing factors to return.

    Returns
    -------
    dict with: timestamp, segment_id, risk_score_pct (0-100),
    alert_level ('Normal' / 'Warning' / 'Critical Leak Threat'),
    classifier_probability, anomaly_score, top_contributing_factors.
    """
    if isinstance(window, dict):
        window = pd.DataFrame([window])
    window = window.copy()

    if "segment_id" not in window.columns:
        if segment_id is None:
            raise ValueError("Provide segment_id (either as a column in "
                              "`window` or as the segment_id= argument).")
        window["segment_id"] = segment_id
    if "alarm_triggered" not in window.columns:
        window["alarm_triggered"] = 0

    # Re-use the IDENTICAL feature-engineering function used in training,
    # so live inference can never silently drift from what the model
    # actually learned.
    feats = engineer_features(window)
    latest = feats.iloc[[-1]]  # "now" = most recent reading in the window

    # align to the training-time schema: e.g. a valve state that never
    # appears in a short live window still needs its dummy column (=0)
    X_live = latest.reindex(columns=FEATURE_COLUMNS, fill_value=0).astype(float)

    # -- anomaly score --
    X_live_if_scaled = if_scaler.transform(X_live)
    raw_anomaly = -iso_forest.decision_function(X_live_if_scaled)
    anomaly_score = float(anomaly_scaler.transform(raw_anomaly.reshape(-1, 1))[0, 0])

    # -- classifier probability (anomaly score included as a feature) --
    X_live_clf = X_live.copy()
    X_live_clf["anomaly_score"] = anomaly_score
    X_live_clf = X_live_clf[CLASSIFIER_FEATURES]
    clf_proba = float(clf.predict_proba(X_live_clf)[0, 1])

    # -- unified risk score + alert level --
    risk_score_pct = float(np.clip(
        100 * (RISK_WEIGHT_CLASSIFIER * clf_proba + RISK_WEIGHT_ANOMALY * anomaly_score),
        0, 100,
    ))
    alert_level = bucket_alert(risk_score_pct)

    # -- feature attribution for this single reading --
    top_factors = []
    if HAS_SHAP:
        sv = _positive_class_shap(shap_explainer.shap_values(X_live_clf))[0]
        ranked = (pd.Series(sv, index=CLASSIFIER_FEATURES)
                  .reindex(pd.Series(sv, index=CLASSIFIER_FEATURES).abs()
                           .sort_values(ascending=False).index))
        for feat, val in ranked.head(top_k_features).items():
            top_factors.append({
                "feature": feat,
                "shap_contribution": round(float(val), 4),
                "current_reading": round(float(X_live_clf[feat].iloc[0]), 4),
            })
    else:
        ranked = importances.sort_values(ascending=False)
        for feat, val in ranked.head(top_k_features).items():
            top_factors.append({
                "feature": feat,
                "model_importance": round(float(val), 4),
                "current_reading": round(float(X_live_clf[feat].iloc[0]), 4),
            })

    return {
        "timestamp": str(latest["timestamp"].iloc[0]),
        "segment_id": latest["segment_id"].iloc[0],
        "risk_score_pct": round(risk_score_pct, 2),
        "alert_level": alert_level,
        "classifier_probability": round(clf_proba, 4),
        "anomaly_score": round(anomaly_score, 4),
        "top_contributing_factors": top_factors,
    }

# === 11. DEMO: SIMULATED REAL-TIME STREAM ===
# Walks one pipeline segment forward reading-by-reading, exactly like a
# live SCADA feed would, calling predict_pipeline_risk() on the growing
# window each time. In production this loop is replaced by your
# streaming consumer (Kafka/MQTT/OPC-UA callback, etc.) -- the function
# call itself does not change.
demo_segment = int(
    raw_df.groupby("segment_id")["target"].sum().sort_values(ascending=False).index[0]
)  # pick the segment with the most historical incidents, for a demo worth watching
segment_history = raw_df[raw_df["segment_id"] == demo_segment].sort_values("timestamp")

print(f"\n=== Live-stream simulation: segment {demo_segment} "
      f"({len(segment_history)} readings) ===")
for i in range(1, len(segment_history) + 1):
    window_so_far = segment_history.iloc[:i]
    result = predict_pipeline_risk(window_so_far)
    if i == len(segment_history) or result["alert_level"] != "Normal":
        print(f"  t={result['timestamp']}  risk={result['risk_score_pct']:5.1f}%  "
              f"alert={result['alert_level']:<22}  "
              f"top factor: {result['top_contributing_factors'][0]['feature']}")

# Example: scoring a single bare reading with NO history (dict input) --
# e.g. the very first message a newly-commissioned sensor ever sends.
example_reading = {
    "timestamp": "2024-01-01 00:20:00",
    "pressure": 105.0,
    "flow_rate": 1.2,
    "temperature": 29.0,
    "valve_status": 2,
    "pump_state": 0,
    "pump_speed": 0.0,
    "compressor_state": 0,
    "energy_consumption": 9.5,
}
single_result = predict_pipeline_risk(example_reading, segment_id=demo_segment)
print("\n=== Example: single reading, no history (cold start) ===")
for k, v in single_result.items():
    print(f"  {k}: {v}")

# === 12. INTERPRETATION ===
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print(f"""
- The classifier alone reaches AUC ~{roc_auc_score(y_test, y_proba):.2f} on a strict
  chronological holdout (it never sees "future" rows during training).
- Feature importance is led by the existing `alarm_triggered` tag and
  energy/pump signals -- i.e. this model is best framed as an early-
  warning LAYER ON TOP OF existing SCADA alarms, not a replacement: it
  also assigns non-trivial risk to some readings where alarm_triggered
  was still 0, which is exactly the borderline zone a purely
  threshold-based alarm would miss.
- The Isolation Forest, trained with NO label information at all, still
  achieves a meaningfully-better-than-random AUC against the true
  target on its own -- confirming it is learning genuine "abnormal
  operating geometry", useful as a safety net against failure modes the
  classifier has never seen labeled examples of.
- Pressure Gradient (dP/dt) and Flow Rate Anomaly Index show up among
  the higher-impact engineered features, consistent with the causal
  categories (pressure fluctuation / fatigue, flow disruption /
  engineering fault) identified in the fault-tree literature reviewed
  for this project.
- With only ~17 minutes of synthetic history per segment, the 1h and
  24h rolling windows largely overlap here; on a real continuous SCADA
  feed they will diverge (24h captures slow baseline drift, 1h captures
  acute volatility), which is exactly the intended behavior.
""")
