# ============================================================
# SCADA Pipeline Risk Model - core ML pipeline (importable module)
# Same modeling approach as scada_risk_model.py, refactored into a
# class so a web server (app.py) can train it once at startup and
# reuse the fitted artifacts for every request.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering for a server process
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


def _positive_class_shap(sv):
    """Normalize SHAP's several possible output shapes to (n_samples, n_features)."""
    if isinstance(sv, list):
        return sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


class SCADARiskModel:
    """
    Time-series SCADA gas-pipeline risk model:
    Isolation Forest (unsupervised anomaly score) -> feeds into ->
    XGBoost/RandomForest classifier -> blended into a 0-100 Risk Score
    and an Alert Level, with SHAP-based feature attribution.
    """

    ROLL_SHORT = "1h"
    ROLL_LONG = "24h"
    EPS = 1e-6
    RAW_NUMERIC_COLS = ["pressure", "flow_rate", "temperature",
                         "pump_speed", "energy_consumption"]
    RISK_WEIGHT_CLASSIFIER = 0.65
    RISK_WEIGHT_ANOMALY = 0.35
    ALERT_THRESHOLDS = {"Normal": 0, "Warning": 33, "Critical Leak Threat": 66}

    def __init__(self):
        self.has_xgboost = HAS_XGBOOST
        self.has_shap = HAS_SHAP
        self.feature_columns = None
        self.classifier_features = None
        self.if_scaler = None
        self.iso_forest = None
        self.anomaly_scaler = None
        self.clf = None
        self.shap_explainer = None
        self.feature_importances = None
        self.metrics = {}
        self.raw_df = None
        self.scored_history = None  # every training row, batch-scored, for dashboards

    # ---------------------------------------------------------------
    # Feature engineering (identical logic used at train time AND at
    # inference time -- this is what prevents train/serve skew)
    # ---------------------------------------------------------------
    def engineer_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

        segment_frames = []
        for seg_id, g in df.groupby("segment_id", sort=False):
            g = g.sort_values("timestamp").set_index("timestamp")

            for col in self.RAW_NUMERIC_COLS:
                for window, tag in [(self.ROLL_SHORT, "1h"), (self.ROLL_LONG, "24h")]:
                    roll = g[col].rolling(window)
                    g[f"{col}_roll_mean_{tag}"] = roll.mean()
                    g[f"{col}_roll_std_{tag}"] = roll.std()
                    g[f"{col}_roll_min_{tag}"] = roll.min()
                    g[f"{col}_roll_max_{tag}"] = roll.max()
                    g[f"{col}_roll_range_{tag}"] = (
                        g[f"{col}_roll_max_{tag}"] - g[f"{col}_roll_min_{tag}"]
                    )

            # Pressure Gradient (dP/dt) -- proxy for Inlet-Outlet gradient;
            # see README for how to swap in true inlet/outlet columns.
            minutes_elapsed = g.index.to_series().diff().dt.total_seconds().div(60)
            g["pressure_gradient"] = g["pressure"].diff() / minutes_elapsed.replace(0, np.nan)
            g["pressure_gradient_roll_std_1h"] = g["pressure_gradient"].rolling(self.ROLL_SHORT).std()

            # Flow Rate Anomaly Index
            g["flow_anomaly_index"] = (
                (g["flow_rate"] - g["flow_rate_roll_mean_1h"])
                / (g["flow_rate_roll_std_1h"] + self.EPS)
            )

            # operational state-change flags
            g["valve_state_changed"] = g["valve_status"].diff().fillna(0).ne(0).astype(int)
            g["pump_state_changed"] = g["pump_state"].diff().fillna(0).ne(0).astype(int)
            g["compressor_state_changed"] = g["compressor_state"].diff().fillna(0).ne(0).astype(int)

            segment_frames.append(g.reset_index())

        out = pd.concat(segment_frames, ignore_index=True)
        out = pd.get_dummies(out, columns=["valve_status"], prefix="valve")

        engineered_cols = [c for c in out.columns
                           if ("roll_" in c) or ("gradient" in c) or ("anomaly_index" in c)]
        out[engineered_cols] = out[engineered_cols].fillna(0)

        return out.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    def bucket_alert(self, risk_score_pct: float) -> str:
        if risk_score_pct >= self.ALERT_THRESHOLDS["Critical Leak Threat"]:
            return "Critical Leak Threat"
        if risk_score_pct >= self.ALERT_THRESHOLDS["Warning"]:
            return "Warning"
        return "Normal"

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    def fit(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df.copy()
        self.raw_df["timestamp"] = pd.to_datetime(self.raw_df["timestamp"])

        featured_df = self.engineer_features(raw_df)
        sorted_df = featured_df.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)

        self.feature_columns = [c for c in sorted_df.columns
                                 if c not in ("timestamp", "segment_id", "event_type", "target")]

        X_all = sorted_df[self.feature_columns].astype(float)
        y_all = sorted_df["target"].astype(int)

        # --- TimeSeriesSplit CV (diagnostic only) ---
        tscv = TimeSeriesSplit(n_splits=5)
        cv_f1 = []
        for tr_idx, va_idx in tscv.split(X_all):
            y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
            if y_tr.nunique() < 2:
                continue
            probe = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)
            probe.fit(X_all.iloc[tr_idx], y_tr)
            cv_f1.append(f1_score(y_va, probe.predict(X_all.iloc[va_idx]), zero_division=0))
        self.metrics["cv_f1_mean"] = float(np.mean(cv_f1)) if cv_f1 else None
        self.metrics["cv_folds"] = len(cv_f1)

        # --- chronological holdout split ---
        unique_times = np.sort(sorted_df["timestamp"].unique())
        cutoff = unique_times[int(len(unique_times) * 0.8)]
        train_df = sorted_df[sorted_df["timestamp"] <= cutoff].reset_index(drop=True)
        test_df = sorted_df[sorted_df["timestamp"] > cutoff].reset_index(drop=True)
        y_train = train_df["target"].astype(int)
        y_test = test_df["target"].astype(int)

        # --- Isolation Forest (unsupervised anomaly detector) ---
        X_train_if = train_df[self.feature_columns].astype(float)
        X_test_if = test_df[self.feature_columns].astype(float)

        self.if_scaler = StandardScaler().fit(X_train_if)
        X_train_if_scaled = self.if_scaler.transform(X_train_if)
        X_test_if_scaled = self.if_scaler.transform(X_test_if)

        train_anomaly_rate = float(np.clip(y_train.mean(), 0.01, 0.49))
        self.iso_forest = IsolationForest(
            n_estimators=200, contamination=train_anomaly_rate, random_state=RANDOM_STATE
        )
        self.iso_forest.fit(X_train_if_scaled)

        raw_train_anomaly = -self.iso_forest.decision_function(X_train_if_scaled)
        raw_test_anomaly = -self.iso_forest.decision_function(X_test_if_scaled)
        self.anomaly_scaler = MinMaxScaler(clip=True).fit(raw_train_anomaly.reshape(-1, 1))

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["anomaly_score"] = self.anomaly_scaler.transform(raw_train_anomaly.reshape(-1, 1)).ravel()
        test_df["anomaly_score"] = self.anomaly_scaler.transform(raw_test_anomaly.reshape(-1, 1)).ravel()

        try:
            self.metrics["anomaly_only_auc"] = float(
                roc_auc_score(train_df["target"], train_df["anomaly_score"])
            )
        except ValueError:
            self.metrics["anomaly_only_auc"] = None

        # --- supervised classifier (anomaly score feeds in as a feature) ---
        self.classifier_features = self.feature_columns + ["anomaly_score"]
        X_train_clf = train_df[self.classifier_features].astype(float)
        X_test_clf = test_df[self.classifier_features].astype(float)

        if self.has_xgboost:
            self.clf = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=RANDOM_STATE,
            )
        else:
            self.clf = RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
            )
        self.clf.fit(X_train_clf, y_train)

        y_pred = self.clf.predict(X_test_clf)
        y_proba = self.clf.predict_proba(X_test_clf)[:, 1]

        self.metrics["holdout_train_size"] = int(len(X_train_clf))
        self.metrics["holdout_test_size"] = int(len(X_test_clf))
        self.metrics["classification_report"] = classification_report(
            y_test, y_pred, digits=3, zero_division=0, output_dict=True
        )
        try:
            self.metrics["holdout_auc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            self.metrics["holdout_auc"] = None
        self.metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

        self.feature_importances = pd.Series(
            self.clf.feature_importances_, index=self.classifier_features
        ).sort_values(ascending=False)

        if self.has_shap:
            self.shap_explainer = shap.TreeExplainer(self.clf)

        # --- batch-score every row for dashboard/history views ---
        anomaly_score, clf_proba, risk, alert = self.score_dataframe(sorted_df)
        self.scored_history = sorted_df[["timestamp", "segment_id", "event_type", "target"]].copy()
        self.scored_history["anomaly_score"] = anomaly_score
        self.scored_history["classifier_probability"] = clf_proba
        self.scored_history["risk_score_pct"] = risk
        self.scored_history["alert_level"] = alert

        return self

    # ---------------------------------------------------------------
    # Batch scoring (already-engineered dataframe -> scores)
    # ---------------------------------------------------------------
    def score_dataframe(self, engineered_df: pd.DataFrame):
        X = engineered_df.reindex(columns=self.feature_columns, fill_value=0).astype(float)
        X_if_scaled = self.if_scaler.transform(X)
        raw_anom = -self.iso_forest.decision_function(X_if_scaled)
        anomaly_score = self.anomaly_scaler.transform(raw_anom.reshape(-1, 1)).ravel()

        X_clf = X.copy()
        X_clf["anomaly_score"] = anomaly_score
        X_clf = X_clf[self.classifier_features]
        clf_proba = self.clf.predict_proba(X_clf)[:, 1]

        risk = np.clip(
            100 * (self.RISK_WEIGHT_CLASSIFIER * clf_proba + self.RISK_WEIGHT_ANOMALY * anomaly_score),
            0, 100,
        )
        alert = np.array([self.bucket_alert(r) for r in risk])
        return anomaly_score, clf_proba, risk, alert

    # ---------------------------------------------------------------
    # Real-time single-window prediction (the production entry point)
    # ---------------------------------------------------------------
    def predict_pipeline_risk(self, window, segment_id=None, top_k_features=5):
        if isinstance(window, dict):
            window = pd.DataFrame([window])
        window = window.copy()
        # normalize dtype early -- a window built by concatenating raw CSV
        # strings with a freshly-built Timestamp (as api_predict's
        # use_history path does) would otherwise leave a mixed-type
        # column that fails to sort later.
        if "timestamp" in window.columns:
            window["timestamp"] = pd.to_datetime(window["timestamp"])

        if "segment_id" not in window.columns:
            if segment_id is None:
                raise ValueError("Provide segment_id (column or argument).")
            window["segment_id"] = segment_id
        if "alarm_triggered" not in window.columns:
            window["alarm_triggered"] = 0
        if "timestamp" not in window.columns:
            window["timestamp"] = pd.Timestamp.now()

        feats = self.engineer_features(window)
        latest = feats.iloc[[-1]]

        X_live = latest.reindex(columns=self.feature_columns, fill_value=0).astype(float)
        X_live_if_scaled = self.if_scaler.transform(X_live)
        raw_anomaly = -self.iso_forest.decision_function(X_live_if_scaled)
        anomaly_score = float(self.anomaly_scaler.transform(raw_anomaly.reshape(-1, 1))[0, 0])

        X_live_clf = X_live.copy()
        X_live_clf["anomaly_score"] = anomaly_score
        X_live_clf = X_live_clf[self.classifier_features]
        clf_proba = float(self.clf.predict_proba(X_live_clf)[0, 1])

        risk_score_pct = float(np.clip(
            100 * (self.RISK_WEIGHT_CLASSIFIER * clf_proba + self.RISK_WEIGHT_ANOMALY * anomaly_score),
            0, 100,
        ))
        alert_level = self.bucket_alert(risk_score_pct)

        top_factors = []
        if self.has_shap:
            sv = _positive_class_shap(self.shap_explainer.shap_values(X_live_clf))[0]
            s = pd.Series(sv, index=self.classifier_features)
            ranked = s.reindex(s.abs().sort_values(ascending=False).index)
            for feat, val in ranked.head(top_k_features).items():
                top_factors.append({
                    "feature": feat,
                    "shap_contribution": round(float(val), 4),
                    "current_reading": round(float(X_live_clf[feat].iloc[0]), 4),
                })
        else:
            ranked = self.feature_importances.sort_values(ascending=False)
            for feat, val in ranked.head(top_k_features).items():
                top_factors.append({
                    "feature": feat,
                    "model_importance": round(float(val), 4),
                    "current_reading": round(float(X_live_clf[feat].iloc[0]), 4),
                })

        def _safe_float(v, default=float("nan")):
            try:
                return float(v) if pd.notna(v) else default
            except (TypeError, ValueError):
                return default

        def _safe_int(v, default=-1):
            try:
                return int(v) if pd.notna(v) else default
            except (TypeError, ValueError):
                return default

        raw_last = window.sort_values("timestamp").iloc[-1]
        return {
            "timestamp": str(latest["timestamp"].iloc[0]),
            "segment_id": int(latest["segment_id"].iloc[0]),
            "risk_score_pct": round(risk_score_pct, 2),
            "alert_level": alert_level,
            "classifier_probability": round(clf_proba, 4),
            "anomaly_score": round(anomaly_score, 4),
            "top_contributing_factors": top_factors,
            "raw": {
                "pressure": _safe_float(raw_last.get("pressure")),
                "flow_rate": _safe_float(raw_last.get("flow_rate")),
                "temperature": _safe_float(raw_last.get("temperature")),
                "valve_status": _safe_int(raw_last.get("valve_status")),
                "pump_state": _safe_int(raw_last.get("pump_state")),
                "pump_speed": _safe_float(raw_last.get("pump_speed")),
                "compressor_state": _safe_int(raw_last.get("compressor_state")),
                "energy_consumption": _safe_float(raw_last.get("energy_consumption")),
                "alarm_triggered": _safe_int(raw_last.get("alarm_triggered"), default=0),
            },
        }

    def simulate_stream(self, segment_id: int):
        """Walk one segment's full raw history forward reading-by-reading,
        exactly like a live SCADA feed, returning one predict_pipeline_risk()
        result per reading (including realistic cold-start behavior at the
        very first few readings)."""
        seg_hist = self.raw_df[self.raw_df["segment_id"] == segment_id].sort_values("timestamp")
        results = []
        for i in range(1, len(seg_hist) + 1):
            r = self.predict_pipeline_risk(seg_hist.iloc[:i])
            last_row = seg_hist.iloc[i - 1]
            r["true_event_type"] = str(last_row.get("event_type", ""))
            r["true_target"] = int(last_row.get("target", 0))
            results.append(r)
        return results

    # ---------------------------------------------------------------
    # Static report plots (same as the standalone script)
    # ---------------------------------------------------------------
    def save_report_plots(self, out_dir: str):
        import os
        os.makedirs(out_dir, exist_ok=True)

        cm = np.array(self.metrics["confusion_matrix"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - SCADA Risk Classifier (holdout)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close()

        top_imp = self.feature_importances.head(15).sort_values()
        plt.figure(figsize=(8, 6))
        top_imp.plot(kind="barh", color="#2e86c1")
        plt.xlabel("Importance Score")
        plt.title("Top 15 Feature Importances")
        plt.grid(True, axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close()
