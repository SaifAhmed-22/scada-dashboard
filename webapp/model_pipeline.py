# ============================================================
# SCADA Pipeline Risk Model - core ML pipeline (importable module)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    average_precision_score,
)

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

try:
    from .titas_context import add_titas_context_features, add_backpressure_features
    HAS_TITAS_CONTEXT = True
except ImportError:
    try:
        from titas_context import add_titas_context_features, add_backpressure_features
        HAS_TITAS_CONTEXT = True
    except ImportError:
        HAS_TITAS_CONTEXT = False

RANDOM_STATE = 42


def _positive_class_shap(sv):
    if isinstance(sv, list):
        return sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


class SCADARiskModel:
    ROLL_SHORT = "1h"
    ROLL_LONG = "24h"
    EPS = 1e-6
    RAW_NUMERIC_COLS = ["pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption"]
    RISK_WEIGHT_CLASSIFIER = 0.65
    RISK_WEIGHT_ANOMALY = 0.35
    ALERT_THRESHOLDS = {"Normal": 0, "Warning": 33, "Critical Leak Threat": 66}

    def __init__(self, use_titas_context=True, use_backpressure=True):
        self.has_xgboost = HAS_XGBOOST
        self.has_shap = HAS_SHAP
        self.lightweight = os.getenv("SCADA_LIGHTWEIGHT", "0") == "1"
        self.has_titas_context = bool(use_titas_context and HAS_TITAS_CONTEXT)
        self.use_backpressure = bool(use_backpressure and HAS_TITAS_CONTEXT)
        self.titas_context_coverage = 0.0
        self.titas_backpressure_coverage = 0.0
        self.titas_feature_columns = []
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
        self.scored_history = None
        self.iqr_bounds = {}

    def save_artifact(self, path, metadata=None):
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        explainer = self.shap_explainer
        self.shap_explainer = None
        try:
            with artifact_path.open("wb") as handle:
                pickle.dump({"model": self, "metadata": metadata or {}}, handle)
        finally:
            self.shap_explainer = explainer

    @classmethod
    def load_artifact(cls, path):
        artifact_path = Path(path)
        with artifact_path.open("rb") as handle:
            payload = pickle.load(handle)
        model = payload["model"]
        if model.has_shap and model.clf is not None:
            model.shap_explainer = shap.TreeExplainer(model.clf)
        return model, payload.get("metadata", {})

    def _add_titas_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.has_titas_context or "well_id" not in df.columns:
            self.titas_feature_columns = []
            self.titas_context_coverage = 0.0
            self.titas_backpressure_coverage = 0.0
            return df
        out = add_titas_context_features(df, well_id_col="well_id")
        candidates = [
            "completion_year", "production_start_year", "total_depth_ft", "perforation_top_ft",
            "perforation_bottom_ft", "effective_perforation_length_ft", "casing_id_in",
            "tubing_id_min_in", "tubing_id_max_in", "total_tubing_length_ft", "flowline_id_min_in",
            "flowline_id_max_in", "total_flowline_length_ft", "safety_valve_depth_ft", "reservoir_pressure_psia",
            "reservoir_temperature_f", "backpressure_C", "backpressure_n", "observed_whp_psia",
            "observed_wht_f", "observed_flow_mmscfd", "simulated_whp_psia", "simulated_wht_f",
            "simulated_flow_mmscfd", "tubing_area_min_in2", "tubing_area_max_in2", "flowline_area_min_in2",
            "flowline_area_max_in2", "depth_to_perforation_ratio", "perforation_fraction_of_depth",
            "flowline_length_to_diameter_ratio", "tubing_length_to_diameter_ratio", "whp_vs_historical_ratio",
            "whp_delta_from_historical_psia", "wht_delta_from_historical_f", "flow_vs_historical_ratio",
            "flow_delta_from_historical_mmscfd",
        ]
        selected = [c for c in candidates if c in out.columns]
        if self.use_backpressure and "flowing_bottomhole_pressure_psia" in out.columns:
            flow_col = "flow_rate_mmscfd" if "flow_rate_mmscfd" in out.columns else None
            if flow_col is None and "flow_rate" in out.columns:
                out["flow_rate_mmscfd"] = pd.to_numeric(out["flow_rate"], errors="coerce")
                flow_col = "flow_rate_mmscfd"
            if flow_col is not None:
                try:
                    out = add_backpressure_features(out, flowing_bottomhole_pressure_col="flowing_bottomhole_pressure_psia", observed_flow_col=flow_col)
                    selected.extend(["expected_flow_mmscfd_backpressure", "flow_residual_mmscfd", "flow_residual_ratio", "absolute_flow_residual_ratio"])
                except ValueError:
                    pass
        present = [c for c in selected if c in out.columns]
        self.titas_feature_columns = present
        if present:
            self.titas_context_coverage = float(out["well_id"].notna().mean())
            if "absolute_flow_residual_ratio" in out.columns:
                self.titas_backpressure_coverage = float(out["absolute_flow_residual_ratio"].notna().mean())
        return out

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
                    g[f"{col}_roll_range_{tag}"] = g[f"{col}_roll_max_{tag}"] - g[f"{col}_roll_min_{tag}"]
            minutes_elapsed = g.index.to_series().diff().dt.total_seconds().div(60)
            g["pressure_gradient"] = g["pressure"].diff() / minutes_elapsed.replace(0, np.nan)
            g["pressure_gradient_roll_std_1h"] = g["pressure_gradient"].rolling(self.ROLL_SHORT).std()
            g["flow_anomaly_index"] = (g["flow_rate"] - g["flow_rate_roll_mean_1h"]) / (g["flow_rate_roll_std_1h"] + self.EPS)
            g["valve_state_changed"] = g["valve_status"].diff().fillna(0).ne(0).astype(int)
            g["pump_state_changed"] = g["pump_state"].diff().fillna(0).ne(0).astype(int)
            g["compressor_state_changed"] = g["compressor_state"].diff().fillna(0).ne(0).astype(int)
            segment_frames.append(g.reset_index())
        out = pd.concat(segment_frames, ignore_index=True)
        if "valve_status" in out.columns:
            out = pd.get_dummies(out, columns=["valve_status"], prefix="valve")
        engineered_cols = [c for c in out.columns if ("roll_" in c) or ("gradient" in c) or ("anomaly_index" in c)]
        out[engineered_cols] = out[engineered_cols].fillna(0)
        return self._add_titas_features(out).sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    def bucket_alert(self, risk_score_pct: float) -> str:
        if risk_score_pct >= self.ALERT_THRESHOLDS["Critical Leak Threat"]:
            return "Critical Leak Threat"
        if risk_score_pct >= self.ALERT_THRESHOLDS["Warning"]:
            return "Warning"
        return "Normal"

    def _fit_iqr_bounds(self, raw_df: pd.DataFrame):
        self.iqr_bounds = {}
        for column in self.RAW_NUMERIC_COLS:
            values = pd.to_numeric(raw_df[column], errors="coerce").dropna()
            if not values.empty:
                q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
                spread = q3 - q1
                self.iqr_bounds[column] = {"lower": q1 - 1.5 * spread, "upper": q3 + 1.5 * spread}

    def add_iqr_features(self, engineered_df: pd.DataFrame) -> pd.DataFrame:
        result = engineered_df.copy()
        for column, bounds in self.iqr_bounds.items():
            result[f"{column}_iqr_outlier"] = ((result[column] < bounds["lower"]) | (result[column] > bounds["upper"])).astype(int)
        return result

    def _numeric_model_features(self, df: pd.DataFrame, non_feature_columns):
        return [c for c in df.columns if c not in non_feature_columns and pd.api.types.is_numeric_dtype(df[c])]

    def fit(self, raw_df: pd.DataFrame):
        required = {"timestamp", "segment_id", "target"} | set(self.RAW_NUMERIC_COLS)
        missing = sorted(required - set(raw_df.columns))
        if missing:
            raise ValueError("Missing required SCADA columns: " + ", ".join(missing))
        self.raw_df = raw_df.copy()
        self.raw_df["timestamp"] = pd.to_datetime(self.raw_df["timestamp"])
        raw_for_cutoff = self.raw_df.copy()
        unique_times = np.sort(raw_for_cutoff["timestamp"].unique())
        if len(unique_times) < 2:
            raise ValueError("Need at least two unique timestamps for chronological evaluation.")
        cutoff = unique_times[min(int(len(unique_times) * 0.8), len(unique_times) - 1)]
        self._fit_iqr_bounds(raw_for_cutoff[raw_for_cutoff["timestamp"] <= cutoff])
        featured_df = self.add_iqr_features(self.engineer_features(raw_df))
        sorted_df = featured_df.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
        non_feature_columns = {"timestamp", "original_timestamp", "segment_id", "reading_sequence", "collision_sequence", "event_type", "target", "well_id"}
        self.feature_columns = self._numeric_model_features(sorted_df, non_feature_columns)
        X_all = sorted_df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_all = sorted_df["target"].astype(int)

        # The full research pipeline keeps five chronological probe folds. The
        # deployed 512-MiB service skips those extra forests to control memory.
        cv_f1 = []
        if not self.lightweight:
            tscv = TimeSeriesSplit(n_splits=5)
            for tr_idx, va_idx in tscv.split(X_all):
                y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
                if y_tr.nunique() < 2:
                    continue
                probe = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=1)
                probe.fit(X_all.iloc[tr_idx], y_tr)
                cv_f1.append(f1_score(y_va, probe.predict(X_all.iloc[va_idx]), zero_division=0))
        self.metrics["cv_f1_mean"] = float(np.mean(cv_f1)) if cv_f1 else None
        self.metrics["cv_folds"] = len(cv_f1)

        train_df = sorted_df[sorted_df["timestamp"] <= cutoff].reset_index(drop=True)
        test_df = sorted_df[sorted_df["timestamp"] > cutoff].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise ValueError("Chronological holdout split produced an empty train or test set.")
        y_train, y_test = train_df["target"].astype(int), test_df["target"].astype(int)
        X_train_if = train_df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_test_if = test_df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.if_scaler = StandardScaler().fit(X_train_if)
        X_train_if_scaled = self.if_scaler.transform(X_train_if)
        X_test_if_scaled = self.if_scaler.transform(X_test_if)
        self.iso_forest = IsolationForest(n_estimators=60 if self.lightweight else 200, contamination="auto", random_state=RANDOM_STATE, n_jobs=1)
        self.iso_forest.fit(X_train_if_scaled)
        raw_train_anomaly = -self.iso_forest.decision_function(X_train_if_scaled)
        raw_test_anomaly = -self.iso_forest.decision_function(X_test_if_scaled)
        self.anomaly_scaler = MinMaxScaler(clip=True).fit(raw_train_anomaly.reshape(-1, 1))
        train_df = train_df.copy(); test_df = test_df.copy()
        train_df["anomaly_score"] = self.anomaly_scaler.transform(raw_train_anomaly.reshape(-1, 1)).ravel()
        test_df["anomaly_score"] = self.anomaly_scaler.transform(raw_test_anomaly.reshape(-1, 1)).ravel()
        try: self.metrics["anomaly_only_auc_train"] = float(roc_auc_score(train_df["target"], train_df["anomaly_score"]))
        except ValueError: self.metrics["anomaly_only_auc_train"] = None
        try: self.metrics["anomaly_only_pr_auc_test"] = float(average_precision_score(y_test, test_df["anomaly_score"]))
        except ValueError: self.metrics["anomaly_only_pr_auc_test"] = None
        self.classifier_features = self.feature_columns + ["anomaly_score"]
        X_train_clf, X_test_clf = train_df[self.classifier_features].astype(float), test_df[self.classifier_features].astype(float)
        if self.has_xgboost:
            self.clf = XGBClassifier(
                n_estimators=80 if self.lightweight else 300,
                max_depth=3 if self.lightweight else 4,
                learning_rate=0.08, subsample=0.85 if self.lightweight else 0.9,
                colsample_bytree=0.8 if self.lightweight else 0.9,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
            )
        else:
            self.clf = RandomForestClassifier(n_estimators=80 if self.lightweight else 300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1)
        self.clf.fit(X_train_clf, y_train)
        y_pred, y_proba = self.clf.predict(X_test_clf), self.clf.predict_proba(X_test_clf)[:, 1]
        self.metrics["holdout_train_size"] = int(len(X_train_clf)); self.metrics["holdout_test_size"] = int(len(X_test_clf))
        self.metrics["classification_report"] = classification_report(y_test, y_pred, digits=3, zero_division=0, output_dict=True)
        try: self.metrics["holdout_auc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError: self.metrics["holdout_auc"] = None
        try: self.metrics["holdout_pr_auc"] = float(average_precision_score(y_test, y_proba))
        except ValueError: self.metrics["holdout_pr_auc"] = None
        self.metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        self.metrics["titas_context_enabled"] = self.has_titas_context
        self.metrics["titas_context_coverage"] = round(self.titas_context_coverage, 4)
        self.metrics["titas_backpressure_enabled"] = self.use_backpressure
        self.metrics["titas_backpressure_coverage"] = round(self.titas_backpressure_coverage, 4)
        self.metrics["titas_feature_count"] = int(len(self.titas_feature_columns))
        self.feature_importances = pd.Series(self.clf.feature_importances_, index=self.classifier_features).sort_values(ascending=False)
        if self.has_shap:
            self.shap_explainer = shap.TreeExplainer(self.clf)
        return self
