"""Compare dashboard-model holdout performance with and without alarm_triggered."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = PROJECT_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from model_pipeline import RANDOM_STATE, SCADARiskModel  # noqa: E402
from scada_adapter import prepare_scada_file  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def evaluate(frame: pd.DataFrame) -> dict:
    model = SCADARiskModel()
    model.has_shap = False
    model.fit(frame)
    report = model.metrics["classification_report"]
    return {
        "classifier": type(model.clf).__name__,
        "feature_count": len(model.feature_columns),
        "isolation_forest_contamination": model.iso_forest.contamination,
        "top_features": [
            {"feature": name, "importance": float(value)}
            for name, value in model.feature_importances.head(10).items()
        ],
        "metrics": {
            "holdout_auc": model.metrics["holdout_auc"],
            "holdout_average_precision": model.metrics["holdout_pr_auc"],
            "positive_class_precision": report["1"]["precision"],
            "positive_class_recall": report["1"]["recall"],
            "positive_class_f1": report["1"]["f1-score"],
            "holdout_train_size": model.metrics["holdout_train_size"],
            "holdout_test_size": model.metrics["holdout_test_size"],
            "holdout_cutoff_timestamp": model.metrics["holdout_cutoff_timestamp"],
            "holdout_start_timestamp": model.metrics["holdout_start_timestamp"],
            "holdout_end_timestamp": model.metrics["holdout_end_timestamp"],
            "confusion_matrix": model.metrics["confusion_matrix"],
            "cv_f1_mean": model.metrics["cv_f1_mean"],
            "cv_folds": model.metrics["cv_folds"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "titas_synthetic_scada.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "titas_scada.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. By default, print the report to stdout.",
    )
    args = parser.parse_args()

    frame, quality = prepare_scada_file(args.data, args.config, require_labels=True)
    if "alarm_triggered" not in frame.columns:
        raise ValueError("Ablation requires the alarm_triggered column")
    if frame["target"].nunique() != 2:
        raise ValueError("Ablation requires both target classes")

    results = {
        "all_features": evaluate(frame),
        "without_alarm_triggered": evaluate(frame.drop(columns=["alarm_triggered"])),
    }
    report = {
        "data_path": str(args.data),
        "data_sha256": file_sha256(args.data),
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "rows": len(frame),
        "positive_rows": int(frame["target"].sum()),
        "data_quality": quality,
        "random_state": RANDOM_STATE,
        "split_method": "chronological holdout; CV folds grouped by timestamp",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "xgboost": installed_version("xgboost"),
            },
        },
        "caveat": (
            "Demo-data ablation only. It measures sensitivity to alarm_triggered; "
            "it does not establish real-world performance or prove temporal leakage."
        ),
        "results": results,
    }
    output = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Saved ablation report: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
