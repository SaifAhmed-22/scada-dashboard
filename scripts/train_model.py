"""Train and persist a pipeline-risk model from a validated SCADA dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = PROJECT_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from model_pipeline import SCADARiskModel  # noqa: E402
from scada_adapter import prepare_scada_file  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "titas_synthetic_scada.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "titas_scada.yml"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "pipeline_risk_model.pkl"
DEFAULT_METRICS = PROJECT_ROOT / "artifacts" / "models" / "pipeline_risk_metrics.json"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--allow-collisions", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Run scripts/generate_titas_simulation.py first."
        )
    frame, quality = prepare_scada_file(args.data, args.config, require_labels=True)
    collisions = quality.get("timestamp_collision_rows", 0)
    if collisions and not args.allow_collisions:
        raise ValueError(
            f"Refusing to train: {collisions} rows belong to ambiguous same-segment timestamps."
        )
    if frame["target"].nunique() < 2:
        raise ValueError("Training data must contain both target classes")

    model = SCADARiskModel().fit(frame)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data),
        "data_sha256": file_sha256(args.data),
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "dataset_status": "simulated_titas_research_data",
        "rows_used": len(frame),
        "data_quality": quality,
        "model_features": model.feature_columns,
        "model_configuration": {
            "classifier": type(model.clf).__name__,
            "classifier_parameters": model.clf.get_params(deep=False),
            "isolation_forest": {
                "n_estimators": model.iso_forest.n_estimators,
                "contamination": model.iso_forest.contamination,
                "random_state": model.iso_forest.random_state,
            },
        },
        "evaluation": {
            "split_method": "chronological holdout; CV folds grouped by timestamp",
            "holdout_cutoff_timestamp": model.metrics.get("holdout_cutoff_timestamp"),
            "holdout_start_timestamp": model.metrics.get("holdout_start_timestamp"),
            "holdout_end_timestamp": model.metrics.get("holdout_end_timestamp"),
            "metrics": model.metrics,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "xgboost": installed_version("xgboost"),
                "shap": installed_version("shap"),
                "Flask": installed_version("Flask"),
                "PyYAML": installed_version("PyYAML"),
            },
        },
        "metrics": model.metrics,
    }
    model.save_artifact(args.model, metadata)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"Validated rows: {len(frame)}")
    print(f"Wells: {frame['well_id'].nunique()}")
    print(f"Invalid rate: {quality['invalid_rate']:.2%}")
    print(f"Holdout AUC: {model.metrics.get('holdout_auc')}")
    print(f"Saved model: {args.model}")


if __name__ == "__main__":
    main()
