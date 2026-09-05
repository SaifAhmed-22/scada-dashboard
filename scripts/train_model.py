"""Train and persist a pipeline-risk model from a validated SCADA dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = PROJECT_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from model_pipeline import SCADARiskModel  # noqa: E402
from scada_adapter import prepare_scada_file  # noqa: E402


DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "scada_pipeline.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "titas_scada.yml"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "pipeline_risk_model.pkl"
DEFAULT_METRICS = PROJECT_ROOT / "artifacts" / "models" / "pipeline_risk_metrics.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    frame, quality = prepare_scada_file(args.data, args.config, require_labels=True)
    if frame["target"].nunique() < 2:
        raise ValueError("Training data must contain both target classes")

    model = SCADARiskModel().fit(frame)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data),
        "data_sha256": file_sha256(args.data),
        "config_path": str(args.config),
        "rows_used": len(frame),
        "data_quality": quality,
        "model_features": model.feature_columns,
        "metrics": model.metrics,
    }
    model.save_artifact(args.model, metadata)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"Validated rows: {len(frame)}")
    print(f"Invalid rate: {quality['invalid_rate']:.2%}")
    print(f"Holdout AUC: {model.metrics.get('holdout_auc')}")
    print(f"Saved model: {args.model}")
    print(f"Saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
