# Final Research Readiness Report

**Project:** Titas SCADA Pipeline Risk Assessment
**Repository:** `SaifAhmed-22/scada-dashboard`
**Last reviewed:** 2026-10-11

## Executive status

The repository contains a Flask demo dashboard, its SCADA risk model, separate
research utilities, and historical Titas reference data. The research helpers
are not wired into dashboard training or inference. The demo dataset has no
approved `well_id` mapping, so the dashboard reports Titas as **REFERENCE / NOT
MAPPED** and uses zero Titas features.

The dashboard reports the Titas layer status and labels the dataset as a
historical demo replay rather than live SCADA.

## What was completed

### ML pipeline

- Centralized the model implementation in `webapp/model_pipeline.py`.
- Preserved the existing time-series feature engineering and Flask inference path.
- Added separate research helpers for Titas context, geometry, historical
  baseline comparisons, and the documented back-pressure equation. These are
  not currently used to train or serve the dashboard model.
- The dashboard model's Isolation Forest `contamination` is currently set from
  `auto`, independently of training labels.
- The model reports ROC-AUC, average precision (PR-AUC), F1-related metrics,
  and confusion matrix on its chronological holdout. Cross-validation folds are
  grouped by timestamp so rows from one instant are not split across folds.
- The reproducible training report records hashes for the data and config,
  evaluation window, model settings, and package versions.

### Titas research layer

- `data/titas/titas_well_context.csv`: historical profiles for TT-1 through TT-11.
- `data/titas/titas_sensitivity_curves.csv`: historical modeled sensitivity scenarios from thesis Tables 8.2–8.7.
- `data/titas/titas_gas_properties.yml`: historical gas properties/composition.
- `webapp/titas_context.py`: reusable context and physics feature functions.
- `scripts/augment_scada_with_titas.py`: explicit-well-id augmentation utility.
- `tests/test_titas_context.py`: Titas-layer unit tests.
- `docs/titas_physics_layer.md`: integration and research protocol.

### Dashboard integration

- `/api/titas-status` provides a dedicated machine-readable status endpoint.
- The Model Performance view reports whether reference files are available,
  whether the SCADA dataset has `well_id`, and whether any Titas features are
  included in the trained model.
- The current demo status is **REFERENCE / NOT MAPPED**. The Titas feature
  utilities are not wired into the dashboard training or inference pipeline.
- The header identifies the app as a historical demo replay, not a live SCADA
  monitor.

### Repository cleanup

- `scripts/scada_risk_model.py` and `webapp/model_pipeline.py` are both present.
  The former is a separate exploratory implementation; the Flask app uses the
  latter. Use `scripts/train_model.py` for validated artifact training on the
  dashboard model implementation.
- `scripts/create_project_guide.py` is present and regenerates the checked-in
  Word guide; it requires `python-docx`.
- Research PDFs and Markdown text extractions are retained together for source
  fidelity and searchable text; neither format is considered redundant by
  default.

## Current architecture

```text
Raw / processed SCADA
        |
        v
SCADA adapter + quality checks
        |
        v
Per-segment chronological features
        |
        +------------------------------+
        |                              |
        v                              v
Historical Titas research       Existing SCADA features
utilities (separate path)                |
                                       v
                               Model feature matrix
                       |
          +------------+------------+
          |                         |
          v                         v
   Isolation Forest          Supervised classifier
          |                         |
          +------------+------------+
                       v
                 Risk score
                       |
             Alert + explanation
```

## Scientific safeguards

1. No `segment_id -> TT-*` mapping is inferred. An approved `well_id` field/mapping is required.
2. Historical 1999 Titas values are not represented as current operating specifications.
3. Pipeline pressure is not silently substituted for flowing bottomhole pressure in the back-pressure equation.
4. Isolation Forest uses `contamination="auto"` and does not derive its
   threshold from target-label prevalence.
5. Chronological evaluation is retained to reduce future-information leakage.
6. Data-quality flags remain outside the model feature matrix so invalid sensor readings are not automatically treated as physical leak signatures.

## Remaining research blockers

### Blocker 1 — authoritative well mapping

The current demo SCADA contract is keyed by `segment_id`. To activate Titas context on real SCADA rows, obtain an authoritative `segment_id -> well_id` or equivalent asset/historian mapping.

### Blocker 2 — real field labels

The demo target labels are not enough to claim real-world leak-detection performance. For publication, assemble independently verified leak/event periods from historian, maintenance, incident, test, or simulation records and document the labeling procedure.

### Blocker 3 — current operating data

The Titas thesis is historical. Current pressure, flow, equipment, topology, and well status should come from authoritative modern field records before operational conclusions are made.

### Blocker 4 — validated physics residuals

The back-pressure residual should only be used after a validated flowing-bottomhole-pressure source or a separately validated pressure-drop model is available.

## Recommended publication experiment

Run three controlled configurations on exactly the same chronological splits:

| Experiment | Features |
|---|---|
| A — Baseline | SCADA temporal/domain features only |
| B — Context | SCADA + Titas static/context features |
| C — Physics-informed | SCADA + Titas context + validated physics residuals |

The current app implements baseline SCADA features only. The optional
`scripts/feature_ablation.py` utility compares the model with and without
`alarm_triggered`; this sensitivity check is a demo-data diagnostic, not a
complete research ablation or proof of temporal leakage.

On the checked-in 1,000-row demo data (SHA-256
`d067b610ab0acf045d959a9a3588f051b9f53b0167357e4610f1cd7b73bc909d`), using
the same chronological holdout for both runs:

| Configuration | ROC-AUC | Average precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| All features | 0.993 | 0.985 | 1.000 | 0.907 | 0.951 |
| Without `alarm_triggered` | 0.993 | 0.983 | 1.000 | 0.791 | 0.883 |

`alarm_triggered` was the top feature in the all-feature fit (importance
0.485). Removing it reduced recall and F1 on this demo split, while ROC-AUC and
average precision changed very little. This establishes sensitivity to that
input, not temporal leakage: its timing/generation semantics must be checked
against real historian data before using it for early warning.

Report:

- F1
- Precision / Recall
- ROC-AUC
- PR-AUC
- false-positive rate
- confusion matrix
- detection lead time, if event timestamps support it
- calibration, if risk scores are presented probabilistically
- SHAP/global feature importance

The key research claim should be conditional on the experiment: **whether field-specific physical context improves anomaly/leak detection over a SCADA-only baseline**.

## Repository readiness assessment

**Software organization:** Good after consolidation.
**Research architecture:** Good foundation.
**Titas integration:** Research utilities are implemented but are not wired into the dashboard model; authoritative well IDs are still required.
**Physics integration:** Partially implemented; requires validated flowing BHP for residual features.
**Dashboard observability:** The UI labels the app as a historical demo and reports whether Titas reference data is available and whether Titas features are active.
**Reproducibility:** Good structure; final paper still needs pinned environment/version metadata and experiment manifests.
**Publication readiness:** Not yet sufficient for a performance claim until real/validated data, labeling, ablation results, and leakage checks are completed.

## Next highest-value work

1. Obtain/construct the authoritative SCADA-to-well mapping.
2. Run baseline/context/physics ablation experiments.
3. Add experiment manifests and fixed data/model hashes.
4. Validate leak labels independently.
5. Analyze false positives and detection delay.
6. Write the methods and results sections from those measured experiments.
