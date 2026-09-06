# Final Research Readiness Report

**Project:** Titas SCADA Pipeline Risk Assessment  
**Repository:** `SaifAhmed-22/scada-dashboard`  
**Review date:** 2026-09-06

## Executive status

The repository has been consolidated around one importable ML pipeline and an optional Titas historical/context layer. The Titas integration has been merged into `main` through PR #1. The repository now separates source data, processed data, Titas reference data, configuration, documentation, scripts, tests, artifacts, and the Flask application.

## What was completed

### ML pipeline

- Centralized the model implementation in `webapp/model_pipeline.py`.
- Preserved the existing time-series feature engineering and Flask inference path.
- Added optional Titas context features when an explicit `well_id` is supplied.
- Added historical baseline deviation features for pressure, temperature, and flow.
- Added Titas well geometry features for tubing, flowline, depth, and perforation characteristics.
- Added the thesis back-pressure equation only when explicit flowing bottomhole pressure is available.
- Changed Isolation Forest to `contamination="auto"` so the unsupervised detector no longer derives contamination from the target labels.
- Added average-precision/PR-AUC support in the core model metrics.

### Titas research layer

- `data/titas/titas_well_context.csv`: historical profiles for TT-1 through TT-11.
- `data/titas/titas_sensitivity_curves.csv`: historical modeled sensitivity scenarios from thesis Tables 8.2–8.7.
- `data/titas/titas_gas_properties.yml`: historical gas properties/composition.
- `webapp/titas_context.py`: reusable context and physics feature functions.
- `scripts/augment_scada_with_titas.py`: explicit-well-id augmentation utility.
- `tests/test_titas_context.py`: Titas-layer unit tests.
- `docs/titas_physics_layer.md`: integration and research protocol.

### Repository cleanup

- Removed the obsolete duplicate standalone `scripts/scada_risk_model.py` implementation.
- Removed the generated-document helper `scripts/create_project_guide.py` from runtime scripts; the existing guide document remains available under `docs/`.
- Reworked the root README around the research architecture and reproducible workflow.

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
Titas context (optional)        Existing SCADA features
        |                              |
        +--------------+---------------+
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
4. The anomaly detector does not use target labels to set its contamination parameter.
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
**Titas integration:** Implemented but intentionally conditional on authoritative well IDs.  
**Physics integration:** Partially implemented; requires validated flowing BHP for residual features.  
**Reproducibility:** Good structure; final paper still needs pinned environment/version metadata and experiment manifests.  
**Publication readiness:** Not yet sufficient for a performance claim until real/validated data, labeling, ablation results, and leakage checks are completed.

## Next highest-value work

1. Obtain/construct the authoritative SCADA-to-well mapping.
2. Run baseline/context/physics ablation experiments.
3. Add experiment manifests and fixed data/model hashes.
4. Validate leak labels independently.
5. Analyze false positives and detection delay.
6. Write the methods and results sections from those measured experiments.
