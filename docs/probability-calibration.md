# Probability calibration and reliability evaluation

Calibration is an **optional, separate development experiment**. Existing published models, holdout predictions, metrics, risk scores and capacity-planning results are unchanged. The final holdout has already been inspected and is excluded from this workflow. No new independent evaluation cohort is available: **independent final performance is not established**.

## Architecture rationale and repository audit

This feature starts after merged capacity-planning PR #6. The audit identified reusable boundaries:

| Existing component | Reuse |
| --- | --- |
| `data.load_consumption`, `ConsumptionData.subset` | Input/label validation and customer-aligned subsets |
| `features.build_features`, `modeling.candidate_models` | Existing features and train-only preprocessing pipelines |
| `training._fit_candidates` | LR/RF comparison using AP on a designated model-selection partition |
| `modeling.model_metrics`, `ranking_metrics`, `ranking_order` | AP, ROC-AUC, Precision/Recall/Lift@50/100/200 and raw-score/customer-ID tie rules |
| `provenance.experiment_metadata`, fingerprints, JSON writer | Dataset, partitions, software, seed, preprocessing and model metadata |
| Holdout manifest checks | Extracted into shared `verify_partition_manifest`; existing evaluation delegates to it |
| Dashboard helpers | Optional JSON reading, file signatures, chart/table/card presentation and bounded caches |
| `capacity_planning.py` | Continues using original saved rankings; calibration diagnostics never replace its inputs |

`calibration.py` owns orchestration and frozen scikit-learn calibration. `reliability.py` owns pure diagnostics and summary comparison; it cannot train a model. `dashboard.load_reliability` verifies saved diagnostic files and recomputes aggregate metrics. The CLI exposes `calibrate`; existing training/finalization/scoring behavior remains unchanged. No new dependencies are added.

## Methodology and leakage controls

1. Require a schema-v2 reference `metrics.json` and private `split_manifest.json`. Verify disjoint membership, sizes, partition fingerprints and source CSV fingerprint. Legacy models remain usable for scoring/dashboard viewing; experiments without recorded membership cannot safely seed calibration.
2. Use **only reference train + validation customer IDs**, including sampled references. Reserve original test IDs as `excluded_holdout`. Source loading validates CSV schema/labels and checksum; holdout features, predictions and evaluation metrics are never computed. Historical winner names and final metrics do not select a new model or mapping.
3. Split the development pool into five mutually disjoint customer roles with a fixed seed. Stratification uses development labels only; fractions are approximate due to rounding.

| Role | Development fraction | Permitted use |
| --- | --- | --- |
| `train` | 50% | Fit base estimators, median imputation and LR scaling |
| `model_selection` | 10% | Choose LR or RF by existing AP rule |
| `calibration_fit` | 20% | Fit sigmoid/eligible isotonic around the frozen base estimator |
| `calibration_selection` | 10% | Choose uncalibrated/sigmoid/isotonic by minimum Brier score |
| `development_evaluation` | 10% | Compare available methods **after selection is locked** |

4. Mapping selection and diagnostics never reuse calibration-fitting labels. Base-model and calibration selection also have separate roles. No mapping is refitted on selection/diagnostic data. Exact Brier ties prefer uncalibrated, sigmoid, then isotonic. Bins, features and seed are fixed inputs, not optimized against the old test or diagnostic results. No decision threshold is tuned.
5. Development needs at least 100 customers and 20 per class for stratification. Sigmoid needs at least 20 calibration-fitting customers and two per class. Isotonic needs at least 1,000 fitting customers, two per class and two distinct native calibration responses (decision values when available, otherwise probabilities). Ineligible methods are skipped with recorded reasons. These are operability guards, **not sample-size/power guarantees**; 1,000 customers need not provide adequate minority support.

`CalibratedClassifierCV` uses `FrozenEstimator` on sklearn 1.6+ and the supported `cv="prefit"` API on sklearn 1.4/1.5. Neither refits the base pipeline. Sigmoid is sklearn's Platt-style mapping, using `decision_function` when available, otherwise `predict_proba`. Isotonic is a non-decreasing step mapping and can overfit small calibration samples. See the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) and [CalibratedClassifierCV API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).

Historical train/validation customers already participated in earlier research. New fitting/selection separation does **not** make them a new independent final cohort. Repeatedly inspecting development diagnostics can induce researcher adaptation. Freeze the protocol before a future independent evaluation.

## Probability and ranking diagnostics

- **Brier score:** mean squared error between binary labels and positive-label probabilities, on the standard binary `[0, 1]` scale. Lower is better.
- **Log loss:** sklearn binary negative log likelihood with explicit `labels=[0, 1]`. Inputs are float64; log-loss endpoints are clipped at float64 machine epsilon. Exact zero/one probabilities remain unchanged for Brier, binning and ranking.
- **Reliability curve:** sklearn fixed equal-width bins over `[0, 1]`. The first bin includes both endpoints; subsequent bins are `(left, right]`. Plot bin mean prediction against observed positive-label fraction. Save counts and boundaries; empty-bin rates are null, not zero.
- **ECE:** sum of `(bin_count / N) × abs(mean_probability − positive_fraction)` over populated bins. The default is ten bins, configurable before the experiment. ECE depends on bin edges/count and sample size, hides within-bin errors, and is not a proper scoring rule or proof of calibration. Brier/log loss assess discrimination and calibration together; lower loss alone does not prove better calibration. See the discussion of binning instability in [Dimitriadis, Gneiting and Jordan](https://arxiv.org/abs/2008.03033).
- **AP/ROC-AUC and Precision/Recall/Lift@K:** existing raw-probability ranking metrics with K capped at cohort size. Isotonic can introduce ties; sigmoid can reverse ordering on adverse fitting data or create numerical ties/saturation. Recompute rankings rather than assuming invariance. Ties use stripped customer IDs lexicographically ascending. Distinct-probability counts expose compression.

Missing labels or empty cohorts return explicitly unavailable metrics. Single-class diagnostic cohorts retain Brier/log loss/ECE; ROC-AUC is unavailable. AP is unavailable with no positives; existing recall/lift conventions apply. Fewer than 100 diagnostic customers, single-class labels and constant probabilities generate warnings. Workflow splitting rejects insufficient labeled development pools, while pure metrics support these diagnostic edge cases.

There are no confidence bands, independent-cohort uncertainty estimates or claims of significant improvement. Sparse bins, minority counts, label noise, grouping effects and population selection limit interpretation.

## Run a synthetic development experiment

Create a fresh validation-only reference without evaluating its reserved holdout:

```powershell
.\.venv\Scripts\python.exe -c "from gridguard.synthetic import save_synthetic; save_synthetic('data/calibration-demo.csv', customers=7500, days=90, seed=42)"
.\.venv\Scripts\gridguard.exe train --input data/calibration-demo.csv --source synthetic --validation-only --output-dir artifacts/calibration-reference
.\.venv\Scripts\gridguard.exe calibrate --input data/calibration-demo.csv --reference-dir artifacts/calibration-reference --output-dir artifacts/calibration-development --seed 42 --n-bins 10 --feature-config baseline
```

For an existing schema-v2 reference, pass its matching source CSV and run directory to `calibrate`. The workflow ignores the historical estimator and fits fresh base candidates on the new `train` role: the old estimator may already have seen the new diagnostic customers. Changing features/seed/bins is a new development analysis, never an independent final evaluation. Use a new empty output directory for each run.

Artifacts remain local:

| File | Contents |
| --- | --- |
| `reliability.json` | Development/selection diagnostics, mapping choices/skips, bins, role sizes/fingerprints, source/reference fingerprints, software, seed, checksums and evaluation scope |
| `split_manifest.json` | Private membership of all five roles plus excluded holdout IDs |
| `calibration_predictions.csv` | Private diagnostic IDs, labels and before/after probabilities; no holdout predictions |
| `model.joblib` | New selected estimator/mapping and feature metadata, compatible with trusted-local `score`; original model unchanged |

Float32 outputs are expanded to float64 before CSV serialization, preserving the exact values used by diagnostics. Private predictions, manifests and weights are git-ignored. The dashboard checks consumed-file fingerprints, disjoint membership, diagnostic ID alignment and recomputed summary agreement. Checksums establish consistency, not authenticity of untrusted files. Estimator weights are neither loaded nor hashed in this view because it does not consume them.

## Measured synthetic development-only results

On 2026-11-12, the example was exercised with Python 3.13.14, NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1 and joblib 1.6.0. Of 7,500 synthetic customers, 1,500 reserved customers were excluded. Development roles had 3,000 / 600 / 1,200 / 600 / 600 customers in the table's order. LR won base-model selection; all calibration methods were eligible. The 600 diagnostic customers contained 54 synthetic positive labels.

| Method | Selection Brier | Diagnostic Brier | Diagnostic log loss | Diagnostic ECE | AP | ROC-AUC | Distinct probabilities |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Uncalibrated | 0.0000011253 | 0.0000007020 | 0.00044774 | 0.00044739 | 1.000 | 1.000 | 592 |
| Sigmoid | 0.0001921695 | 0.0001768143 | 0.00272622 | 0.00262636 | 1.000 | 1.000 | 600 |
| Isotonic | 0.0000758942 | 0.0000675047 | 0.00060336 | 0.00056624 | 1.000 | 1.000 | 6 |

The fixed selection rule retained **uncalibrated** outputs: optional calibration need not improve losses. All methods had diagnostic Precision@50=1.000, Recall@50=0.9259, Precision@100=0.540 and Recall@100=1.000; isotonic nevertheless compressed predictions to six distinct values. Separate tests cover cases where ties change top-K membership.

These are **simulated development diagnostics on a deliberately easy injected-pattern generator**, not real customer results or operational calibration evidence. The tiny losses and perfect discrimination do not generalize to realistic inspections. No real-data estimator or historical artifact was retrained/rewritten for this example; independent final calibration performance remains unestablished. Exact values may differ across supported software versions, which are recorded with seeds and partitions.

## Read-only dashboard

```powershell
$env:GRIDGUARD_RELIABILITY_REPORT = 'artifacts/calibration-development/reliability.json'
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Choose **Model Reliability**, or enter its report path directly. This view works independently of the legacy selector, including without a legacy `metrics.json`. It shows the development-only warning, reliability diagram, bin counts, Brier/log-loss/ECE/AP/ROC-AUC, ranking comparison and fitting/selection metadata. Missing calibration artifacts do not break other workspaces. Its eight-entry/five-minute cache includes report, manifest and diagnostic CSV signatures. No estimator loading or fitting occurs while rendering.

Even after calibration, outputs concern **historical dataset labels**, not confirmed theft probabilities. Dataset shift, biased/noisy labels and an unrepresentative calibration population can invalidate deployment reliability. Independent human review remains necessary; calibration cannot justify allegations, disconnection or billing actions.

## Verification

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover separation/overlap, holdout feature exclusion, selection despite altered historical final metrics, independent diagnostic outcomes, train-only imputation/scaling, frozen base estimators, reproducibility, eligibility, exact scores/ECE, endpoints, missing/single-class labels, constant distributions, tie-induced ranking changes, bundle compatibility, artifact corruption and read-only AppTest rendering. CI runs the suite and existing synthetic smoke workflows with current sklearn and minimum-supported sklearn 1.4.2, plus a development-only calibration CLI smoke test.

Local verification on 2026-11-12: Ruff and 167 pytest tests passed. The existing synthetic CLI demo/experiment/finalize/evaluation smoke passed in a temporary directory. A separate seven-section AppTest exercised the all-method synthetic reliability report with model/calibrator loading/fitting blocked during rendering. SHA-256 checks confirmed all 65 existing local artifact/report files remained unchanged.

The minimum-sklearn CI environment constrains NumPy to 1.x, pandas to 2.x and PyArrow below 26 for binary/runtime compatibility; the current-version job uses the normal dependency resolution. These CI-only constraints do not change application dependencies or existing artifact environments. Jobs verify dependency consistency/imports and run independently so one failure does not cancel the other.
