# Model explainability and error analysis

This optional workflow studies global AP sensitivity and observed classifier errors using **separate development cohorts**. It fits a fresh, fixed Random Forest (RF); it does not explain the historical final model or establish independent final performance. The inspected final holdout is excluded. Published models, scores, metrics, calibration and anomaly artifacts remain unchanged.

## Audit and architecture rationale

The starting main includes merged PR #8. Existing modules provide most of the necessary machinery:

| Component | Reuse |
| --- | --- |
| `features.build_features` | Existing customer-level feature definitions and original column names |
| Provenance/reference helpers | Schema-v2 source checksum, exact membership, partition fingerprints and disjointness |
| `training.development_partitions` | Extracted from the anomaly workflow: original train retained, original validation split in half |
| `training.fit_development_baseline` | Shared fixed RF recipe and train-fitted imputer, with one worker for deterministic sums |
| `modeling.classification_metrics` | Shared declared-threshold confusion counts/rates; legacy defaults/zero-division outputs retained |
| `score_metrics`, `ranking_metrics` | Existing AP/ROC-AUC and deterministic top-K definitions |
| `error_analysis.aggregate_errors` | Existing primary-threshold subgroup errors, quality bins and global queue errors; optional capacities keep defaults compatible |
| `anomaly_metrics.score_distribution` | Existing overflow-safe statistics; optional common histogram ranges enable outcome comparisons |
| Dashboard helpers | Shared saved-development-file verification, bounded caching, cards, charts and privacy conventions |

`explainability.py` orchestrates offline fitting/permutation/artifact creation. `explainability_metrics.py` contains pure repeat-summary and saved-score error diagnostics. The dashboard checks arithmetic and fingerprints but cannot load or rescore a model. No dependency, SHAP implementation or partial-dependence workflow is added; the requested global sensitivity and descriptive distributions are available with existing dependencies.

## Development protocol and leakage protection

The CLI accepts a standard schema-v2 reference containing `metrics.json` and `split_manifest.json`, plus its matching source CSV. Dataset checksum, recorded customer IDs, sizes, partition fingerprints and disjointness must match before fitting. Sampled references keep only their recorded customers.

| Role | Membership | Permitted use |
| --- | --- | --- |
| `train` | Original reference training customers, unchanged | Fit fixed RF and preprocessing |
| `development_importance` | Seeded stratified half of reference validation | Frozen-pipeline permutation importance |
| `development_evaluation` | Other half of reference validation | Error counts, rates, groups and distributions |
| `excluded_holdout` | Original final holdout | Membership record only; no feature extraction, predictions or analysis |

All readings for a customer remain together. Training requires both classes; reference validation needs at least four customers and two per class. Insufficient provenance/data fail explicitly. Pure diagnostics also handle empty, unlabeled and single-class cohorts.

The historical model may have been selected using all reference-validation labels. Reusing it would not provide independent explainability on those labels. Instead fit the existing RF recipe (120 trees, depth 12, minimum leaf size 3, balanced subsample weights) on original train, with `n_jobs=1`. Median imputation is fitted once inside that pipeline. Its state is reused unchanged for ordinary predictions and every permutation; the importance/error cohorts never fit preprocessing.

Features, model parameters, seed, repeat count, thresholds and inspection capacities are predeclared. No importance or error result selects features, thresholds, a model family, calibration mapping or hyperparameters. Selecting a displayed feature or saved threshold in the dashboard only changes the view. Repeated inspection of development results can still induce researcher adaptation; freeze a separate protocol before a future independent study.

The source loader validates the CSV, including its label schema, but downstream feature extraction receives only recorded train+validation customers. Previously researched development customers are not a new untouched cohort. These results must not be substituted for historical final results or prospective inspection performance.

## Permutation methodology

Use `sklearn.inspection.permutation_importance` on the **frozen complete pipeline**, raw engineered feature columns and the importance cohort, with `scoring="average_precision"`, full-cohort `max_samples=1.0`, one worker, configurable repeats (default ten) and fixed seed (default 42). Each column is independently shuffled; preprocessing is applied using its existing train-fitted state.

For feature j and repeat r:

```text
d[j,r] = AP(original importance cohort) − AP(cohort with feature j permuted)
mean[j] = mean_r d[j,r]
SD[j] = population standard deviation_r d[j,r]  (ddof=0)
```

Save every repeat decrease, mean and SD, retaining negative means and exact original feature names/order. A negative value means shuffling improved AP in that sample, which can reflect noise, overfitting or dependence; it is not a negative causal effect and does not trigger automatic feature removal. Values near machine precision are numerical noise, not a defensible feature ordering. With one repeat SD is zero by construction, not evidence of certainty.

**Repeat SD is not a confidence interval.** It describes permutation randomness conditional on one fitted model and cohort, excluding sampling, training and label uncertainty. AP measures ranking quality, so changes in score magnitude without changed ordering can yield zero importance. Correlated/redundant features can share or mask importance. Independent shuffling can create feature combinations that never occur naturally, especially for statistics derived from the same readings. Low importance does not prove a feature is irrelevant; high importance is neither causal evidence nor individual attribution. See the [official permutation-importance guide](https://scikit-learn.org/stable/modules/permutation_importance.html) and [API](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html).

Empty, missing-label or single-class importance cohorts return an explicit unavailable status, an empty repeat array and null mean/SD summaries; they are not passed to the estimator or permutation routine. Constant features and missing values are supported through the fitted pipeline. Name/order mismatches and overlap with train, error-evaluation or excluded IDs are rejected.

## Error definitions, subgroup support and plots

Thresholds default to **0.2, 0.5 and 0.8**, with primary group threshold **0.5**. Configured lists must include 0.5; no optimization is performed. A score equal to the threshold is predicted positive. TP/FP/TN/FN always refer to the **observed dataset label**, not verified theft.

| Quantity | Definition |
| --- | --- |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN) |
| False-positive rate | FP/(FP+TN) |
| False-negative rate | FN/(FN+TP) |
| F1 | 2TP/(2TP+FP+FN) |
| Confusion matrix | Rows observed negative/positive; columns predicted negative/positive |
| Precision@K / Recall@K | Existing observed-positive counts among raw-score/ID-ranked customers; K capped at cohort size |

New threshold summaries use null for undefined denominators. Existing `model_metrics` retains historical `zero_division=0` precision/recall/F1 conventions. Empty cohorts have zero confusion counts and unavailable rates; missing labels produce no invented outcomes. Single-class ROC-AUC is unavailable and existing no-positive AP/recall/lift conventions apply.

The primary-threshold group analysis reuses fixed missing-reading-rate, mean-consumption and coefficient-of-variation bins. Group sizes and positive support accompany errors. Groups below 20 customers by default keep only support/suppression information; rates are marked insufficient when there are fewer than five positives or negatives. The dashboard withholds those unreliable rates while retaining supported descriptive counts. These are safeguards, not power calculations or formal privacy guarantees. Stored legacy 0.2/0.8 high-score cutoff counts remain descriptive uncalibrated-score heuristics, not statistical confidence.

Quality-group charts always refer to **0.5**, even when another saved threshold is selected for global counts/distributions. This scope is stated in the UI. Missingness can reflect outages, meter faults or collection issues; group/error relationships do not establish causal data-quality effects or fairness.

For each declared threshold, store outcome-wise distributions of all original features, raw model scores and illustrative `round(100 × score)` risk scores. Outcome groups use common per-feature bins and fixed score ranges `[0,1]`/`[0,100]`; plot fractions of observed values within each outcome. These conditional fractions are not outcome prevalence. Raw missing features stay missing in diagnostics; they are not replaced by imputed zeros in charts. Counts of missing/observed values remain explicit. Value distributions require at least 20 observed values by default; small, empty or all-missing outcomes have no exposed value distributions. Empirical histogram ranges are display summaries, never decision thresholds.

## Reproduction

Generate a clean synthetic source and validation-only reference:

```powershell
.\.venv\Scripts\python.exe -c "from gridguard.synthetic import save_synthetic; save_synthetic('data/explain-clean.csv', customers=7500, days=90, seed=42)"
.\.venv\Scripts\gridguard.exe train --input data/explain-clean.csv --source synthetic --validation-only --output-dir artifacts/explain-reference-clean
.\.venv\Scripts\gridguard.exe explain --input data/explain-clean.csv --reference-dir artifacts/explain-reference-clean --output-dir artifacts/explain-clean --seed 42 --n-repeats 10 --thresholds 0.2 0.5 0.8 --capacities 50 100 200
```

The fixed label-noise stress example flips exactly 375 of 7,500 observed labels, without changing readings:

```powershell
.\.venv\Scripts\python.exe -c "from gridguard.synthetic import make_synthetic; import numpy as np; f=make_synthetic(customers=7500, days=90, seed=42); i=np.random.default_rng(123).choice(len(f), size=375, replace=False); f.loc[i,'FLAG']=1-f.loc[i,'FLAG']; f.to_csv('data/explain-noise.csv', index=False)"
.\.venv\Scripts\gridguard.exe train --input data/explain-noise.csv --source synthetic --validation-only --output-dir artifacts/explain-reference-noise
.\.venv\Scripts\gridguard.exe explain --input data/explain-noise.csv --reference-dir artifacts/explain-reference-noise --output-dir artifacts/explain-noise --seed 42 --n-repeats 10
```

Use fresh empty output directories. For an existing standard reference, supply its matching source CSV; no old estimator is loaded. Legacy runs without recorded schema-v2 membership cannot safely seed this analysis, while legacy scoring/dashboard behavior remains supported.

## Versioned, local-only artifacts

| File | Contents |
| --- | --- |
| `explainability.json` | Schema-v2/explainability-v1 config, mean/SD importance, aggregate errors, role/dataset fingerprints, model/feature metadata, seeds/software and explicit development scope |
| `permutation_repeats.json` | Original feature order, repeat AP decreases, baseline AP and cohort/configuration counts |
| `explainability_predictions.csv` | Private importance/error customer IDs, observed labels, raw features and unpermuted scores; no train/holdout predictions |
| `split_manifest.json` | Private exact membership of all four roles |
| `model.joblib` | New fixed train-fitted RF pipeline; historical weights unchanged |

All outputs are git-ignored, including custom-directory filenames. Full reports contain local source paths; publish only curated aggregates. Expand float32 values to float64 before CSV serialization to retain exact saved values/ties. The read-only loader checks consumed-file fingerprints, disjoint role alignment, feature names/order, repeat means/SD/baseline arithmetic and recomputed saved-score diagnostics. It does **not** rerun permutations or authenticate untrusted files. Model hashes are recorded offline; weights are never opened by the workspace.

## Measured synthetic examples

Measured on 2026-11-12 using Python 3.13.14, NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1 and joblib 1.6.0. Both predeclared scenarios used 7,500 synthetic customers, 90 reading days, model/split/permutation seed 42, baseline features and ten repeats. Each had 4,500 train, 750 importance, 750 error-evaluation and 1,500 excluded-holdout customers.

| Scenario | Importance positives / AP | Error positives / AP | TN / FP / FN / TP at 0.5 |
| --- | --- | --- | --- |
| Clean generator | 68 / 1.0000 | 67 / 1.0000 | 683 / 0 / 0 / 67 |
| Fixed 5% observed-label flips | 98 / 0.7102 | 98 / 0.7398 | 650 / 2 / 33 / 65 |

Clean single-column AP importances were effectively zero (largest mean approximately `1.8e−16`), despite perfect discrimination. This is consistent with metric saturation and redundant correlated signals; it does **not** prove no features drive the RF. No noise-scale ordering is promoted as a scientific result.

Selected stress-case importance results, measured **only on the importance cohort**:

| Original feature | Mean AP decrease | Repeat SD |
| --- | --- | --- |
| `recent_drop_ratio` | 0.01706 | 0.00845 |
| `recent_previous_ratio` | 0.01650 | 0.00843 |
| `zero_ratio` | 0.01458 | 0.00250 |
| `std_kwh` | 0.00744 | 0.00675 |
| `median_kwh` | −0.01356 | 0.01242 |
| `weekend_weekday_ratio` | −0.00860 | 0.00451 |
| `missing_ratio` | −0.00421 | 0.00323 |

Six stress-case feature means were negative and were retained. Their signs do not justify causal conclusions or automatic removal. Threshold tradeoffs on the separate stress error cohort were:

| Predeclared threshold | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- |
| 0.2 | 192 | 20 | 0.2889 | 0.7959 |
| 0.5 | 2 | 33 | 0.9701 | 0.6633 |
| 0.8 | 2 | 33 | 0.9701 | 0.6633 |

No threshold was selected from these results. The two-person FP value distributions were suppressed; the 33-person FN cohort supported descriptive plots. Because noisy labels also change stratification and fitted models, the scenarios are **not a paired causal estimate** of label-quality impact. Errors refer to noisy observed flags, not physical theft. The generator is deliberately easy, noise is simulated, and neither example establishes real customer reliability. No real-data estimator or historical artifact was retrained/rewritten. Exact values can vary across recorded software versions.

## Read-only dashboard and interpretation

```powershell
$env:GRIDGUARD_EXPLAINABILITY_REPORT = 'artifacts/explain-noise/explainability.json'
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Choose **Model Explainability** or enter the report path. The workspace is independent of the legacy run selector. It shows importance/error bars, exact confusion counts, predefined threshold/capacity tables, primary-threshold group errors, outcome distributions and provenance. It caches eight reports for five minutes, invalidated by all consumed files' signatures. No estimator loading, fitting, permutation computation or model rescoring occurs. It exposes no customer IDs or customer-level downloads; missing optional artifacts leave legacy sections usable.

Historical inspection selection, noisy labels, uncertain event dates, customer-level rather than temporal splits, dataset shift and weak subgroup support limit operational interpretation. Importance is global model sensitivity, not a customer explanation or evidence of electricity theft. Independent human verification remains required. Before operational use, predeclare further development choices, freeze them and evaluate on an independent, representative cohort with dated outcomes and uncertainty analysis. Do not tune using the previously inspected holdout.

## Verification

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover train/cohort separation, holdout exclusion, train-only preprocessing, deterministic repeats/negative importances, feature alignment, constant/correlated/missing inputs, unavailable cohorts, exact confusion/threshold boundaries, subgroup/observation suppression, saved-artifact integrity, read-only/rescore-free rendering and legacy compatibility. CI retains current/minimum-supported sklearn jobs and all existing CLI smoke workflows, adding the optional `explain` smoke check.

Local verification passed Ruff, all 243 pytest tests, all nine dashboard workspaces and the synthetic demo, experiment, finalize, saved-holdout evaluation, calibration, anomaly and explainability CLI smoke workflows. All 65 pre-existing files under `artifacts/` and `reports/` retained their SHA-256 hashes; no files were added there. Synthetic measurements and smoke outputs used temporary directories and were removed after verification.
