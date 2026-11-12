# Model card — GridGuard AI v0.1

## Intended use

Demonstrate statistical analysis, supervised anomaly/fraud-risk modeling, and inspection prioritization on **customer-level** electricity consumption histories.

## Not intended for

- Proving an individual stole electricity.
- Automated punishment, disconnection, billing adjustments or fraud allegations.
- Live operational deployment or country-to-country transfer without independent validation.

## Data

**Demo:** generated from known statistical patterns. Positives exhibit artificially injected consumption drops, so the task is deliberately easy. Metrics must carry the **synthetic** label.

**Optional real-world dataset:** Labeled customer-by-day electricity consumption readings obtained separately. Record the actual dataset source, citation, and permission terms before publishing any empirical results. Public accessibility does not imply redistribution rights.

Both formats use wide daily readings per customer and binary historical \`FLAG\`. No theft dates, geospatial data, detailed inspections, or independent Turkish deployment labels are included.

## Training and model selection

- Customer-stratified train / validation / test split: 60 / 20 / 20.
- Median imputation fitted only on train; standardization for logistic regression.
- Baseline: dummy prevalence prior.
- Candidates: class-weighted logistic regression and class-weighted random forest.
- Select highest **validation average precision**. No test-set hyperparameter tuning.
- Report test ROC-AUC, average precision, F1/precision/recall at 0.5, and Precision/Recall/Lift@50/100/200 using raw probabilities and customer-ID tie breaks.

Four fixed feature configurations are available. `baseline` preserves the pilot and legacy bundles; `calendar` inserts missing dates as NaN; `quality` adds longest missing/zero streaks and IQR; `combined` applies both. No extreme-reading clipping or missing-to-zero replacement is applied. `experiment` selects on validation without scoring test. `finalize` reuses the selected frozen train-only estimator and verifies data, split, model and software provenance. No train+validation refit is performed.

The [research note](pipeline-evaluation.md) records data attribution, the original pilot, controlled validation ablations and the final holdout evaluation. The observed AP gain on one fixed split is not evidence of generalization or statistical significance. New holdout reports verify membership from local manifests; legacy reports explicitly lack that guarantee. Only trusted local joblib bundles should be loaded.

For highly imbalanced data, performance should be evaluated relative to class prevalence and field inspection capacity. A fixed 0.5 threshold is used for illustration, not tuned operations.

## Model output

\`risk_score\` is rounded 100× the model's predicted class-1 probability. The number is **not probability-calibrated**, even though it is derived from \`predict_proba\`. High/medium/low display bands are arbitrary. \`review_signal\` comprises transparent *rules*, **not feature attribution** or causal explanations.

## Optional calibration research

The optional `calibrate` command creates a separate development-only experiment; published models, scores and holdout results stay unchanged. The reference holdout is excluded. Five disjoint development roles separate base fitting, base selection, calibration fitting, mapping selection and diagnostics. Base selection uses AP; mapping selection uses Brier on its own partition. Sigmoid and eligible isotonic mappings wrap a frozen base estimator and are never fitted in the dashboard. Uncalibrated outputs can remain the winner.

Saved reports compare Brier score, log loss, fixed-bin ECE/reliability curves, AP, ROC-AUC and top-K metrics. Isotonic requires at least 1,000 calibration-fitting customers and can create ranking ties; ECE depends on binning and does not prove calibration. The measured example uses synthetic development data only. **Independent final calibration performance is not established**: the historical holdout was inspected and no new independent cohort is available. See the [methodology, measured synthetic results and limitations](probability-calibration.md).

Calibrated values estimate historical positive-label frequency under source-population assumptions. They are not confirmed electricity-theft probabilities. Label noise, inspection selection bias, population representativeness and dataset shift can invalidate calibration. No customer should be accused, penalized or disconnected based solely on raw or calibrated outputs.

## Optional anomaly benchmark

The optional `benchmark-anomaly` workflow compares Isolation Forest and a robust median/MAD baseline with a fresh, fixed RF using the same original training customers and shared train-only imputation. The historical holdout is excluded. Two disjoint development cohorts provide preliminary diagnostics and evaluation reporting; no method, feature, threshold or ensemble is selected from evaluation outcomes. Historical supervised and calibration results remain unchanged.

Native anomaly scores are not probabilities or confirmed theft evidence. Unusual readings may indicate outages, data quality problems, customer differences or ordinary consumption variability. Disagreement does not establish that either model is wrong. The [benchmark guide](anomaly-benchmark.md) records methodology, mathematical definitions, synthetic measurements and limitations, including why additional positives outside a top-K queue do not establish improved yield at equal capacity. Independent final/operational performance is not established.

## Optional model explainability and error analysis

The optional `explain` command fits a fresh fixed RF on the original reference training customers, retaining train-only preprocessing. It excludes the historical holdout and splits reference validation into disjoint importance and error-evaluation cohorts. It does not explain the historical validation-selected final estimator. No feature, method, threshold or capacity is selected from diagnostic outcomes; independent final performance is not established.

Global permutation importance measures the decrease in development Average Precision after shuffling each original engineered feature through the frozen pipeline. Reports retain negative means and the population standard deviation across fixed-seed repeats. Repeat variability is not a confidence interval or uncertainty over training samples. Correlated/derived features can mask or share importance, and AP saturation can produce near-zero numerical values despite perfect ranking. Small, single-class or unavailable cohorts are explicitly marked.

Error analysis compares observed labels with predictions at predefined thresholds and capacities, with confusion counts, missing-reading/consumption groups and feature/risk-score distributions. Sparse groups are suppressed or marked unreliable. An FP/FN is disagreement with a historical or synthetic flag, not proof of innocence or theft. Features and missingness may be associated with errors without causing them. Saved artifacts are local and versioned; the read-only dashboard exposes only aggregates and never loads, trains or rescores a model. See [methodology, clean and noisy synthetic measurements, reproduction and operational limits](model-explainability.md).

## Risks and limitations

- Selection bias and noisy labels from historical inspections.
- Missing readings caused by meter failures, outages, and data ingestion.
- Geographic transfer from the training distribution network to another region is untested.
- A retrospective customer split does not measure time-to-detection.
- False positives impose inspection costs and can unfairly affect customers.
- A model may exploit quirks of a simulated anomaly rather than realistic theft patterns.
- Holdout metrics do not imply calibrated probabilities or operational precision.

## Required future work

Add dated outcome labels, temporal validation, independent-cohort uncertainty and calibration validation, business-cost validation, drift monitoring, security and privacy review, and human-in-the-loop auditing before any operational use. The available customer-bootstrap AP interval is descriptive and assumes independent customers; optional development calibration does not complete that independent validation.
