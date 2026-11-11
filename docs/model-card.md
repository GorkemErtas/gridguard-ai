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

## Risks and limitations

- Selection bias and noisy labels from historical inspections.
- Missing readings caused by meter failures, outages, and data ingestion.
- Geographic transfer from the training distribution network to another region is untested.
- A retrospective customer split does not measure time-to-detection.
- False positives impose inspection costs and can unfairly affect customers.
- A model may exploit quirks of a simulated anomaly rather than realistic theft patterns.
- Holdout metrics do not imply calibrated probabilities or operational precision.

## Required future work

Add dated outcome labels, temporal validation, independent-cohort uncertainty analysis, score calibration, business-cost evaluation, drift monitoring, security and privacy review, and human-in-the-loop auditing before any operational use. The available customer-bootstrap AP interval is descriptive and assumes independent customers.
