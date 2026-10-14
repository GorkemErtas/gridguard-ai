# Model card — GridGuard AI v0.1

## Intended use

Demonstrate statistical analysis, supervised anomaly/fraud-risk modeling, and inspection prioritization on **customer-level** electricity consumption histories.

## Not intended for

- Proving an individual stole electricity.
- Automated punishment, disconnection, billing adjustments or fraud allegations.
- Live operational deployment or country-to-country transfer without independent validation.

## Data

**Demo:** generated from known statistical patterns. Positives exhibit artificially injected consumption drops, so the task is deliberately easy. Metrics must carry the **synthetic** label.

**Optional research dataset:** SGCC Electricity Theft Detection, 42,372 customers over roughly 1,035 days, from the [IEEE PES catalog](https://ieee-pes-data-sharing.org/datasets/detail/663b8e0e-3ace-48f8-b983-4cb59b39fa6f). Its public catalog lists **License: N/A**; research accessibility does not imply permission to redistribute. Download separately and verify usage terms.

Both formats use wide daily readings per customer and binary historical \`FLAG\`. No theft dates, geospatial data, detailed inspections, or independent Turkish deployment labels are included.

## Training and model selection

- Customer-stratified train / validation / test split: 60 / 20 / 20.
- Median imputation fitted only on train; standardization for logistic regression.
- Baseline: dummy prevalence prior.
- Candidates: class-weighted logistic regression and class-weighted random forest.
- Select highest **validation average precision**. No test-set hyperparameter tuning.
- Report test ROC-AUC, average precision, F1/precision/recall at 0.5, and Precision@50.

For highly imbalanced data, performance should be evaluated relative to class prevalence and field inspection capacity. A fixed 0.5 threshold is used for illustration, not tuned operations.

## Model output

\`risk_score\` is rounded 100× the model's predicted class-1 probability. The number is **not probability-calibrated**, even though it is derived from \`predict_proba\`. High/medium/low display bands are arbitrary. \`review_signal\` comprises transparent *rules*, **not feature attribution** or causal explanations.

## Risks and limitations

- Selection bias and noisy labels from historical inspections.
- Missing readings caused by meter failures, outages, and data ingestion.
- Geographic transfer from the SGCC dataset to an İzmir distribution network is untested.
- A retrospective customer split does not measure time-to-detection.
- False positives impose inspection costs and can unfairly affect customers.
- A model may exploit quirks of a simulated anomaly rather than realistic theft patterns.
- Holdout metrics do not imply calibrated probabilities or operational precision.

## Required future work

Add dated outcome labels, temporal validation, confidence intervals, score calibration, business-cost evaluation, drift monitoring, security and privacy review, and human-in-the-loop auditing before any operational use.
