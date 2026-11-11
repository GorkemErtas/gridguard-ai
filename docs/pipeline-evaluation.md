# Pipeline audit and risk-ranking evaluation

Audit and local experiments completed on 2026-11-11. The existing pandas/scikit-learn architecture, Dummy/LR/RF candidates, sampling algorithm and 60/20/20 customer split are retained.

## Confirmed findings and changes

| Finding | Change and evidence |
| --- | --- |
| Training top-K metrics broke ties by split order; exported queues and holdout reports broke them by customer ID. A constant-prior model could report different Precision@K values. | One raw-probability ordering with ID ties now drives metrics, exports and the dashboard. Regression tests cover ties, shuffled input, rounding collisions and invalid K/scores. Dummy AP is unchanged; its tied top-K precision intentionally changes. |
| Last-30/previous-30 features counted reading columns. Daily differences could span an absent calendar day. | An explicit `calendar` configuration inserts missing dates as NaN before window/difference calculations. The original calculation remains the default `baseline` for pilot and old-bundle compatibility. |
| Holdout validation checked row count, allowing another same-size customer cohort to pass. | New artifacts bind prediction bytes to the report and validate a disjoint local split manifest, partition fingerprints and exact test membership. Legacy artifacts remain readable with `legacy_unverified` provenance. These controls catch artifact mixups; they are not signed attestation. |
| Every training invocation evaluated test, making feature experiments prone to repeated test inspection. | `experiment` runs four validation-only ablations. It never extracts test features or produces test predictions. `finalize` loads the frozen validation winner and verifies dataset, split, model checksum and software versions before final scoring. |
| Experiments omitted feature configuration, dataset fingerprint, classifier parameters and library versions. Existing output paths could be overwritten. | Schema-v2 reports/bundles add this provenance. Training and report generation require empty output directories. An allowlisted `validation_summary.json` excludes input paths and customer membership. |
| Error exports contained customer rows by default and lacked quality-group analysis. | Aggregate reports separate 0.5-threshold errors, fixed high-confidence cutoffs and global top-K errors. Customer-level CSV export is opt-in. Fixed quality bins suppress rates for groups smaller than 20. This is not a formal privacy guarantee. |
| Dashboard history omitted year-first slash dates and could display invalid negatives or bridge missing dates. Preflight counted infinities as nonnumeric cells. | History uses the loader's date parser and missing-value policy, inserts calendar gaps and scans bounded chunks. Preflight distinguishes nonnumeric/nonfinite readings and reports effective training missingness. |

No cross-customer feature leakage or train-fitted preprocessing leakage was found. The loader already rejected duplicate IDs and isolated labels; split-level guards and regression tests now protect those assumptions directly. Imputer medians and scaler means are checked against the training partition. Existing models still score with the original 15 features when their bundle lacks a configuration field.

## Controlled feature experiment

Only three interpretable features were added: longest consecutive missing-reading days, longest consecutive zero-reading days, and consumption interquartile range. Streaks use the full calendar; missing dates break zero streaks. IQR ignores missing readings. Missing values are never filled with zero, and extreme observed readings remain in the data. Undefined feature values are imputed inside each training pipeline.

The four configurations were fixed before running validation. No hyperparameter search, new model family, threshold tuning, dependency or train+validation refit was introduced. The small feature comparison already addresses the audit findings; a broader search is not justified by this single split.

The local CSV contains 42,372 customers, 1,034 reading columns across 2014-01-01 through 2016-10-31, one missing calendar day, 8.5316% positive labels and 25.6399% missing/nonfinite readings. The largest positive reading is 800,003.32 kWh; this audit does not establish that it is an error.

Experiment `pipeline-validation-20261111` used the original stratified 10,000-customer sample, seed 42 and 6,000/2,000/2,000 split. Its test customer membership was checked against the old pilot using IDs only. The full aggregate report, parameters, features, fingerprints and versions are in [pipeline-validation.json](experiments/pipeline-validation.json).

| Configuration | LR validation AP | RF validation AP | RF ROC-AUC | Precision@50 | Precision@100 | Precision@200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline`: original features/preprocessing | 0.25420 | 0.27249 | 0.76004 | 0.40 | 0.37 | 0.315 |
| `calendar`: preprocessing only | 0.25461 | 0.28384 | 0.76127 | 0.48 | 0.32 | 0.300 |
| `quality`: three additional features only | 0.25240 | **0.31025** | **0.78254** | 0.46 | 0.41 | 0.325 |
| `combined`: calendar + additional features | 0.25208 | 0.30140 | 0.77572 | 0.42 | 0.42 | 0.320 |

Dummy validation AP is 0.08500 throughout. Original LR and RF validation AP were reproduced exactly in this environment. The `quality` RF won on validation AP, not on top-50 precision or any test metric. Its existing recent-window features retain pilot semantics so this is a features-only ablation; the new streaks still use calendar days. Baseline remains the default, and a serialized bundle explicitly selects the scoring configuration.

## Validation error analysis

The selected model's [aggregate validation errors](experiments/validation-error-analysis.json) show 55 false positives and 130 false negatives at 0.5, with 39 positives scoring at most 0.2 and no negatives scoring at least 0.8. These are raw score cutoffs, not calibrated confidence estimates. At top 50 there are instead 27 negatives in the queue and 147 positives outside it; threshold and capacity errors describe different decisions.

| Missing-reading fraction | Customers | Positive prevalence | AP | False-positive rate at 0.5 | False-negative rate at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0, 0.10) | 1,001 | 0.0669 | 0.2763 | 0.0182 | 0.8358 |
| [0.10, 0.25) | 294 | 0.0986 | 0.3745 | 0.0226 | 0.6897 |
| [0.25, 0.50) | 215 | 0.0930 | 0.4085 | 0.0667 | 0.6000 |
| [0.50, 1.00] | 490 | 0.1102 | 0.3189 | 0.0436 | 0.7778 |

The JSON also reports variability and consumption-magnitude groups. Rates use the group's actual negatives/positives as denominators. AP depends on prevalence, so these differences do not establish causality, fairness or a meter-quality mechanism. Small groups and single-class groups cannot support all metrics; undefined metrics are JSON null.

## Final holdout evaluation

After code checks and the synthetic end-to-end smoke test passed, configuration `quality` and its train-only RF were frozen. Experiment `pipeline-final-20261111` scored the existing 2,000-customer test partition once. No refitting or selection changes followed the results. The original `artifacts/real-pilot` and `reports/real-pilot` were preserved.

| Metric | Original pilot | Final frozen model |
| --- | ---: | ---: |
| Average Precision | 0.29631 | 0.32281 |
| ROC-AUC | 0.78204 | 0.78831 |
| Precision@50 | 0.40 | 0.50 |
| Precision@100 | 0.37 | 0.43 |
| Precision@200 | 0.325 | 0.345 |
| Precision at 0.5 | 0.36559 | 0.45055 |
| Recall at 0.5 | 0.19883 | 0.23977 |
| F1 at 0.5 | 0.25758 | 0.31298 |

Test prevalence is 0.0855. Recall@50/100/200 is 0.1462/0.2515/0.4035; lift is 5.8480/5.0292/4.0351. The final confusion matrix is `[[1779, 50], [130, 41]]` at 0.5. A 1,000-resample customer bootstrap (seed 42) gives a descriptive AP 95% interval of [0.26174, 0.39047]. It is not a paired significance test of improvement. Saved-prediction analysis does not refit or rescore the estimator. See [final summary](experiments/pipeline-final.json) and [aggregate holdout errors](experiments/holdout-error-analysis.json).

## Verification and remaining limits

Ruff and 40 synthetic unit/integration tests passed locally. The existing 120-customer/90-day synthetic demo and the new `experiment → finalize → evaluate` CLI smoke workflow passed. CI uses generated fixtures only. The tests cover schema/date handling, deterministic sampling/features/ranking, customer separation, train-only preprocessing, artifact substitution, metadata, legacy compatibility and the validation-only dashboard.

Measured gains apply to one retrospective customer split. They do not establish temporal, geographic or deployment performance, probability calibration or a causal explanation for theft labels. Additional feature blocks, clipping rules, trend/seasonal features, model searches and refitting remain hypotheses. Any further selection should use development data and a fresh independent final cohort rather than repeatedly consulting this now-reported test set. The bootstrap assumes independent customers. Observation onset, label reliability and data ownership need further research.

## Research data attribution

The local real-world file used for the pilot and these experiments is the State Grid Corporation of China consumption dataset distributed through the [publisher's dataset repository](https://github.com/henryRDlab/ElectricityTheftDetection). Its README requests citation of Zibin Zheng, Yatao Yang, Xiangdong Niu, Hong-Ning Dai and Yuren Zhou (2018), [Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids](https://doi.org/10.1109/TII.2017.2785963), IEEE Transactions on Industrial Informatics 14(4), 1606–1615. The repository describes 1,035 calendar days; the actual local CSV has 1,034 date columns, which preflight reports explicitly.

Generic module and command names do not alter this provenance. Public access and a citation request do not establish redistribution rights. Only aggregate research summaries are committed; customer IDs, raw readings, local manifests, prediction exports and trained weights remain local.
