# Leakage-safe anomaly detection benchmark

This opt-in benchmark investigates whether unsupervised rankings complement a supervised electricity-consumption classifier. It compares a fresh Random Forest (RF), Isolation Forest (IF) and a robust median/MAD baseline. There is no ensemble, automatic winner selection or operational threshold.

**Results are development-only.** The inspected final holdout is excluded, and historical models, metrics, calibration reports, capacity-planning inputs and risk scores remain unchanged. Independent final performance is not established.

## Repository audit and architecture

The starting main includes merged calibration PR #7 and capacity-planning PR #6. Reuse avoids a second training stack:

| Existing component | Reuse |
| --- | --- |
| `data.py`, `ConsumptionData.subset` | CSV validation, normalized IDs and aligned customer subsets |
| `features.build_features` | Existing customer-level features; fixed baseline configuration by default |
| `modeling.candidate_models` | Existing RF learning recipe, with one worker for deterministic score summation |
| RF pipeline median imputer | Fitted once on train and shared unchanged by both anomaly detectors |
| `ranking_order`, `ranking_metrics`, `score_metrics` | Common AP/ROC-AUC/Precision/Recall/Lift and deterministic ranking |
| Provenance/calibration verification | Dataset/software metadata and fingerprints; reference filtering extracted into shared `load_reference_development` |
| `error_analysis.QUALITY_BINS` | Existing missing-reading-rate bins |
| Dashboard helpers | Shared saved-development-artifact validation, bounded caching, cards, charts and tables |
| `capacity_planning.py` | Unchanged; anomaly scores do not replace the supervised inspection queue |

`anomaly.py` owns fixed fitting and the robust estimator; `anomaly_metrics.py` owns pure distributions, paired rankings and subgroup diagnostics. Probability APIs retain their existing `[0,1]` validation. Explicit `score_domain="unbounded"` supports native anomaly scores without converting them into apparent probabilities. No dependency is added.

## Evaluation protocol and leakage controls

Require a standard schema-v2 reference `metrics.json`, its private `split_manifest.json`, and the matching source CSV. Verify dataset checksum, recorded IDs, sizes, partition fingerprints and disjointness before fitting. Sampled references keep exactly their recorded membership.

| Role | Membership | Permitted use |
| --- | --- | --- |
| `train` | Original reference train, unchanged | Fit RF, imputation, IF and MAD statistics |
| `development_validation` | Seeded stratified half of original validation | Preliminary diagnostics; no automatic selection |
| `development_evaluation` | Other half of original validation | Paired reporting after configurations are fixed |
| `excluded_holdout` | Original final holdout | Record IDs only; no feature extraction, predictions or evaluation |

Every method fits the same training customers and scores the same development customers. IF/MAD use **all training customers**, without label-based filtering, and fitting receives no labels. RF alone uses training labels. No validation, calibration or holdout customers enter fitting. The workflow consumes no calibration bundle or calibration-role manifest.

The historical supervised estimator may have been selected using its entire validation cohort. Freezing it would not remove that selection dependence. Therefore use a **fresh, predeclared RF**, explicitly distinct from published results. Its existing learning parameters are 120 trees, depth 12, minimum leaf size 3 and balanced subsample weights; only execution is set to `n_jobs=1` to prevent parallel floating-point summation from affecting ties.

There is no parameter, method, feature, contamination, threshold or ensemble search. Configurations and capacities are fixed inputs and never chosen using evaluation metrics. Validation diagnostics remain separate from evaluation reporting; no outcomes feed back into fitting. Revising a configuration after reviewing results creates another development analysis, not an independent test.

The source loader validates CSV schema/labels, but returns only recorded train+validation customers to downstream feature extraction. Training must contain both classes for RF; reference validation needs at least four customers and two per class for the stratified split. Insufficient references fail explicitly. Previously researched development customers do not become a new independent cohort through repartitioning.

## Scores and numerical behavior

### Isolation Forest

Fixed settings: 100 trees, `max_samples="auto"` (at most 256 train customers per tree), `max_features=1.0`, `bootstrap=False`, `contamination="auto"`, recorded seed and one worker. Fit only shared train-imputed features.

Score = **`-score_samples(X)`**: higher means more anomalous, not a probability. Sklearn defines `decision_function = score_samples − offset_`; contamination affects its offset/threshold, not raw ordering for the same forest. This benchmark never uses threshold-based `predict` flags as labels. See the [official IsolationForest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

### Robust median/MAD baseline

On the same imputed training matrix, let m_j be feature j's training median, u_j=max(1, |m_j|), and d_j=median_train |x_j/u_j − m_j/u_j|. For each scored customer:

```text
z_j = min(1,000,000, |x_j/u_j − m_j/u_j| / (1.4826 × max(d_j, 1e−9)))
customer_score = mean_j log(1 + z_j)
```

Include only features with at least one observed training value; if none exist, score zero. The 1.4826 factor is normal-consistency scaling. The positive MAD floor is relative to the median's units and handles constant/zero-MAD features. A constant feature contributes zero at its reference value and a bounded deviation when it changes. Entirely missing training features are ignored by MAD and recorded by name.

Median midpoint averaging uses halved values to avoid overflow. Median-based units prevent a single extreme observation from setting the numerical scale. Clip deviations before `log1p` and averaging to bound overflow and single-feature domination. These constants are fixed guards, not tuned parameters. Nonfinite/un-imputed inputs are rejected; missing data use the shared train-fitted imputer first.

Equal weighting can double-count correlated features. Both unusually low and high values can score highly; neither indicates theft. The score is not a probability and its numerical units cannot be compared with RF or IF scores.

## Evaluation and complementarity definitions

For k=min(requested K, cohort size) and F(k) positive labels among the top k:

| Metric | Definition |
| --- | --- |
| AP | Existing sklearn average precision, not trapezoidal PR area |
| ROC-AUC | Existing binary ranking discrimination |
| Precision@K | F(k)/k |
| Recall@K | F(k)/all cohort positives |
| Lift@K | Precision@K/cohort prevalence |
| Intersection | Size of the two top-K sets' intersection |
| Jaccard | Intersection size/union size |
| Overlap fraction | Intersection size/actual k |
| Left/right only | Customers selected by exactly one method |
| Nonoverlap positive rate | Historical positives/nonoverlap customers, when group size permits |
| Spearman rho | Pearson correlation of average score ranks, preserving ties |

Top-K ties use native score descending, then stripped customer ID lexicographically ascending. Correlation does **not** break score ties by ID: constant scores must not acquire artificial correlation. Rho is unavailable with fewer than two customers or constant inputs. No p-values/significance claims are produced. See the [Spearman reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html); implementation uses existing NumPy/pandas, without an added dependency.

Capacities are positive integers, sorted/deduplicated and capped. Saturated queues can have Jaccard=1 simply because both contain everyone. Empty unions have undefined Jaccard. Empty/unlabeled diagnostic cohorts have unavailable discrimination metrics; single-class ROC-AUC is undefined, and no-positive AP/recall/lift retain existing null conventions. The workflow requires labels for RF, while pure diagnostics explicitly handle missing labels. Small and constant cohorts produce warnings.

Distributions include distinct values, min/5th percentile/median/95th percentile/max, mean, standard deviation and 20-bin histograms. Histograms use each cohort's native min/max for display only, never model thresholds. Internal arithmetic is rescaled to avoid overflow; saved and ranked scores stay unchanged. Constant distributions use one point bin. Axes/units remain separate by method.

Missing-reading bins reuse `[0,0.1)`, `[0.1,0.25)`, `[0.25,0.5)` and `[0.5,∞)`. Group AP/ROC-AUC require at least 20 customers by default and five per class. Otherwise metrics are unavailable with a reason. Small-group positive counts and nonoverlap rates are suppressed. These safeguards are not power calculations, formal privacy guarantees, fairness conclusions or causal evidence.

## Reproduction and local artifacts

```powershell
.\.venv\Scripts\python.exe -c "from gridguard.synthetic import save_synthetic; save_synthetic('data/anomaly-demo.csv', customers=7500, days=90, seed=42)"
.\.venv\Scripts\gridguard.exe train --input data/anomaly-demo.csv --source synthetic --validation-only --output-dir artifacts/anomaly-reference
.\.venv\Scripts\gridguard.exe benchmark-anomaly --input data/anomaly-demo.csv --reference-dir artifacts/anomaly-reference --output-dir artifacts/anomaly-development --seed 42 --feature-config baseline --capacities 50 100 200
```

For an existing reference, provide its matching source CSV and standard run directory. Legacy runs lacking schema-v2 membership cannot seed a fair benchmark; existing legacy scoring/dashboard support is unchanged. Output directories must be empty.

| File | Contents |
| --- | --- |
| `anomaly_benchmark.json` | Schema-v2, benchmark-version-1 configs, score definitions, software/seeds, dataset/partition/file fingerprints, both cohorts' aggregates and development-only scope |
| `split_manifest.json` | Private membership of train, two development cohorts and excluded holdout |
| `anomaly_predictions.csv` | Private development IDs, labels, missingness and native scores; no train/holdout predictions |
| `anomaly_models.joblib` | Private fitted RF/imputer, IF and MAD; separate from legacy bundles |

Outputs stay local and are git-ignored even in custom directories. Full reports contain local source paths; publish only curated aggregates. Expand scores to float64 before CSV serialization to preserve exact values/ties. Shared dashboard checks validate consumed-file checksums, disjoint membership, cohort alignment and recomputed aggregates. They do not authenticate untrusted files or open estimator weights.

## Measured synthetic development results

Measured on 2026-11-12: 7,500 synthetic customers, 90 days, seed 42, baseline features. Original train=4,500, development-validation=750, development-evaluation=750, excluded holdout=1,500. Evaluation contained **67 positive labels**. Software: Python 3.13.14, NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, joblib 1.6.0.

| Method | AP | ROC-AUC | Precision@50 | Recall@50 | Lift@50 | Precision@100 | Recall@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fresh RF | 1.0000 | 1.0000 | 1.00 | 0.7463 | 11.1940 | 0.67 | 1.0000 |
| Isolation Forest | 0.5981 | 0.9697 | 0.66 | 0.4925 | 7.3881 | 0.61 | 0.9104 |
| Robust MAD | 0.8310 | 0.9817 | 0.74 | 0.5522 | 8.2836 | 0.63 | 0.9403 |

| Pair | Spearman rho | Top-50 intersection | Top-50 Jaccard | RF-only count/rate | Anomaly-only count/rate |
| --- | --- | --- | --- | --- | --- |
| RF / IF | 0.4696 | 26 | 0.3514 | 24 / 100% | 24 / 29.17% (7 positives) |
| RF / MAD | 0.4707 | 28 | 0.3889 | 22 / 100% | 22 / 40.91% (9 positives) |
| IF / MAD | 0.9292 | 45 | 0.8182 | Not applicable | Nonoverlap rates suppressed (5 per side) |

Some positives appear outside RF's top 50, but unions require **74 or 72 inspections**, not 50. This does not establish improved yield at equal capacity. At K=100, RF already found all 67 positives; anomaly-only groups had no additional positives. IF and MAD largely agreed. The evidence supports examining disagreement, **not an ensemble or deployment claim**.

RF had 21 distinct scores, including many zero/one ties; each anomaly method had 750. Native medians were RF=0, IF=0.4069, MAD=0.4641; numerical magnitudes are not comparable units. Missingness support was narrow: 749 customers had less than 10% missing readings and the one-person higher-missingness group was suppressed. Higher-missingness performance cannot be assessed here.

These are **synthetic injected-drop diagnostics**, not real customer or operational results. Perfect RF discrimination reflects an easy simulator. MAD outperforming IF here illustrates why algorithm complexity alone is not improvement. No real-data model was retrained/rescored; no configuration was revised using these evaluation outcomes. Exact results can vary across recorded software versions.

## Read-only dashboard

```powershell
$env:GRIDGUARD_ANOMALY_REPORT = 'artifacts/anomaly-development/anomaly_benchmark.json'
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Choose **Anomaly Detection** and a saved development cohort. The workspace works independently of the legacy selector and caches eight aggregate reports for five minutes, invalidated by report/manifest/prediction signatures. It exposes no customer IDs or customer-level download. It displays metric comparisons, native distributions, top-K overlap, tie-aware correlations, nonoverlap rates, quality groups and provenance. Rendering never loads weights, trains/refits or predicts from a model. Missing optional artifacts leave other workspaces usable.

## Limitations and next steps

Historical customer-level `FLAG` labels lack event onset dates and may reflect biased inspection coverage or label errors. Unusual patterns can arise from equipment failures, outages, collection problems or a different population. Neither anomaly scores nor disagreement confirm theft or establish that either model is wrong. Automated enforcement is not justified.

Training includes historical positives, so IF/MAD can learn some positive patterns as normal. Correlated features, quality artifacts, a small minority class, sparse subgroups and a single simulator limit interpretation. There are no confidence intervals, multiple-comparison adjustments or claims of significant improvement/independent final performance.

Before considering combinations, predeclare selection using development-validation data, freeze it, and obtain an independent cohort with dated inspection outcomes and adequate subgroup support. Compare equal inspection budgets, population representativeness, shift and uncertainty. Do not reuse the inspected holdout to choose features, parameters, thresholds or ensembles.

## Verification

Run Ruff and pytest with development extras. Tests cover train-only fitting/preprocessing, exact membership, holdout exclusion, deterministic scores/ties, constant/missing/extreme features, correct ranking/overlap/Jaccard/correlation, weak subgroups, artifact corruption, legacy compatibility and dashboard fitting/loading guards. CI retains current/minimum-sklearn coverage and all existing synthetic smoke workflows, adding anomaly CLI verification.

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

Local verification on 2026-11-12: Ruff and all **209 tests** passed. Isolated CLI smoke checks passed for demo, feature experiments, frozen finalization, saved-holdout evaluation, calibration and anomaly benchmarking. A separate synthetic AppTest rendered all eight workspaces and switched both anomaly cohorts with model loading/fitting blocked during rendering. SHA-256 checks confirmed all **65 existing local artifact/report files unchanged**.
