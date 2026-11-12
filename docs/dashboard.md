# Local risk analytics dashboard

The dashboard reads existing experiment metadata and saved predictions. It never loads joblib bundles, trains estimators, tunes thresholds or rewrites artifacts. Install the app extra with Streamlit 1.65+ for the responsive native containers, accessible charts and theme options used here. Plotly remains the existing chart dependency. Fonts use the local system sans-serif stack; no remote font request is needed.

## Launch with the existing final experiment

From Windows PowerShell:

```powershell
Set-Location 'C:\Users\gorke\OneDrive\Belgeler\GitHub\gridguard-ai'
.\.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
$env:GRIDGUARD_ARTIFACT_DIR = 'artifacts/pipeline-final-20261111'
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://localhost:8501`. The sidebar also discovers `metrics.json` up to two experiment-directory levels below `artifacts/`. Select a saved experiment or use Custom path. The environment variable only chooses the initial path; it contains no metrics. Remove it with `Remove-Item Env:GRIDGUARD_ARTIFACT_DIR` to return to the `artifacts` default.

For this final run, the Data quality section defaults to `reports/pipeline-final-20261111/error_analysis.json`. The source audit defaults to `reports/data-quality.json`; choose another existing preflight JSON if needed. A source audit may describe the full CSV while model metadata describe a sample; these scopes are labeled separately. No audit is computed by loading the entire raw dataset during an interaction.

## Sections and definitions

- **Overview:** observed AP/ROC-AUC, experiment population, missingness, risk bands, raw-score distribution, cohort/provenance information and operational limitations.
- **Model performance:** precision/recall/F1 and confusion matrix at the illustrative 0.5 threshold, Precision/Recall/Lift@50/100/200, and saved validation comparisons. With labels available, metrics are derived from the saved raw predictions; AP is average precision, not trapezoidal PR area. K is capped at the cohort size. Without labels, stored summary metrics can be shown, but customer ranking metrics remain unavailable.
- **Model Reliability:** optional saved development-only calibration diagnostics: reliability diagram, bin counts, Brier/log-loss/ECE/AP/ROC-AUC, ranking comparison and method/partition/software metadata. It works independently of the legacy selector and never loads/fits an estimator or calibrator. Set `GRIDGUARD_RELIABILITY_REPORT` or enter a `reliability.json` path. Missing calibration artifacts do not affect the original workspaces. See [calibration workflow and limitations](probability-calibration.md).
- **Inspection queue:** exact raw-score order, with customer-ID tie breaks. Band and literal ID-search filters establish eligibility before filling capacity. Global ranks are retained. Displayed precision uses displayed queue positives/queue size; recall uses all cohort positives; lift divides queue precision by full-cohort prevalence. These retrospective filtered-queue metrics are different from the unfiltered top-K metrics in Model performance.
- **Inspection Capacity Planning:** capacity slider, exact unfiltered Precision/Recall/Lift@K, observed positives, analytical random baselines, operational scenario comparison and marginal yield. Optional user-entered inspection costs and hypothetical recovery values produce illustrative arithmetic, not validated financial forecasts. Candidate CSVs reuse the private export allowlist. See the [capacity planning guide](inspection-capacity-planning.md) for formulas, edge cases and the reusable module.
- **Customer analytics:** local ID search returns at most 100 selectable matches. History loading requires explicit form submission and scans 250 customers per chunk. The daily chart preserves gaps, distinguishes missing/invalid days from observed zeros, and retains extreme observations. Signals are descriptive rules, not model attribution. The source CSV used for history is not fingerprint-verified against the model dataset.
- **Data quality:** saved sample metadata, label counts, holdout missingness where available, optional source-audit counts (including zeros, negatives and nonnumeric/nonfinite readings), and aggregate error groups. Missing reports and suppressed/undefined group rates stay unavailable rather than becoming zero.

## Compatibility, privacy and limitations

The optional **Anomaly Detection** workspace reads `anomaly_benchmark.json` from `GRIDGUARD_ANOMALY_REPORT` or a user-entered path. It works independently of the legacy selector and displays only aggregate saved RF/IF/MAD comparisons, native-score distributions, overlap/Jaccard, tie-aware rank correlations, nonoverlap label rates and supported missing-reading groups. It never loads weights or trains/refits a model, and provides no customer IDs or customer-level downloads. The eight-entry/five-minute cache includes report, manifest and prediction-file signatures. Missing optional reports do not block original workspaces. See [anomaly methodology and reproduction](anomaly-benchmark.md).

Legacy and schema-v2 experiments are supported without reading estimator bundles. Validation-only runs show validation summaries and aggregate validation groups, with no invented customer queue. Synthetic experiments are prominently labeled. The previously inspected holdout is described as retrospective; it is not presented as untouched or as prospective inspection performance.

New artifacts validate prediction checksums and customer partition manifests. If the optional local manifest is absent, the checksum still must match and membership is labeled unverified. Legacy reports explicitly lack recorded membership verification. A malformed primary metrics/predictions file blocks rendering with a recoverable selector; missing optional aggregate reports do not block the other sections.

All downloaded CSVs are private customer-level material, even with IDs excluded: ranks and scores can link records. Exports default to excluding IDs and always omit ground truth, reading signals and arbitrary extra fields. ID inclusion requires explicit opt-in; spreadsheet formula prefixes in IDs are escaped. Browser views contain local customer IDs for inspection, so keep the app on localhost and avoid public hosting or sharing screenshots with identifying information. The application is not an authentication/access-control system.

Caches are bounded (8 experiments, 8 capacity curves, 8 reliability reports, 8 anomaly reports, 32 optional reports, 16 individual histories) with a five-minute TTL; file modification signatures invalidate caches when artifacts change. Capacity curve keys also include the prediction frame. Saved development-report signatures include the report, manifest and prediction CSV. Only the selected section executes. The raw CSV is read only on history submission; repeated lookups may still scan a large portion of the file, but memory stays bounded. The dashboard does not validate source-audit provenance or guarantee future model performance. Original risk scores remain uncalibrated; neither raw, calibrated nor anomaly scores establish theft or justify automated enforcement.

## Verification

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

The synthetic dashboard smoke test in `tests/test_dashboard.py` loads a temporary labeled synthetic artifact, navigates all eight sections (optional evaluations are gracefully unavailable without their artifacts) and exercises an empty queue filter. Planning tests cover zero/full capacity, private exports, financial inputs, invalid capacities and degenerate cohorts. Reliability and anomaly tests render saved development diagnostics with model loading/fitting blocked, verify scope/integrity controls, and check legacy compatibility. Existing pipeline tests use their own small synthetic training fixtures.

The original dashboard change passed Ruff and 47 tests, including its synthetic dashboard/history smoke test and a separate read-only AppTest check against the existing `pipeline-final-20261111` artifacts. Capacity planning extends this coverage with `tests/test_capacity_planning.py` and the planning interactions above. No real-data training, tuning, estimator loading or new holdout scoring is required.
