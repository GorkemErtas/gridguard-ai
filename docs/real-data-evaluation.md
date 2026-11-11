# Real-world consumption data: local training and evaluation

This project supports a generic labeled customer-by-day consumption CSV. Required columns are `CONS_NO` (or `customer_id`), binary `FLAG`, and at least 60 date columns. Calendar headers such as `2014/1/1` and `1/1/2014` are supported.

## Local preparation

If your data provider distributes a multipart ZIP, keep all archive parts together and extract from the main `.zip` using 7-Zip. Place the resulting CSV at `data/consumption.csv`. Never commit the raw data or customer-level artifacts.

## Validate and compare on validation data (Windows PowerShell)

```powershell
python -m pip install -e ".[app,dev]"
python -m gridguard.data_preflight --input data/consumption.csv --output reports/data-quality.json
gridguard experiment --input data/consumption.csv --source custom --sample-size 10000 --seed 42 --experiment-id validation-next --output-dir artifacts/validation-next
```

This compares `baseline`, `calendar`, `quality` and `combined` with identical customer partitions. It records all candidate validation metrics and selects the highest non-dummy validation AP. No test features, predictions or metrics are produced. Each run includes `validation_error_analysis.json`, plus a local split manifest, fitted model and full metadata. `validation_summary.json` contains allowlisted aggregate research fields without the local input path or customer membership. Review it and the data rights before sharing.

Every output directory must be empty. Use new experiment paths instead of overwriting `artifacts/real-pilot` or prior research results. For a single baseline reproduction without test evaluation:

```powershell
gridguard train --input data/consumption.csv --source custom --sample-size 10000 --seed 42 --feature-config baseline --validation-only --experiment-id baseline-next --output-dir artifacts/baseline-next
```

## Freeze selection and evaluate once

After validation decisions are final, evaluate the selected saved estimator under a separate experiment ID. It remains fitted on train only, with no train+validation refit. Finalization verifies the input fingerprint, exact partitions, model checksum and recorded software versions. Load only trusted local joblib artifacts.

```powershell
gridguard finalize --input data/consumption.csv --experiment-dir artifacts/validation-next --experiment-id final-next --output-dir artifacts/final-next
python scripts/evaluate_holdout.py --artifacts artifacts/final-next --output-dir reports/final-next --seed 42 --bootstrap 1000
streamlit run app/streamlit_app.py
```

Select `artifacts/final-next` in the dashboard's artifact-directory field. The evaluation script summarizes saved predictions, checks their checksum/customer membership and writes aggregate `evaluation.json` and `error_analysis.json`. It does not call a model. Add `--export-customer-errors` only for a private local `errors.csv`; never commit it. Legacy pilot predictions can still be summarized in a new report directory, but report `legacy_unverified` membership provenance.

Omitting `--sample-size` creates a different full-data experiment and different partitions. Its results are not a comparison on the pilot's fixed test cohort. Once the final test has been reported, use a fresh independent cohort for further model selection claims.

## Methodology

- Retrospective customer-level binary classification; a risk score is not evidence of wrongdoing.
- Model selection uses validation Average Precision (AP, not trapezoidal PR-curve area). Report AP, ROC-AUC, Precision/Recall/Lift@50/100/200, false positives, and false negatives.
- Current saved estimator is fitted to the 60% training split, not refitted to train+validation.
- Labels lack a known event-onset time. Features from the complete observation period are not a time-ahead forecast.
- Bootstrap intervals assume independent sampled customers and do not establish temporal or geographic generalization.
- Never use test results for repeated model selection.
- Keep customer IDs, raw readings, trained artifacts, and error reports private unless the data license explicitly allows redistribution.

`baseline` and older bundles retain the original 15-feature calculation. `calendar` inserts absent calendar dates as missing before aggregation and recent windows; `quality` adds missing/zero streaks and IQR without changing the baseline calculations; `combined` applies both. Missing days break zero streaks. Extreme readings are retained. See the [audit and measured results](pipeline-evaluation.md) for the experiment protocol, trade-offs and limitations.

## Data provenance and rights

Document the actual source, authors, citation, license or permission terms in your research notes and any public report using its results. Generic code and filenames do **not** remove attribution obligations. Do not publish source records or trained weights until redistribution rights have been verified.

Source attribution for the measured pilot and follow-up experiments is retained in the [research note](pipeline-evaluation.md#research-data-attribution).
