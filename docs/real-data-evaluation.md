# Real-world consumption data: local training and evaluation

This project supports a generic labeled customer-by-day consumption CSV. Required columns are `CONS_NO` (or `customer_id`), binary `FLAG`, and at least 60 date columns. Calendar headers such as `2014/1/1` and `1/1/2014` are supported.

## Local preparation

If your data provider distributes a multipart ZIP, keep all archive parts together and extract from the main `.zip` using 7-Zip. Place the resulting CSV at `data/consumption.csv`. Never commit the raw data or customer-level artifacts.

## Validate, train and evaluate (Windows PowerShell)

```powershell
python -m pip install -e ".[app,dev]"
python -m gridguard.data_preflight --input data/consumption.csv --output reports/data-quality.json
gridguard train --input data/consumption.csv --source custom --sample-size 10000 --output-dir artifacts/real-pilot
python scripts/evaluate_holdout.py --artifacts artifacts/real-pilot --output-dir reports/real-pilot
```

After verifying pilot quality and metrics, repeat without `--sample-size` and use distinct `artifacts/real-full` and `reports/real-full` paths.

## Methodology

- Retrospective customer-level binary classification; a risk score is not evidence of wrongdoing.
- Model selection uses validation Average Precision. Report held-out PR-AUC, ROC-AUC, Precision@50/100/200, false positives, and false negatives.
- Current saved estimator is fitted to the 60% training split, not refitted to train+validation.
- Labels lack a known event-onset time. Features from the complete observation period are not a time-ahead forecast.
- Bootstrap intervals assume independent sampled customers and do not establish temporal or geographic generalization.
- Never use test results for repeated model selection.
- Keep customer IDs, raw readings, trained artifacts, and error reports private unless the data license explicitly allows redistribution.

## Data provenance and rights

Document the actual source, authors, citation, license or permission terms in your research notes and any public report using its results. Generic code and filenames do **not** remove attribution obligations. Do not publish source records or trained weights until redistribution rights have been verified.
