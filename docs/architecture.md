# Architecture and design decisions

## Goal

Demonstrate an end-to-end, readable approach to ranking **customer-level** electricity non-technical-loss risk. The project deliberately avoids a production-shaped distributed architecture without production needs.

## Layers

| Layer | Module | Responsibility |
| --- | --- | --- |
| Input contract | \`data.py\` | Normalize wide customer-by-day schema, sort dates, validate IDs and labels, reject unknown columns, treat negative readings as missing |
| Feature engineering | \`features.py\` | Deterministic and testable customer-level statistical signals |
| Modeling | \`modeling.py\` | Candidate pipelines, meaningful imbalanced-learning metrics, ranking |
| Experiment provenance | `provenance.py` | Dataset/partition checksums, versions, preprocessing metadata, aggregate validation summary |
| Evaluation and error analysis | `evaluation.py`, `error_analysis.py` | Saved-holdout integrity, bootstrap AP interval, aggregate threshold/top-K errors and fixed quality groups |
| Training and inference | \`training.py\` | Stratified 60/20/20 split, validation-only selection, final test, serialization, new-data scoring |
| Orchestration | \`cli.py\` | Explicit workflows; optional bounded SQLite export |
| Presentation | \`app/streamlit_app.py\` | **Read-only** holdout report, inspection queue and reading plot |

The dashboard consumes saved predictions and metrics; it does not retrain on every page refresh. Models are trained once via the CLI and reused for later inference.

## Leakage and evaluation controls

1. The \`FLAG\` target is held in a separate series; it cannot enter the feature matrix.
2. All dates for one customer stay together: split on customers, never split rows/days into train and test.
3. Feature extraction uses only one customer's readings and no population-level fitted transform.
4. Imputation and scaling are fitted **inside** the training split by scikit-learn pipelines.
5. Model choice uses the validation set's average precision, never test performance.
6. The report's \`test_predictions.csv\` contains only test customers.
7. \`gridguard score\` is an inference workflow and MUST NOT be used as proof of held-out accuracy on the same training data.
8. `gridguard experiment` uses only train/validation features and saves a frozen validation winner; `finalize` evaluates that same estimator without refitting. Empty output directories preserve earlier experiments.
9. New holdout exports include a prediction checksum and private split manifest. Evaluation rejects overlap, membership substitution and mixed artifacts; old reports are explicitly unverified.

Feature configurations are recorded with the model, so inference applies the training-time calculation. Baseline remains compatible with existing bundles. Calendar handling and the three quality features are explicit controlled ablations. See the [audit](pipeline-evaluation.md) and [reproduction commands](real-data-evaluation.md).

**Caveat:** because dataset labels are assigned at the *customer level* without theft onset dates, this is historical account-level classification, **not prospective detection**. A prospective evaluation would require dated inspection outcomes, temporal cohort design, and separation of customers with repeated records.

## Cost-conscious deployment

- CPU-only sklearn pipelines; no GPU.
- Reproducible local artifacts, no paid inference API.
- Lightweight Streamlit app, no always-on database.
- SQLite is optional for SQL demonstration, with a 1,000-customer export cap.
- No Redis, message queue, object storage, or microservice boundary without a justified scaling bottleneck.
- Never commit \`data/\`, \`artifacts/\` or \`reports/\`.

For a real deployment, investigate data ownership and governance, ingestion contracts, temporal validation, label acquisition and bias, model and data drift, human review workflows, audit logs, access control and alert fatigue before infrastructure choices.

## Reproducibility

Use Python 3.11+, install the editable package, run \`gridguard demo\`, and execute \`pytest -q\`. CI runs lint, synthetic tests, the existing demo smoke test, and the validation/finalization/saved-holdout workflow. Real customer data are never required.
