# GridGuard AI

**Electricity consumption analytics, non-technical loss risk scoring, and inspection prioritization.**

GridGuard AI is a reproducible, lightweight machine-learning demonstration for electricity distribution analytics. It takes **daily consumption readings by customer**, builds statistical features, compares two classifiers, and ranks held-out customers for possible investigation.

> **Important:** A risk score is a model-generated prioritization signal, **not evidence of electricity theft**. This project is a portfolio prototype, not an operational fraud-detection system.

## What it does

- Validates customer-by-day wide consumption CSVs (`CONS_NO`, `FLAG`, date columns) and a generic `customer_id` equivalent.
- Extracts consumption and data-quality signals without using the target label as a feature.
- Splits customers into **train / validation / test (60 / 20 / 20)** with stratification and a fixed random seed.
- Compares Logistic Regression and Random Forest using **validation average precision (AP)**; reports held-out test metrics and **Precision/Recall/Lift@K**.
- Saves the selected model, evaluation report, and **out-of-sample test predictions**.
- Shows an interactive Streamlit dashboard with inspection ranking, per-customer reading history, and model metrics.
- Compares inspection capacities with retrospective top-K metrics, random baselines, marginal yield and optional illustrative financial scenarios.
- Supports optional development-only probability calibration and a read-only Model Reliability view, preserving published holdout results.
- Benchmarks Isolation Forest and robust median/MAD against a fresh fixed RF, with disjoint development cohorts and aggregate complementarity analysis.
- Generates clearly labeled **synthetic data** so the demo runs without downloading a third-party dataset.
- Supports optional SQLite export and sample SQL queries.

## Quick start

Python **3.11+** recommended.

```bash
git clone https://github.com/GorkemErtas/gridguard-ai.git
cd gridguard-ai
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[app,dev]"

# Generate synthetic readings, train models, and evaluate held-out customers
gridguard demo

# Launch dashboard
streamlit run app/streamlit_app.py
```

Results are written to `artifacts/` and the generated data goes into `data/`. Both are git-ignored.

The dark analytics dashboard provides Overview, Model performance, Model Reliability, Anomaly Detection, Inspection queue, Inspection Capacity Planning, Customer analytics and Data quality views. See the [dashboard guide](docs/dashboard.md) for exact PowerShell commands using existing final artifacts, metric definitions, private exports and compatibility details. The [capacity planning guide](docs/inspection-capacity-planning.md) explains capacity scenarios, marginal yield and the optional illustrative financial calculator. No model training is required to view an existing experiment.

The optional [anomaly benchmark](docs/anomaly-benchmark.md) adds an **Anomaly Detection** workspace. It compares native anomaly rankings with a fresh development RF without changing published models or reusing the final holdout. Saved reports include ranking overlap, tie-aware correlation and missing-reading groups; anomaly scores are not probabilities or confirmed theft evidence.

### Train on a local labeled consumption dataset

Obtain a dataset under terms that permit your intended research use, record its provenance and license, and place the extracted CSV in `data/consumption.csv`.

```bash
gridguard experiment --input data/consumption.csv --source custom --sample-size 10000 --output-dir artifacts/real-validation
# Once validation decisions are final, evaluate the frozen winner once:
gridguard finalize --input data/consumption.csv --experiment-dir artifacts/real-validation --output-dir artifacts/real-final
streamlit run app/streamlit_app.py
```

Choose `artifacts/real-final` in the dashboard. Use empty output directories to preserve earlier experiments. Omit `--sample-size` to use the full file; this changes the cohort and cannot be compared directly with the pilot's fixed test split. The sample is stratified by label. Third-party data are **not included**; confirm rights before use or redistribution.

See the [local commands](docs/real-data-evaluation.md) and [pipeline audit with measured validation/final results](docs/pipeline-evaluation.md). `train` remains available for a single configuration; use `--validation-only` during development. Existing model bundles keep the baseline feature calculation.

### Other commands

```bash
gridguard score --input data/consumption.csv --model artifacts/model.joblib --output reports/risk_ranking.csv
gridguard export-sqlite --input data/demo.csv --output data/demo.sqlite --limit-customers 200
pytest -q
ruff check .
```

The dashboard defaults to `artifacts/` and can optionally display readings from the original local CSV file. `score` outputs rankings for all supplied customers; **do not interpret training-set predictions as a held-out evaluation**.

## Architecture

```text
CSV / synthetic generator
        |
        v
  data.py (schema + data quality)
        |
        v
  features.py (customer-level features)
        |
        v
  training.py (stratified split -> compare models -> final test)
        |
        +---- model.joblib / metrics.json / test_predictions.csv
        |
        +---- app/streamlit_app.py (read-only dashboard)

  cli.py: demo | train | experiment | finalize | score | export-sqlite
  sql/: optional SQLite exploratory analysis
```

See [architecture](docs/architecture.md) and [model card](docs/model-card.md) for methodology, limitations, and responsible use.

## Evaluation design

**Why Average Precision?** With a minority positive class, ROC-AUC alone can hide poor precision. Selection uses validation **AP**, followed by a **single** final test evaluation. Precision@K measures the fraction of positives among K inspection candidates. Recall@K measures the fraction of all positives found; lift compares precision with prevalence. Rankings use unrounded probabilities and customer-ID tie breaks. AP is not trapezoidal PR-curve area.

The risk score is `round(100 * model.predict_proba(...))`, **not a calibrated probability**. Display bands (low/medium/high) are illustrative and are not company inspection policy.

Optional [`gridguard calibrate`](docs/probability-calibration.md) creates a separate experiment with disjoint base fitting/selection, calibration fitting/selection and diagnostic roles. It compares uncalibrated, sigmoid and eligible isotonic outputs using proper scores, reliability diagrams and existing ranking metrics. The previously inspected holdout is excluded; independent final calibration performance is not established. **Model Reliability** reads saved diagnostics and never trains a calibrator. Calibrated values concern historical labels and cannot establish confirmed theft.

**Known limitations:** historical labels are customer-level (not theft dates), results from one distribution network may not generalize to another, missing readings can reflect outages or collection problems, and historical labels need independent validation. No individual should be accused or penalized based solely on this output.

## Tech stack

Python · pandas · NumPy · scikit-learn · SQLite · Streamlit · Plotly · pytest · Ruff · GitHub Actions

## License and data rights

No software license has been selected yet. The source is publicly visible, but reuse rights are not automatically granted. Third-party datasets may have separate, restrictive terms; do not redistribute them without permission.
