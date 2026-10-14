# GridGuard AI

**Electricity consumption analytics, non-technical loss risk scoring, and inspection prioritization.**

GridGuard AI is a reproducible, lightweight machine-learning demonstration for electricity distribution analytics. It takes **daily consumption readings by customer**, builds statistical features, compares two classifiers, and ranks held-out customers for possible investigation.

> **Important:** A risk score is a model-generated prioritization signal, **not evidence of electricity theft**. This project is a portfolio prototype, not an operational fraud-detection system.

## What it does

- Validates SGCC-style wide consumption CSVs (`CONS_NO`, `FLAG`, date columns) and a generic `customer_id` equivalent.
- Extracts consumption and data-quality signals without using the target label as a feature.
- Splits customers into **train / validation / test (60 / 20 / 20)** with stratification and a fixed random seed.
- Compares Logistic Regression and Random Forest using **validation average precision (PR-AUC)**; reports untouched test-set metrics and **Precision@K**.
- Saves the selected model, evaluation report, and **out-of-sample test predictions**.
- Shows an interactive Streamlit dashboard with inspection ranking, per-customer reading history, and model metrics.
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

### Use the real SGCC dataset

Download and extract the dataset yourself from the [IEEE PES data sharing catalog](https://ieee-pes-data-sharing.org/datasets/detail/663b8e0e-3ace-48f8-b983-4cb59b39fa6f), review the source's terms, and place the extracted CSV in `data/sgcc.csv`.

```bash
gridguard train --input data/sgcc.csv --source sgcc --sample-size 10000
streamlit run app/streamlit_app.py
```

Omit `--sample-size` to train on the full file. The sample is stratified by label, not just the first rows. Real SGCC data are **not included** in this repository because the catalog does not specify a reuse license.

### Other commands

```bash
gridguard score --input data/sgcc.csv --model artifacts/model.joblib --output reports/risk_ranking.csv
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

  cli.py: demo | train | score | export-sqlite
  sql/: optional SQLite exploratory analysis
```

See [architecture](docs/architecture.md) and [model card](docs/model-card.md) for methodology, limitations, and responsible use.

## Evaluation design

**Why PR-AUC?** With a minority positive class, ROC-AUC alone can hide poor precision. Selection is made using validation **average precision**, followed by a **single** final test evaluation. Precision@K illustrates the real operational question: *if inspection capacity is K, how many truly positive customers are among the top K?*

The risk score is `round(100 * model.predict_proba(...))`, **not a calibrated probability**. Display bands (low/medium/high) are illustrative and are not company inspection policy.

**Known limitations:** historical labels are customer-level (not theft dates), SGCC results may not generalize to Türkiye, missing readings can reflect outages or collection problems, and historical labels need independent validation. No individual should be accused or penalized based solely on this output.

## Tech stack

Python · pandas · NumPy · scikit-learn · SQLite · Streamlit · Plotly · pytest · Ruff · GitHub Actions

## License and data rights

No software license has been selected yet. The source is publicly visible, but reuse rights are not automatically granted. The SGCC dataset has separate, potentially restrictive terms; do not redistribute it with this repository.
