# SGCC data extraction and local training (Windows)

The original data repository is [henryRDlab/ElectricityTheftDetection](https://github.com/henryRDlab/ElectricityTheftDetection). Its README describes a split ZIP archive containing SGCC daily readings for 42,372 customers over 1,035 days, January 2014–October 2016. The three files `data.z01`, `data.z02` and `data.zip` are **parts of one archive**, not three independent ZIPs.

## 1. Unpack the archive

Install 7-Zip and keep all three files in the same folder, with their original names.

- Right-click **data.zip** → **7-Zip** → **Extract to "data\\"**. Do not extract `data.z01` or `data.z02` separately.
- Alternatively, in PowerShell opened in the download folder: `7z x .\\data.zip -o.\\sgcc_extracted`
- Look inside the resulting folder for a CSV, and check its header. If it is called `data.csv`, copy it into the GridGuard project as `data/sgcc.csv`. Do not rename a ZIP to CSV.
- Extraction errors usually mean an incomplete file, mismatched filenames, or a ZIP part located in a different folder.

## 2. Validate before training

Run the following commands from the root of the GridGuard project after checking out branch `feature/sgcc-evaluation`:

```powershell
python -m pip install -e ".[app,dev]"
python -m gridguard.sgcc_preflight --input data/sgcc.csv --output reports/sgcc-data-quality.json
```

Review total rows, positive-label prevalence, date span, missing calendar dates and missing/negative/zero readings. Do not commit data or customer-level outputs. The CSV parser accepts both `2014/1/1` and `1/1/2014` date header formats.

## 3. Run a small pilot, then full evaluation

```powershell
gridguard train --input data/sgcc.csv --source sgcc --sample-size 10000 --output-dir artifacts/sgcc-pilot
python scripts/evaluate_holdout.py --artifacts artifacts/sgcc-pilot --output-dir reports/sgcc-pilot
```

Only after reviewing the pilot and its performance, run the full dataset:

```powershell
gridguard train --input data/sgcc.csv --source sgcc --output-dir artifacts/sgcc-full
python scripts/evaluate_holdout.py --artifacts artifacts/sgcc-full --output-dir reports/sgcc-full
```

## Evaluation protocol and caveats

- This is a retrospective customer-level classification study, not proof of electricity theft or a prospective field trial.
- Models are selected using validation Average Precision, with an untouched customer-level holdout test; never tune on test labels.
- Compare test AP to positive prevalence (a no-skill reference) and evaluate ROC-AUC, Precision@50/100/200 and false-positive/false-negative cases.
- The current training implementation **does not refit** the selected estimator on combined train+validation; the saved model was fitted to the 60% training split.
- SGCC `FLAG` labels do not identify theft onset; whole-history features are not a time-ahead prediction experiment. No geographic transfer, time-forward generalization, or calibrated probability claim is supported.
- Confidence intervals are descriptive bootstrap resamples over the held-out customers.
- Keep all raw CSVs, customer IDs, test prediction files, model artifacts, and error reports local while dataset and redistribution permissions remain unverified; `.gitignore` already excludes the `data/`, `artifacts/`, and `reports/` outputs.
- Cite Zheng et al. (2018), *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids* (DOI: 10.1109/TII.2017.2785963), when describing use of these data.
