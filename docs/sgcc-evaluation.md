# Real SGCC evaluation protocol

This is a **retrospective customer-level classification** study, not proof of theft or a prospective field trial.

1. Download the dataset from the [IEEE PES SGCC catalog](https://ieee-pes-data-sharing.org/datasets/detail/663b8e0e-3ace-48f8-b983-4cb59b39fa6f). Review usage rights. Never commit the raw dataset, customer identifiers, or generated reports.
2. Inspect column names, unique customers, date continuity, label prevalence, missing readings, negatives, and implausible consumption. Record these checks before training.
3. Install: `python -m pip install -e ".[app,dev]"`
4. First pilot: `gridguard train --input data/sgcc.csv --source sgcc --sample-size 10000 --output-dir artifacts/sgcc-pilot`
5. Evaluate: `python scripts/evaluate_holdout.py --artifacts artifacts/sgcc-pilot --output-dir reports/sgcc-pilot`
6. Full experiment: repeat without `--sample-size` into separate output directories.
7. Compare test average precision against positive prevalence; inspect ROC-AUC, precision at available K, and confusion matrix. Review false positives and false negatives in `errors.csv` **locally**.
8. Do not select models or thresholds by repeatedly consulting the held-out test set. Use validation data for those choices. The current training implementation evaluates the selected model trained on the 60% training partition; refitting on train+validation is a separate future change that must be tested.
9. Record limitations: no dated theft onset, inspection selection bias, SGCC-to-Türkiye transfer uncertainty, uncalibrated probabilities, and no prospective outcome measurement.

Bootstrap confidence intervals in the evaluation script are descriptive IID customer resamples, not a substitute for temporal or geographic validation. Error files may contain sensitive customer identifiers; do not publish them.
