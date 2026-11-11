"""Evaluate saved held-out SGCC predictions without retraining or tuning.

Usage:
    python scripts/evaluate_holdout.py --artifacts artifacts --output-dir reports/sgcc
Only use artifacts produced by `gridguard train`, never in-sample `gridguard score`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def evaluate(artifacts: Path, output_dir: Path, *, seed: int = 42, bootstrap: int = 1000) -> dict:
    report = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    df = pd.read_csv(artifacts / "test_predictions.csv", dtype={"customer_id": "string"})
    required = {"customer_id", "true_label", "model_score", "risk_score"}
    if not required.issubset(df):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(df))}")
    if df.empty or df.customer_id.isna().any() or df.customer_id.duplicated().any():
        raise ValueError("Expected nonempty, unique customer IDs")
    if df.true_label.isna().any() or not df.true_label.isin([0, 1]).all():
        raise ValueError("Invalid binary ground-truth labels")
    if not np.isfinite(df.model_score).all() or not df.model_score.between(0, 1).all():
        raise ValueError("Invalid model probabilities")
    if len(df) != report["split_sizes"]["test"]:
        raise ValueError("Predictions do not match test split size")
    y = df.true_label.to_numpy(dtype=int)
    p = df.model_score.to_numpy(dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must be represented in the holdout")
    ordered = df.sort_values(["model_score", "customer_id"], ascending=[False, True]).copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "source": report["dataset"]["source"],
        "test_customers": len(df),
        "positive_prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix_at_0_5": confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist(),
        "precision_at_k": {},
    }
    for k in (50, 100, 200):
        actual_k = min(k, len(ordered))
        metrics["precision_at_k"][str(k)] = {
            "evaluated_k": actual_k,
            "precision": float(ordered.true_label.iloc[:actual_k].mean()),
            "positives_found": int(ordered.true_label.iloc[:actual_k].sum()),
        }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) == 2:
            samples.append(average_precision_score(y[idx], p[idx]))
    metrics["average_precision_bootstrap_95_ci"] = (
        [float(x) for x in np.quantile(samples, [0.025, 0.975])]
        if samples else None
    )
    errors = df.copy()
    errors["predicted_label_at_0_5"] = (errors.model_score >= 0.5).astype(int)
    errors["error_type"] = np.select(
        [
            (errors.true_label == 0) & (errors.predicted_label_at_0_5 == 1),
            (errors.true_label == 1) & (errors.predicted_label_at_0_5 == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    errors.loc[errors.error_type != "correct"].sort_values(
        "model_score", ascending=False
    ).to_csv(output_dir / "errors.csv", index=False)
    (output_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sgcc"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    print(json.dumps(evaluate(args.artifacts, args.output_dir, bootstrap=args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
