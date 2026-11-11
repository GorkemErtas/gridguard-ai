"""Summarize saved holdout predictions without retraining or tuning."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from gridguard.error_analysis import aggregate_errors
from gridguard.modeling import model_metrics
from gridguard.provenance import file_fingerprint, partition_fingerprint, write_json


def _verify_provenance(artifacts: Path, report: dict, predictions: pd.DataFrame) -> str:
    if report.get("schema_version", 1) < 2:
        return "legacy_unverified: only schema and split size checked; membership not recorded"
    if report.get("test") is None:
        raise ValueError("Validation-only experiments have no final holdout evaluation.")
    if report.get("test_predictions_sha256") != file_fingerprint(
        artifacts / "test_predictions.csv"
    ):
        raise ValueError("Holdout prediction fingerprint mismatch.")
    manifest = json.loads((artifacts / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_fingerprint_sha256"] != report["dataset"]["fingerprint_sha256"]:
        raise ValueError("Dataset fingerprint mismatch in split manifest.")
    parts = manifest["partitions"]
    seen = set()
    for key in ("train", "validation", "test"):
        ids = pd.Series(parts[key], dtype="string").str.strip()
        if (
            ids.isna().any()
            or ids.eq("").any()
            or ids.duplicated().any()
            or len(ids) != report["split_sizes"][key]
            or seen.intersection(ids)
        ):
            raise ValueError("Split manifest contains invalid or overlapping customer partitions.")
        if partition_fingerprint(ids) != report["partition_fingerprints"][key]:
            raise ValueError("Customer partition fingerprint mismatch.")
        seen.update(ids)
    if set(predictions.customer_id) != set(parts["test"]):
        raise ValueError("Prediction customers differ from the held-out test partition.")
    return "verified: prediction checksum and disjoint customer partition manifest"


def evaluate(
    artifacts: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    bootstrap: int = 1000,
    export_customer_errors: bool = False,
    min_group_size: int = 20,
) -> dict:
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    artifacts, output_dir = Path(artifacts), Path(output_dir)
    report = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    df = pd.read_csv(
        artifacts / "test_predictions.csv",
        dtype={"customer_id": "string"},
        float_precision="round_trip",
    )
    required = {"customer_id", "true_label", "model_score", "risk_score"}
    if not required.issubset(df):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(df))}")
    df["customer_id"] = df.customer_id.str.strip()
    if (
        df.empty
        or df.customer_id.isna().any()
        or df.customer_id.eq("").any()
        or df.customer_id.duplicated().any()
    ):
        raise ValueError("Expected nonempty, unique customer IDs")
    if df.true_label.isna().any() or not df.true_label.isin([0, 1]).all():
        raise ValueError("Invalid binary ground-truth labels")
    if len(df) != report["split_sizes"]["test"]:
        raise ValueError("Predictions do not match test split size")
    y, p = df.true_label.to_numpy(dtype=int), df.model_score.to_numpy(dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must be represented in the holdout")
    computed = model_metrics(y, p, customer_ids=df.customer_id)
    provenance = _verify_provenance(artifacts, report, df)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty report directory: {output_dir}")
    metrics = {
        "source": report["dataset"]["source"],
        "experiment_id": report.get("experiment_id", "legacy"),
        "holdout_provenance": provenance,
        "test_customers": len(df),
        "positive_prevalence": computed["positive_prevalence"],
        "average_precision": computed["average_precision"],
        "roc_auc": computed["roc_auc"],
        "confusion_matrix_at_0_5": computed["confusion_matrix"],
        "precision_at_k": computed["ranking_at_k"],
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) == 2:
            samples.append(average_precision_score(y[idx], p[idx]))
    metrics["bootstrap"] = {
        "seed": seed,
        "requested_iterations": bootstrap,
        "valid_iterations": len(samples),
        "assumption": "independent customers; descriptive interval, not tuning",
    }
    metrics["average_precision_bootstrap_95_ci"] = (
        [float(x) for x in np.quantile(samples, [0.025, 0.975])] if samples else None
    )
    analysis = aggregate_errors(df, min_group_size=min_group_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "error_analysis.json", analysis)
    if export_customer_errors:
        errors = df.copy()
        errors["predicted_label_at_0_5"] = (p >= 0.5).astype(int)
        errors["error_type"] = np.where(y == 0, "false_positive", "false_negative")
        errors.loc[y != (p >= 0.5)].sort_values(
            ["model_score", "customer_id"], ascending=[False, True]
        ).to_csv(output_dir / "errors.csv", index=False)
    write_json(output_dir / "evaluation.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/holdout"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument(
        "--export-customer-errors",
        action="store_true",
        help="Write private customer-level errors.csv (local use only)",
    )
    args = parser.parse_args()
    if args.bootstrap < 0 or args.min_group_size < 1:
        parser.error("--bootstrap must be nonnegative and --min-group-size must be positive")
    print(
        json.dumps(
            evaluate(
                args.artifacts,
                args.output_dir,
                seed=args.seed,
                bootstrap=args.bootstrap,
                min_group_size=args.min_group_size,
                export_customer_errors=args.export_customer_errors,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
