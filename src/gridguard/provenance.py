"""Aggregate experiment provenance; customer membership stays in local artifacts."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from gridguard.data import ConsumptionData, load_consumption


def file_fingerprint(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def partition_fingerprint(ids) -> str:
    payload = json.dumps(list(map(str, ids)), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_partition_manifest(
    report: dict, manifest: dict, *, names=("train", "validation", "test")
) -> dict[str, list[str]]:
    """Verify recorded sizes, fingerprints and disjoint customer membership.

    Shared by holdout evaluation and optional development-only workflows.
    This verifies consistency, not authenticity of untrusted local files.
    """
    if manifest["dataset_fingerprint_sha256"] != report["dataset"]["fingerprint_sha256"]:
        raise ValueError("Dataset fingerprint mismatch in split manifest.")
    parts = manifest["partitions"]
    if not isinstance(parts, dict) or set(parts) != set(names):
        raise ValueError("Split manifest partition names differ from the expected protocol.")
    seen, normalized = set(), {}
    for key in names:
        if not isinstance(parts[key], list):
            raise ValueError("Split manifest customer partitions must be lists.")
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
        normalized[key] = ids.tolist()
    return normalized


def load_reference_development(input_path, reference_dir):
    """Verify a standard schema-v2 reference and return only train+validation.

    Used by opt-in development workflows. Historical models/predictions/metrics
    are never used for selection. Source schema validation reads the CSV, but
    downstream feature extraction cannot access its reserved test customers.
    """
    reference_dir = Path(reference_dir)
    parent = json.loads((reference_dir / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(parent, dict) or parent.get("schema_version", 1) < 2:
        raise ValueError(
            "Development requires a schema-v2 reference with a recorded split manifest."
        )
    manifest = json.loads((reference_dir / "split_manifest.json").read_text(encoding="utf-8"))
    parts = verify_partition_manifest(parent, manifest)
    if not parts["test"]:
        raise ValueError("The reference must record a nonempty reserved final holdout.")
    if file_fingerprint(input_path) != parent["dataset"]["fingerprint_sha256"]:
        raise ValueError("Dataset fingerprint differs from the reference experiment.")
    full = load_consumption(input_path)
    positions = {customer: i for i, customer in enumerate(full.customer_ids)}
    membership = [customer for ids in parts.values() for customer in ids]
    if len(membership) != parent["dataset"]["customers"] or not set(membership) <= positions.keys():
        raise ValueError("Reference customer membership differs from the source dataset.")
    indices = [positions[customer] for customer in parts["train"] + parts["validation"]]
    return full.subset(indices), parent, parts


def experiment_metadata(
    data: ConsumptionData,
    partitions: dict,
    *,
    input_path,
    sample_size,
    source,
    seed,
    experiment_id,
    feature_config,
) -> dict:
    dates = pd.to_datetime(data.readings.columns)
    return {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "dataset": {
            "source": source,
            "input_path": str(input_path),  # Local convenience for the dashboard; do not publish.
            "fingerprint_sha256": file_fingerprint(input_path),
            "customers": len(data),
            "requested_sample_size": sample_size,
            "sampling": "stratified fixed-seed sample, original CSV order retained",
            "reading_days": len(dates),
            "date_start": dates[0].date().isoformat(),
            "date_end": dates[-1].date().isoformat(),
            "missing_calendar_days": int((dates[-1] - dates[0]).days + 1 - len(dates)),
            "positive_customers": int(data.labels.sum()),
            "positive_rate": float(data.labels.mean()),
            "missing_reading_rate": float(data.readings.isna().to_numpy().mean()),
        },
        "methodology": {
            "split": "customer-level stratified 60/20/20",
            "seed": seed,
            "selection_metric": "validation average_precision",
            "fitted_on": "train only; no train+validation refit",
            "ranking_ties": "raw probability descending, customer_id ascending",
            "risk_score": "rounded 100 * uncalibrated predicted probability",
            "bands": "illustrative: high >=80, medium >=50, low <50",
        },
        "split_sizes": {key: len(idx) for key, idx in partitions.items()},
        "partition_fingerprints": {
            key: partition_fingerprint(data.customer_ids.iloc[idx])
            for key, idx in partitions.items()
        },
        "feature_config": feature_config,
        "preprocessing": {
            "invalid_readings": "negative, nonnumeric and nonfinite values become missing",
            "extreme_readings": "retained; no clipping or zero fill",
            "calendar": "insert missing dates as NaN"
            if feature_config in ("calendar", "combined")
            else "original observed date columns (pilot compatibility)",
            "windows": "last 30/previous 30 calendar days"
            if feature_config in ("calendar", "combined")
            else "last 30/previous 30 reading columns",
            "learned_transforms": "train-only median imputation; train-only LR scaling",
        },
        "software": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("gridguard-ai", "numpy", "pandas", "scikit-learn", "joblib")
            },
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def validation_summary(report: dict) -> dict:
    """Allowlist the aggregate fields appropriate for a research summary."""
    if report["test_evaluated"] or any(run["test"] is not None for run in report["runs"].values()):
        raise ValueError("Validation summaries must not include test evaluations.")
    baseline = report["runs"]["baseline"]
    return {
        "experiment_id": report["experiment_id"],
        "selection_metric": report["selection_metric"],
        "selected_feature_config": report["selected_feature_config"],
        "selected_model": report["selected_model"],
        "test_evaluated": False,
        "dataset": {
            key: value
            for key, value in baseline["dataset"].items()
            if key
            in {
                "source",
                "fingerprint_sha256",
                "customers",
                "reading_days",
                "date_start",
                "date_end",
                "missing_calendar_days",
                "positive_customers",
                "positive_rate",
                "missing_reading_rate",
                "requested_sample_size",
                "sampling",
            }
        },
        "seed": baseline["methodology"]["seed"],
        "fitted_on": baseline["methodology"]["fitted_on"],
        "split_sizes": baseline["split_sizes"],
        "partition_fingerprints": baseline["partition_fingerprints"],
        "software": baseline["software"],
        "candidate_parameters": baseline["candidate_parameters"],
        "runs": {
            config: {
                key: run[key]
                for key in (
                    "experiment_id",
                    "feature_columns",
                    "preprocessing",
                    "selected_model",
                    "validation",
                )
            }
            for config, run in report["runs"].items()
        },
    }
