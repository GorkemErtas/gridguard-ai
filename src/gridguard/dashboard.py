"""Read-only dashboard helpers. Never load estimators or fit a model here."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.anomaly_metrics import ANOMALY_PARTITIONS, benchmark_summary
from gridguard.evaluation import _verify_provenance
from gridguard.features import risk_band
from gridguard.modeling import model_metrics, ranking_order
from gridguard.provenance import file_fingerprint, verify_partition_manifest
from gridguard.reliability import (
    CALIBRATION_PARTITIONS,
    METHODS,
    reliability_metrics,
    summaries_match,
)


@dataclass
class DashboardRun:
    report: dict
    predictions: pd.DataFrame
    kind: str
    provenance: str
    warnings: list[str]


def discover_runs(root: str | Path = "artifacts") -> list[str]:
    """Bound discovery to the existing experiment/config directory convention."""
    root = Path(root)
    return sorted(
        str(path.parent)
        for pattern in ("metrics.json", "*/metrics.json", "*/*/metrics.json")
        for path in root.glob(pattern)
    )


def file_signature(
    folder: str | Path, names=("metrics.json", "test_predictions.csv", "split_manifest.json")
) -> tuple:
    """Cache invalidation without reading model weights or raw consumption."""
    root = Path(folder)
    return tuple(
        (name, (root / name).stat().st_mtime_ns, (root / name).stat().st_size)
        for name in names
        if (root / name).is_file()
    )


def optional_report(path: str | Path) -> tuple[dict, str | None]:
    path = Path(path)
    if not path.is_file():
        return {}, f"Optional report unavailable: {path.name}"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result, None
    except (OSError, ValueError) as exc:
        return {}, f"Could not read {path.name}: {exc}"


def load_run(folder: str | Path) -> DashboardRun:
    folder = Path(folder)
    report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("validation", {}), dict):
        raise ValueError("Invalid metrics report")
    synthetic = report.get("dataset", {}).get("source") == "synthetic"
    phase = "Validation-only" if report.get("test") is None else "Final holdout"
    kind = f"{'Synthetic' if synthetic else 'Consumption data'} · {phase}"
    if report.get("test") is None:
        return DashboardRun(report, pd.DataFrame(), kind, "Test predictions not produced", [])
    predictions = pd.read_csv(
        folder / "test_predictions.csv",
        dtype={"customer_id": "string"},
        float_precision="round_trip",
    )
    if not {"customer_id", "model_score"}.issubset(predictions):
        raise ValueError("Predictions require customer_id and raw model_score")
    predictions["customer_id"] = predictions.customer_id.str.strip()
    order = ranking_order(predictions.model_score, predictions.customer_id)
    if predictions.empty:
        raise ValueError("Prediction file is empty")
    expected = report.get("split_sizes", {}).get("test")
    if expected is not None and len(predictions) != expected:
        raise ValueError("Prediction count differs from saved test split")
    if "true_label" in predictions:
        if predictions.true_label.isna().any() or not predictions.true_label.isin([0, 1]).all():
            raise ValueError("Invalid prediction labels")
    warnings = []
    if report.get("schema_version", 1) >= 2 and not (folder / "split_manifest.json").is_file():
        if report.get("test_predictions_sha256") != file_fingerprint(
            folder / "test_predictions.csv"
        ):
            raise ValueError("Holdout prediction fingerprint mismatch")
        provenance = "Unverified: optional local split manifest unavailable"
        warnings.append(provenance)
    else:
        provenance = _verify_provenance(folder, report, predictions)
    predictions = predictions.iloc[order].reset_index(drop=True)
    # Display bands always agree with the raw score, including older exports.
    predictions["risk_score"] = np.rint(predictions.model_score * 100).astype(int)
    predictions["risk_band"] = predictions.risk_score.map(risk_band)
    predictions["rank"] = np.arange(1, len(predictions) + 1)
    return DashboardRun(report, predictions, kind, provenance, warnings)


def _development_artifacts(path, *, artifact_type, partition_names, prediction_name):
    """Shared read-only validation for saved development evaluation artifacts."""
    path = Path(path)
    report, warning = optional_report(path)
    if warning:
        raise ValueError(warning)
    if report.get("artifact_type") != artifact_type or report.get("schema_version") != 2:
        raise ValueError("Unsupported saved development evaluation report.")
    if not isinstance(report.get("methodology"), dict):
        raise ValueError("Invalid evaluation methodology metadata.")
    if (
        report.get("evaluation_scope") != "development_only"
        or report.get("independent_evaluation") is not None
        or report["methodology"].get("independent_final_performance_established") is not False
    ):
        raise ValueError(
            "Expected development-only diagnostics; independent final results are not established."
        )
    for filename, key in (
        ("split_manifest.json", "manifest_sha256"),
        (prediction_name, "predictions_sha256"),
    ):
        if file_fingerprint(path.parent / filename) != report.get(key):
            raise ValueError(f"Evaluation artifact fingerprint mismatch: {filename}")
    manifest, warning = optional_report(path.parent / "split_manifest.json")
    if warning:
        raise ValueError(warning)
    partitions = verify_partition_manifest(report, manifest, names=partition_names)
    predictions = pd.read_csv(
        path.parent / prediction_name, dtype={"customer_id": "string"}, float_precision="round_trip"
    )
    if "customer_id" not in predictions or predictions.empty:
        raise ValueError("Missing evaluation prediction customers.")
    predictions["customer_id"] = predictions.customer_id.str.strip()
    return report, partitions, predictions


def load_reliability(path: str | Path) -> dict:
    """Verify saved calibration summaries without opening an estimator bundle."""
    report, partitions, predictions = _development_artifacts(
        path,
        artifact_type="calibration_reliability",
        partition_names=CALIBRATION_PARTITIONS,
        prediction_name="calibration_predictions.csv",
    )
    methods = report["methods"]
    if (
        not isinstance(methods, dict)
        or "uncalibrated" not in methods
        or not set(methods) <= set(METHODS)
        or report["selected_method"] not in methods
        or report["binning"]["strategy"] != "uniform"
    ):
        raise ValueError("Invalid calibration methods or binning metadata.")
    for method in methods:
        selection = methods[method]["selection"]
        loss = selection["brier_score"]
        if (
            not isinstance(loss, (int, float))
            or not np.isfinite(loss)
            or not 0 <= loss <= 1
            or selection["customers"] != report["split_sizes"]["calibration_selection"]
        ):
            raise ValueError("Invalid calibration-selection summary.")
    expected = min(
        (method for method in METHODS if method in methods),
        key=lambda method: methods[method]["selection"]["brier_score"],
    )
    if report["selected_method"] != expected:
        raise ValueError("Saved mapping differs from the calibration-selection Brier winner.")
    required = {"customer_id", "true_label", *methods}
    if not required.issubset(predictions) or predictions.empty:
        raise ValueError("Missing calibration diagnostic prediction columns or customers.")
    if set(predictions.customer_id) != set(partitions["development_evaluation"]):
        raise ValueError(
            "Calibration diagnostic customers differ from recorded evaluation membership."
        )
    for method in methods:
        computed = reliability_metrics(
            predictions.true_label,
            predictions[method],
            customer_ids=predictions.customer_id,
            n_bins=report["binning"]["n_bins"],
        )
        if not summaries_match(methods[method]["development_evaluation"], computed):
            raise ValueError("Saved reliability metrics differ from the diagnostic predictions.")
    return report


def load_anomaly_benchmark(path: str | Path) -> dict:
    """Return aggregate, verified benchmark data; never return customer records."""
    report, partitions, predictions = _development_artifacts(
        path,
        artifact_type="anomaly_benchmark",
        partition_names=ANOMALY_PARTITIONS,
        prediction_name="anomaly_predictions.csv",
    )
    if report.get("benchmark_version") != 1:
        raise ValueError("Unsupported anomaly benchmark version.")
    names = {"development_validation", "development_evaluation"}
    if (
        "partition" not in predictions
        or set(predictions.partition) != names
        or set(report["evaluations"]) != names
    ):
        raise ValueError("Anomaly prediction partition roles differ from the evaluation protocol.")
    for name in sorted(names):
        frame = predictions.loc[predictions.partition.eq(name)]
        if set(frame.customer_id) != set(partitions[name]):
            raise ValueError(
                "Anomaly prediction customers differ from recorded evaluation membership."
            )
        ranking_order(frame["supervised_rf"], frame.customer_id)
        computed = benchmark_summary(
            frame,
            capacities=report["configuration"]["capacities"],
            min_group_size=report["configuration"]["min_group_size"],
        )
        if not summaries_match(report["evaluations"][name], computed):
            raise ValueError("Saved anomaly summaries differ from the evaluation predictions.")
    return report


def filter_queue(predictions: pd.DataFrame, *, capacity: int, bands: list[str], search=""):
    """Filter eligibility first, then fill capacity in exact global score order."""
    if capacity < 1:
        raise ValueError("Capacity must be positive")
    eligible = predictions.loc[predictions.risk_band.isin(bands)]
    if search.strip():
        eligible = eligible.loc[
            eligible.customer_id.str.contains(search.strip(), case=False, regex=False)
        ]
    return eligible.head(capacity).copy()


def queue_metrics(queue: pd.DataFrame, population: pd.DataFrame) -> dict:
    result = {"customers": len(queue), "precision": None, "recall": None, "lift": None}
    if queue.empty or "true_label" not in queue or "true_label" not in population:
        return result
    found, positives = int(queue.true_label.sum()), int(population.true_label.sum())
    precision = float(queue.true_label.mean())
    prevalence = float(population.true_label.mean())
    return {
        **result,
        "precision": precision,
        "recall": found / positives if positives else None,
        "lift": precision / prevalence if prevalence else None,
    }


def prediction_metrics(predictions: pd.DataFrame) -> dict:
    if predictions.empty or "true_label" not in predictions:
        return {}
    return model_metrics(
        predictions.true_label.to_numpy(),
        predictions.model_score.to_numpy(),
        customer_ids=predictions.customer_id,
    )


def private_export(queue: pd.DataFrame, *, include_ids=False) -> bytes:
    """Explicit allowlist: never export labels, signals or arbitrary extra columns."""
    columns = ["rank", "risk_score", "risk_band", "model_score"]
    if include_ids:
        columns.insert(1, "customer_id")
    export = queue[columns].copy()
    if include_ids:
        # Protect spreadsheet users from CSV formula injection via external IDs.
        export["customer_id"] = export.customer_id.map(
            lambda value: (
                "'" + str(value)
                if str(value).lstrip().startswith(("=", "+", "-", "@"))
                else str(value)
            )
        )
    return export.to_csv(index=False).encode("utf-8")


def quality_groups(analysis: dict, feature: str) -> pd.DataFrame:
    groups = analysis.get("data_quality_groups", {}).get(feature, {}).get("groups", {})
    return pd.DataFrame(
        [
            {
                "group": name,
                **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
            }
            for name, metrics in groups.items()
        ]
    )
