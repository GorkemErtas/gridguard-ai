"""Read-only dashboard helpers. Never load estimators or fit a model here."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.evaluation import _verify_provenance
from gridguard.features import risk_band
from gridguard.modeling import model_metrics, ranking_order
from gridguard.provenance import file_fingerprint


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


def file_signature(folder: str | Path) -> tuple:
    """Cache invalidation without reading model weights or raw consumption."""
    root = Path(folder)
    return tuple(
        (name, (root / name).stat().st_mtime_ns, (root / name).stat().st_size)
        for name in ("metrics.json", "test_predictions.csv", "split_manifest.json")
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
