"""Training orchestration with validation-only selection and untouched test."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gridguard.data import load_consumption, sample_customers
from gridguard.features import build_features
from gridguard.modeling import candidate_models, model_metrics, rank_customers


def train(
    input_path: str | Path,
    *,
    output_dir: str | Path = "artifacts",
    sample_size: int | None = None,
    source: str = "custom",
    seed: int = 42,
) -> dict:
    data = load_consumption(input_path, require_labels=True)
    data = sample_customers(data, size=sample_size, seed=seed)
    assert data.labels is not None
    labels = data.labels.to_numpy(dtype=int)
    positives = int(labels.sum())
    if len(labels) < 80 or positives < 10 or len(labels) - positives < 10:
        raise ValueError("At least 80 customers with >=10 samples per class are needed.")

    features = build_features(data)
    indices = np.arange(len(data))
    development, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        development,
        test_size=0.25,  # 0.25 of 80% -> 20% validation.
        random_state=seed,
        stratify=labels[development],
    )

    models = candidate_models(seed)
    val_reports: dict[str, dict] = {}
    best_name = ""
    best_score = -1.0
    for name, model in models.items():
        model.fit(features.iloc[train_idx], labels[train_idx])
        probabilities = model.predict_proba(features.iloc[val_idx])[:, 1]
        val_reports[name] = model_metrics(labels[val_idx], probabilities)
        if name != "dummy_prior" and val_reports[name]["average_precision"] > best_score:
            best_score = val_reports[name]["average_precision"]
            best_name = name

    chosen = models[best_name]
    test_probabilities = chosen.predict_proba(features.iloc[test_idx])[:, 1]
    test_report = model_metrics(labels[test_idx], test_probabilities)
    ranking = rank_customers(
        data.customer_ids.iloc[test_idx].reset_index(drop=True),
        features.iloc[test_idx].reset_index(drop=True),
        test_probabilities,
    )
    ground_truth = dict(zip(data.customer_ids.iloc[test_idx].astype(str), labels[test_idx]))
    ranking["true_label"] = ranking["customer_id"].map(ground_truth).astype(int)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": chosen,
        "model_name": best_name,
        "feature_columns": features.columns.tolist(),
        "seed": seed,
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    ranking.to_csv(output_dir / "test_predictions.csv", index=False)

    report = {
        "dataset": {
            "source": source,
            "input_path": str(input_path),
            "customers": len(data),
            "reading_days": len(data.readings.columns),
            "positive_customers": positives,
            "positive_rate": float(labels.mean()),
            "missing_reading_rate": float(data.readings.isna().to_numpy().mean()),
        },
        "methodology": {
            "split": "customer-level stratified 60/20/20",
            "seed": seed,
            "selection_metric": "validation average_precision",
            "test_usage": "one-time holdout evaluation",
            "risk_score": "rounded 100 * uncalibrated predicted probability",
            "bands": "illustrative: high >=80, medium >=50, low <50",
        },
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "validation": val_reports,
        "selected_model": best_name,
        "test": test_report,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def score(
    input_path: str | Path, model_path: str | Path, output_path: str | Path
) -> pd.DataFrame:
    """Score fresh customers. Load joblib only from your own trusted model file."""
    bundle = joblib.load(model_path)
    data = load_consumption(input_path, require_labels=False)
    features = build_features(data)
    if features.columns.tolist() != bundle["feature_columns"]:
        raise ValueError("Model feature schema differs from the incoming data.")
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    ranked = rank_customers(data.customer_ids, features, probabilities)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)
    return ranked
