"""Offline, development-only permutation importance and supervised error analysis."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

from gridguard.anomaly_metrics import inspection_capacities
from gridguard.explainability_metrics import (
    EXPLAINABILITY_PARTITIONS,
    analysis_thresholds,
    error_diagnostics,
    importance_summary,
)
from gridguard.features import FEATURE_CONFIGS, build_features
from gridguard.modeling import ranking_order, score_metrics
from gridguard.provenance import (
    experiment_metadata,
    file_fingerprint,
    load_reference_development,
    partition_fingerprint,
    verify_partition_manifest,
    write_json,
)
from gridguard.training import development_partitions, fit_development_baseline


def permutation_diagnostics(
    model,
    features,
    labels,
    *,
    customer_ids,
    training_ids,
    excluded_ids,
    n_repeats=10,
    seed=42,
):
    """AP permutation sensitivity of a fitted, named-feature pipeline.

    Membership supplied by the verified workflow must exclude training, final
    holdout and the separate error cohort. Shuffling raw engineered columns
    passes through the existing frozen preprocessing; nothing is fitted here.
    """
    names = features.columns.tolist()
    expected = list(getattr(model, "feature_names_in_", []))
    if names != expected or not expected:
        raise ValueError("Original feature names/order differ from the fitted pipeline.")
    if len(features) != len(customer_ids):
        raise ValueError("Importance features and customer IDs must align.")
    ids = [*training_ids, *customer_ids, *excluded_ids]
    ranking_order(np.zeros(len(ids)), ids)
    score_metrics(labels, np.zeros(len(features)), customer_ids=customer_ids, capacities=())
    y = None if labels is None else np.asarray(labels, dtype=int)
    status = (
        "labels_unavailable"
        if y is None
        else "empty"
        if not len(y)
        else "single_class"
        if len(np.unique(y)) < 2
        else "available"
    )
    repeats = {
        "status": status,
        "feature_names": names,
        "n_repeats": n_repeats,
        "seed": int(seed),
        "customers": len(features),
        "positive_customers": int(y.sum()) if y is not None else None,
        "baseline_average_precision": None,
        "decreases": [],
    }
    # Validate configuration/names even when the cohort cannot support AP importance.
    importance_summary({**repeats, "status": "labels_unavailable"})
    if status == "available":
        baseline = float(get_scorer("average_precision")(model, features, y))
        result = permutation_importance(
            model,
            features,
            y,
            scoring="average_precision",
            n_repeats=int(n_repeats),
            random_state=seed,
            n_jobs=1,
            max_samples=1.0,
        )
        repeats.update(baseline_average_precision=baseline, decreases=result.importances.tolist())
    return importance_summary(repeats), repeats


def explain(
    input_path,
    *,
    reference_dir,
    output_dir,
    seed=42,
    feature_config="baseline",
    n_repeats=10,
    thresholds=(0.2, 0.5, 0.8),
    capacities=(50, 100, 200),
    min_group_size=20,
    experiment_id=None,
):
    """Fit a fresh fixed RF on original train, explain/evaluate on separate roles.

    Historical models may have been selected using reference validation labels.
    They are never loaded. No importance/threshold/error result selects a model,
    feature, hyperparameter, threshold or calibrator. Final holdout is excluded.
    """
    thresholds, capacities = analysis_thresholds(thresholds), inspection_capacities(capacities)
    if (
        isinstance(n_repeats, (bool, np.bool_))
        or not isinstance(n_repeats, (int, np.integer))
        or n_repeats < 1
    ):
        raise ValueError("n_repeats must be a positive integer.")
    if (
        isinstance(min_group_size, (bool, np.bool_))
        or not isinstance(min_group_size, (int, np.integer))
        or min_group_size < 1
    ):
        raise ValueError("min_group_size must be a positive integer.")
    if feature_config not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature configuration: {feature_config}")
    reference_dir, output_dir = Path(reference_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty explainability output directory: {output_dir}")
    data, parent, original = load_reference_development(input_path, reference_dir)
    parts = development_partitions(data, original, seed=seed)
    parts["development_importance"] = parts.pop("development_validation")
    train = data.subset(parts["train"])
    train_features = build_features(train, config=feature_config)
    report = experiment_metadata(
        data,
        parts,
        input_path=input_path,
        sample_size=None,
        source=parent["dataset"]["source"],
        seed=seed,
        experiment_id=experiment_id or output_dir.name,
        feature_config=feature_config,
    )
    membership = {name: data.customer_ids.iloc[indices].tolist() for name, indices in parts.items()}
    membership["excluded_holdout"] = original["test"]
    report["split_sizes"]["excluded_holdout"] = len(original["test"])
    report["partition_fingerprints"]["excluded_holdout"] = partition_fingerprint(original["test"])
    manifest = {
        "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
        "partitions": membership,
    }
    verify_partition_manifest(report, manifest, names=EXPLAINABILITY_PARTITIONS)
    model = fit_development_baseline(train_features, train.labels.to_numpy(dtype=int), seed=seed)
    frames = {}
    importance = repeats = None
    for name in ("development_importance", "development_evaluation"):
        cohort = data.subset(parts[name])
        features = build_features(cohort, config=feature_config)
        if name == "development_importance":
            importance, repeats = permutation_diagnostics(
                model,
                features,
                cohort.labels,
                customer_ids=cohort.customer_ids,
                training_ids=membership["train"],
                excluded_ids=membership["excluded_holdout"] + membership["development_evaluation"],
                n_repeats=int(n_repeats),
                seed=seed,
            )
        frame = features.astype(float).copy()
        frame["customer_id"] = cohort.customer_ids.to_numpy()
        frame["true_label"] = cohort.labels.to_numpy(dtype=int)
        frame["model_score"] = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
        frame["partition"] = name
        frames[name] = frame
    diagnostics = error_diagnostics(
        frames["development_evaluation"],
        train_features.columns.tolist(),
        thresholds=thresholds,
        capacities=capacities,
        min_group_size=min_group_size,
    )
    report.update(
        {
            "artifact_type": "model_explainability",
            "explainability_version": 1,
            "evaluation_scope": "development_only",
            "independent_evaluation": None,
            "selected_model": "fresh_fixed_random_forest",
            "feature_columns": train_features.columns.tolist(),
            "model_parameters": model.named_steps["classifier"].get_params(deep=False),
            "configuration": {
                "thresholds": thresholds,
                "capacities": capacities,
                "n_repeats": int(n_repeats),
                "min_group_size": int(min_group_size),
                "histogram_bins": 20,
                "primary_group_threshold": 0.5,
            },
            "importance": importance,
            "errors": diagnostics,
            "source_reference": {
                "experiment_id": parent["experiment_id"],
                "metrics_sha256": file_fingerprint(reference_dir / "metrics.json"),
                "manifest_sha256": file_fingerprint(reference_dir / "split_manifest.json"),
            },
        }
    )
    report["dataset"]["sampling"] = "reference train+validation membership, no new sampling"
    report["methodology"].update(
        {
            "split": "original train retained; original validation split 50/50 into importance/error roles",
            "fitted_on": "original train only; existing train-fitted preprocessing reused without refit",
            "selection_metric": "none; fresh fixed RF, no outcome-driven model/feature/threshold selection",
            "test_usage": "inspected final holdout excluded; no features, predictions or analysis",
            "independent_final_performance_established": False,
            "importance_scope": "fresh development RF; not an explanation of the historical final model",
            "permutation": "AP decrease, raw engineered feature columns, all importance customers, one worker",
            "permutation_seed": int(seed),
            "repeat_std": "ddof=0; permutation variation, not a confidence interval",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_manifest.json", manifest)
    write_json(output_dir / "permutation_repeats.json", repeats)
    pd.concat(frames.values(), ignore_index=True).to_csv(
        output_dir / "explainability_predictions.csv", index=False
    )
    joblib.dump(
        {
            "model": model,
            "model_name": "random_forest",
            "feature_config": feature_config,
            "feature_columns": train_features.columns.tolist(),
            "seed": seed,
            "software": report["software"],
        },
        output_dir / "model.joblib",
    )
    for filename, key in (
        ("split_manifest.json", "manifest_sha256"),
        ("permutation_repeats.json", "repeats_sha256"),
        ("explainability_predictions.csv", "predictions_sha256"),
        ("model.joblib", "model_sha256"),
    ):
        report[key] = file_fingerprint(output_dir / filename)
    write_json(output_dir / "explainability.json", report)
    return report
