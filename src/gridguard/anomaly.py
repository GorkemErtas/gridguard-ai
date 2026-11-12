"""Fixed development-only RF/Isolation Forest/robust-MAD benchmark."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from gridguard.anomaly_metrics import (
    ANOMALY_PARTITIONS,
    benchmark_summary,
    inspection_capacities,
)
from gridguard.error_analysis import QUALITY_BINS
from gridguard.features import FEATURE_CONFIGS, build_features
from gridguard.modeling import candidate_models
from gridguard.provenance import (
    experiment_metadata,
    file_fingerprint,
    load_reference_development,
    partition_fingerprint,
    verify_partition_manifest,
    write_json,
)


class RobustMAD(BaseEstimator):
    """Mean log1p absolute robust z-deviation; larger means more unusual.

    Training medians/MADs only. MAD floor is relative to max(1, abs(median));
    z values are capped before averaging logs to prevent overflow/domination.
    All-missing training columns can be excluded using observed_features.
    Missing features must first pass through the shared training-only imputer.
    """

    def __init__(self, mad_floor=1e-9, deviation_cap=1e6):
        self.mad_floor = mad_floor
        self.deviation_cap = deviation_cap

    def _matrix(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("Robust MAD requires finite, imputed two-dimensional features.")
        return values

    def fit(self, X, y=None, *, observed_features=None):
        values = self._matrix(X)
        if not len(values):
            raise ValueError("Robust MAD requires nonempty training customers.")
        if (
            not np.isfinite(self.mad_floor)
            or self.mad_floor <= 0
            or not np.isfinite(self.deviation_cap)
            or self.deviation_cap <= 0
        ):
            raise ValueError("MAD floor and deviation cap must be finite and positive.")
        self.n_features_in_ = values.shape[1]
        self.active_features_ = (
            np.ones(values.shape[1], dtype=bool)
            if observed_features is None
            else np.asarray(observed_features, dtype=bool)
        )
        if self.active_features_.shape != (values.shape[1],):
            raise ValueError("Observed-feature mask must align with training columns.")
        # Halving before median's midpoint averaging avoids overflow near float64 limits.
        self.center_ = np.median(values / 2, axis=0) * 2
        self.unit_ = np.maximum(1, np.abs(self.center_))
        deviations = np.abs(values / self.unit_ - self.center_ / self.unit_)
        self.mad_ = np.median(deviations / 2, axis=0) * 2
        return self

    def score_samples(self, X):
        check_is_fitted(self)
        values = self._matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature columns differ from the fitted robust baseline.")
        if not self.active_features_.any():
            return np.zeros(len(values))
        with np.errstate(over="ignore", divide="ignore"):
            deviations = np.abs(values / self.unit_ - self.center_ / self.unit_)
            z = (deviations / np.maximum(self.mad_, self.mad_floor)) / 1.4826
        z = np.minimum(z, self.deviation_cap)
        return np.log1p(z[:, self.active_features_]).mean(axis=1)


def benchmark(
    input_path,
    *,
    reference_dir,
    output_dir,
    seed=42,
    feature_config="baseline",
    capacities=(50, 100, 200),
    min_group_size=20,
    experiment_id=None,
):
    """All methods fit exactly the reference's original training customers.

    Reference validation is split into development validation/evaluation. All
    configurations are fixed before fitting; neither cohort selects a method,
    parameter, feature or threshold. No historical estimator or holdout score
    is loaded. The RF is a fresh fixed comparator, not a published model result.
    """
    capacities = inspection_capacities(capacities)
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
        raise FileExistsError(f"Choose an empty anomaly output directory: {output_dir}")
    data, parent, original = load_reference_development(input_path, reference_dir)
    training = np.arange(len(original["train"]))
    validation_pool = np.arange(len(training), len(data))
    if set(data.labels.iloc[training]) != {0, 1}:
        raise ValueError("The supervised training comparator requires both label classes.")
    labels = data.labels.iloc[validation_pool].to_numpy(dtype=int)
    if len(labels) < 4 or min(int(labels.sum()), int((1 - labels).sum())) < 2:
        raise ValueError("Reference validation requires >=4 customers and >=2 per class.")
    validation, evaluation = train_test_split(
        validation_pool,
        test_size=0.5,
        stratify=labels,
        random_state=seed,
    )
    parts = {
        "train": training,
        "development_validation": validation,
        "development_evaluation": evaluation,
    }
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
    private_parts = {key: data.customer_ids.iloc[idx].tolist() for key, idx in parts.items()}
    private_parts["excluded_holdout"] = original["test"]
    report["split_sizes"]["excluded_holdout"] = len(original["test"])
    report["partition_fingerprints"]["excluded_holdout"] = partition_fingerprint(original["test"])
    manifest = {
        "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
        "partitions": private_parts,
    }
    verify_partition_manifest(report, manifest, names=ANOMALY_PARTITIONS)

    train_data = data.subset(training)
    features = build_features(train_data, config=feature_config)
    # Use the exact existing RF recipe. Its fitted median imputer is shared by
    # both anomaly detectors; no second preprocessing recipe or refit is added.
    supervised = candidate_models(seed)["random_forest"]
    supervised.named_steps["classifier"].set_params(n_jobs=1)
    supervised.fit(features, train_data.labels.to_numpy(dtype=int))
    imputer = supervised.named_steps["imputer"]
    imputed = imputer.transform(features)
    isolation = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=seed,
        n_jobs=1,
    ).fit(imputed)
    robust = RobustMAD().fit(imputed, observed_features=features.notna().any(axis=0).to_numpy())

    predictions = []
    evaluations = {}
    for name, indices in (
        ("development_validation", validation),
        ("development_evaluation", evaluation),
    ):
        cohort = data.subset(indices)
        frame = build_features(cohort, config=feature_config)
        transformed = imputer.transform(frame)
        scored = frame[["missing_ratio"]].copy()
        scored["customer_id"] = cohort.customer_ids.to_numpy()
        scored["true_label"] = cohort.labels.to_numpy(dtype=int)
        scored["partition"] = name
        scored["supervised_rf"] = np.asarray(supervised.predict_proba(frame)[:, 1], dtype=float)
        scored["isolation_forest"] = np.asarray(-isolation.score_samples(transformed), dtype=float)
        scored["robust_mad"] = robust.score_samples(transformed)
        # Expand float32 quality features too, preserving recomputed bin membership.
        scored["missing_ratio"] = scored.missing_ratio.astype(float)
        evaluations[name] = benchmark_summary(
            scored, capacities=capacities, min_group_size=min_group_size
        )
        predictions.append(scored)
    report.update(
        {
            "artifact_type": "anomaly_benchmark",
            "benchmark_version": 1,
            "evaluation_scope": "development_only",
            "independent_evaluation": None,
            "configuration": {
                "capacities": capacities,
                "min_group_size": int(min_group_size),
                "histogram_bins": 20,
                "missing_ratio_bins": [
                    None if np.isinf(edge) else edge for edge in QUALITY_BINS["missing_ratio"]
                ],
                "group_min_class_support": 5,
            },
            "feature_columns": features.columns.tolist(),
            "evaluations": evaluations,
            "model_parameters": {
                "supervised_rf": supervised.named_steps["classifier"].get_params(deep=False),
                "isolation_forest": isolation.get_params(deep=False),
                "robust_mad": robust.get_params(deep=False),
            },
            "score_definitions": {
                "supervised_rf": "Fresh development RF predict_proba[:,1]; uncalibrated ranking signal.",
                "isolation_forest": "-score_samples(shared train-imputed features); higher means more anomalous, not probability.",
                "robust_mad": "Mean log1p capped absolute robust z across train-observed features; higher is more anomalous, not probability.",
            },
            "all_missing_training_features": features.columns[~robust.active_features_].tolist(),
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
            "split": "original train retained; original validation split 50/50; original test excluded",
            "fitted_on": "original train only, including shared median imputation and robust statistics",
            "selection_metric": "none; fixed configurations, no method/threshold/feature selection",
            "test_usage": "previously inspected final holdout excluded; no features or scores extracted",
            "independent_final_performance_established": False,
            "supervised_comparator": "fresh predeclared Random Forest, not the historical published estimator",
            "calibration": "none; no calibration customers or calibration mappings in this benchmark",
            "ranking_ties": "native score descending, customer_id ascending",
            "risk_score": "no risk-score replacement; anomaly scores have native units",
            "contamination": "auto affects IF decision offset; benchmark uses -score_samples, never threshold labels",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_manifest.json", manifest)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "anomaly_predictions.csv", index=False
    )
    joblib.dump(
        {
            "imputer": imputer,
            "supervised_rf": supervised,
            "isolation_forest": isolation,
            "robust_mad": robust,
            "feature_columns": features.columns.tolist(),
            "feature_config": feature_config,
            "seed": seed,
        },
        output_dir / "anomaly_models.joblib",
    )
    for filename, key in (
        ("split_manifest.json", "manifest_sha256"),
        ("anomaly_predictions.csv", "predictions_sha256"),
        ("anomaly_models.joblib", "model_sha256"),
    ):
        report[key] = file_fingerprint(output_dir / filename)
    write_json(output_dir / "anomaly_benchmark.json", report)
    return report
