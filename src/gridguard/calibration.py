"""Opt-in, development-only calibration experiments; never score the old holdout."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

try:  # Keep the existing sklearn>=1.4 contract.
    from sklearn.frozen import FrozenEstimator
except ImportError:  # sklearn 1.4/1.5: supported prefit API, no estimator refit.
    FrozenEstimator = None

from gridguard.data import ConsumptionData
from gridguard.features import FEATURE_CONFIGS, build_features
from gridguard.modeling import ranking_metrics, ranking_order
from gridguard.provenance import (
    experiment_metadata,
    file_fingerprint,
    load_reference_development,
    partition_fingerprint,
    verify_partition_manifest,
    write_json,
)
from gridguard.reliability import (
    CALIBRATION_PARTITIONS,
    ECE_NOTICE,
    reliability_metrics,
    validate_bins,
)
from gridguard.training import _fit_candidates

MIN_SIGMOID_SAMPLES = 20
MIN_ISOTONIC_SAMPLES = 1000


def split_development(data: ConsumptionData, *, seed=42) -> dict[str, np.ndarray]:
    """Five fixed stratified roles, using only an already-excluded development pool.

    Fractions are approximately 50/10/20/10/10, subject to integer rounding.
    Minimum sizes are operability guards, not statistical validity guarantees.
    """
    if data.labels is None:
        raise ValueError("Calibration development requires binary labels.")
    ranking_metrics(data.labels, np.zeros(len(data)), 1, customer_ids=data.customer_ids)
    if len(data.readings) != len(data):
        raise ValueError("Customer IDs and readings must be aligned.")
    y = data.labels.to_numpy(dtype=int)
    if len(y) < 100 or min(int(y.sum()), int((1 - y).sum())) < 20:
        raise ValueError("Calibration development requires >=100 customers and >=20 per class.")

    def divide(indices, fraction):
        return train_test_split(indices, test_size=fraction, stratify=y[indices], random_state=seed)

    try:
        training, remaining = divide(np.arange(len(data)), 0.5)
        selection_pool, fitting = divide(remaining, 0.4)
        final_pool, model_selection = divide(selection_pool, 1 / 3)
        calibration_selection, evaluation = divide(final_pool, 0.5)
    except ValueError as exc:
        raise ValueError(
            "Insufficient development data for five stratified customer roles."
        ) from exc
    return {
        "train": training,
        "model_selection": model_selection,
        "calibration_fit": fitting,
        "calibration_selection": calibration_selection,
        "development_evaluation": evaluation,
    }


def fit_calibrators(
    model, features, labels, *, calibration_ids, training_ids, evaluation_ids
) -> tuple[dict, dict]:
    """Fit mappings on disjoint calibration customers around a frozen base model.

    evaluation_ids includes all model/mapping selection and diagnostic customers.
    The caller must supply the actual base-training membership. The workflow
    additionally verifies the complete recorded manifest before any fitting.
    """
    ids = [*training_ids, *calibration_ids, *evaluation_ids]
    ranking_order(np.zeros(len(ids)), ids)  # Reject normalized duplicates/overlap.
    ranking_metrics(labels, np.zeros(len(calibration_ids)), 1, customer_ids=calibration_ids)
    if len(features) != len(calibration_ids):
        raise ValueError("Calibration features, labels and customer IDs must be aligned.")
    y = np.asarray(labels, dtype=int)
    fitted, skipped = {}, {}
    for method, minimum in (("sigmoid", MIN_SIGMOID_SAMPLES), ("isotonic", MIN_ISOTONIC_SAMPLES)):
        if len(y) < minimum or min(int(y.sum()), int((1 - y).sum())) < 2:
            skipped[method] = f"Requires >={minimum} fitting customers and >=2 per class."
            continue
        if method == "isotonic":
            responses = (
                model.decision_function(features)
                if hasattr(model, "decision_function")
                else model.predict_proba(features)[:, 1]
            )
            if len(np.unique(responses)) < 2:
                skipped[method] = "Isotonic requires at least two distinct base-model responses."
                continue
        if FrozenEstimator is None:
            calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
        else:
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrated.fit(features, y)
        fitted[method] = calibrated
    return fitted, skipped


def calibrate(
    input_path: str | Path,
    *,
    reference_dir: str | Path,
    output_dir: str | Path,
    seed=42,
    feature_config="baseline",
    n_bins=10,
    experiment_id: str | None = None,
) -> dict:
    """Fresh opt-in experiment; historical model/results stay untouched.

    A schema-v2 reference manifest identifies the previously reserved holdout.
    Only reference train+validation customers enter new development partitions.
    New base models fit train only; base selection uses model_selection AP;
    calibration fits calibration_fit only; method selection uses the minimum
    calibration_selection Brier (ties: uncalibrated, sigmoid, isotonic). No refit
    follows selection. Diagnostics are computed after this choice is locked.
    """
    validate_bins(n_bins)
    n_bins = int(n_bins)
    if feature_config not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature configuration: {feature_config}")
    reference_dir, output_dir = Path(reference_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty calibration output directory: {output_dir}")
    development, parent, original_parts = load_reference_development(input_path, reference_dir)
    partitions = split_development(development, seed=seed)
    report = experiment_metadata(
        development,
        partitions,
        input_path=input_path,
        sample_size=None,
        source=parent["dataset"]["source"],
        seed=seed,
        experiment_id=experiment_id or output_dir.name,
        feature_config=feature_config,
    )
    report["dataset"]["sampling"] = "reference train+validation membership; no new sampling"
    report["dataset"]["reference_cohort_customers"] = parent["dataset"]["customers"]
    private_parts = {
        name: development.customer_ids.iloc[indices].tolist()
        for name, indices in partitions.items()
    }
    private_parts["excluded_holdout"] = original_parts["test"]
    report["split_sizes"]["excluded_holdout"] = len(original_parts["test"])
    report["partition_fingerprints"]["excluded_holdout"] = partition_fingerprint(
        original_parts["test"]
    )
    new_manifest = {
        "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
        "partitions": private_parts,
    }
    verify_partition_manifest(report, new_manifest, names=CALIBRATION_PARTITIONS)
    chosen, model_name, columns, base_selection, _ = _fit_candidates(
        development,
        {"train": partitions["train"], "validation": partitions["model_selection"]},
        config=feature_config,
        seed=seed,
    )
    fitting = development.subset(partitions["calibration_fit"])
    calibrated, skipped = fit_calibrators(
        chosen,
        build_features(fitting, config=feature_config),
        fitting.labels,
        calibration_ids=fitting.customer_ids,
        training_ids=private_parts["train"],
        evaluation_ids=[
            customer
            for name in ("model_selection", "calibration_selection", "development_evaluation")
            for customer in private_parts[name]
        ],
    )
    models = {"uncalibrated": chosen, **calibrated}
    selection = development.subset(partitions["calibration_selection"])
    selection_features = build_features(selection, config=feature_config)
    selection_metrics = {
        method: reliability_metrics(
            selection.labels,
            model.predict_proba(selection_features)[:, 1],
            customer_ids=selection.customer_ids,
            n_bins=n_bins,
        )
        for method, model in models.items()
    }
    selected_method = min(models, key=lambda method: selection_metrics[method]["brier_score"])
    # Diagnostic predictions and metrics are computed only after locking the method.
    diagnostics = development.subset(partitions["development_evaluation"])
    diagnostic_features = build_features(diagnostics, config=feature_config)
    predictions = pd.DataFrame(
        {
            "customer_id": diagnostics.customer_ids,
            "true_label": diagnostics.labels,
        }
    )
    method_results = {}
    for method, model in models.items():
        # Some sklearn estimators return float32. Expand before CSV serialization
        # so saved diagnostics retain the exact values used by float64 metrics.
        predictions[method] = np.asarray(
            model.predict_proba(diagnostic_features)[:, 1], dtype=np.float64
        )
        method_results[method] = {
            "fitted_on": "train" if method == "uncalibrated" else "calibration_fit (base frozen)",
            "selection": selection_metrics[method],
            "development_evaluation": reliability_metrics(
                diagnostics.labels,
                predictions[method],
                customer_ids=diagnostics.customer_ids,
                n_bins=n_bins,
            ),
        }
    report.update(
        {
            "artifact_type": "calibration_reliability",
            "evaluation_scope": "development_only",
            "independent_evaluation": None,
            "selected_model": model_name,
            "selected_method": selected_method,
            "base_selection": base_selection,
            "methods": method_results,
            "skipped_methods": skipped,
            "binning": {"strategy": "uniform", "n_bins": n_bins, "notice": ECE_NOTICE},
            "calibration_minimums": {
                "sigmoid": MIN_SIGMOID_SAMPLES,
                "isotonic": MIN_ISOTONIC_SAMPLES,
            },
            "source_reference": {
                "experiment_id": parent["experiment_id"],
                "metrics_sha256": file_fingerprint(reference_dir / "metrics.json"),
                "manifest_sha256": file_fingerprint(reference_dir / "split_manifest.json"),
            },
        }
    )
    report["methodology"].update(
        {
            "split": "reference train+validation only; stratified 50/10/20/10/10 development roles",
            "fitted_on": "base/preprocessing: train only; mappings: calibration_fit only; no refit",
            "selection_metric": "base: model_selection AP; mapping: calibration_selection Brier",
            "method_tie_order": ["uncalibrated", "sigmoid", "isotonic"],
            "test_usage": "previously inspected holdout excluded; no features, predictions or evaluation",
            "risk_score": "existing published scores unchanged; optional new bundle uses selected mapping",
            "log_loss_clipping": f"sklearn float64 machine epsilon: {np.finfo(np.float64).eps}",
            "independent_final_performance_established": False,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_manifest.json", new_manifest)
    predictions.to_csv(output_dir / "calibration_predictions.csv", index=False)
    bundle = {
        "model": models[selected_method],
        "uncalibrated_model": chosen,
        "model_name": model_name,
        "calibration_method": selected_method,
        "feature_columns": columns,
        "feature_config": feature_config,
        "seed": seed,
        "software": report["software"],
        "experiment_id": report["experiment_id"],
        "fitted_on": report["methodology"]["fitted_on"],
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    for filename, key in (
        ("model.joblib", "model_sha256"),
        ("calibration_predictions.csv", "predictions_sha256"),
        ("split_manifest.json", "manifest_sha256"),
    ):
        report[key] = file_fingerprint(output_dir / filename)
    write_json(output_dir / "reliability.json", report)
    return report
