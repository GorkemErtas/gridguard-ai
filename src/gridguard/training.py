"""Training orchestration with validation-only selection and an optional final test."""

import json
from pathlib import Path

import joblib
import pandas as pd

from gridguard.data import ConsumptionData, load_consumption, sample_customers, split_customers
from gridguard.error_analysis import aggregate_errors
from gridguard.features import FEATURE_CONFIGS, build_features
from gridguard.modeling import candidate_models, model_metrics, rank_customers
from gridguard.provenance import (
    experiment_metadata,
    file_fingerprint,
    partition_fingerprint,
    validation_summary,
    write_json,
)


def _write_holdout(data, partitions, chosen, *, config, report, output_dir):
    test = data.subset(partitions["test"])
    features = build_features(test, config=config)
    probabilities = chosen.predict_proba(features)[:, 1]
    report["test"] = model_metrics(
        test.labels.to_numpy(dtype=int), probabilities, customer_ids=test.customer_ids
    )
    ranking = rank_customers(test.customer_ids, features, probabilities)
    truth = dict(zip(test.customer_ids.astype(str), test.labels.to_numpy(dtype=int)))
    ranking["true_label"] = ranking["customer_id"].map(truth).astype(int)
    # Local quality columns allow aggregate analysis without reloading the raw CSV.
    quality = features[["missing_ratio", "coefficient_variation", "mean_kwh"]].copy()
    quality.index = test.customer_ids.astype(str).to_numpy()
    for column in quality:
        ranking[column] = ranking["customer_id"].map(quality[column])
    ranking.to_csv(output_dir / "test_predictions.csv", index=False)
    report["test_predictions_sha256"] = file_fingerprint(output_dir / "test_predictions.csv")


def _fit_candidates(data: ConsumptionData, partitions: dict, *, config: str, seed: int):
    # Test features and predictions are deliberately absent from this selection step.
    training = data.subset(partitions["train"])
    validation = data.subset(partitions["validation"])
    x_train = build_features(training, config=config)
    x_val = build_features(validation, config=config)
    models = candidate_models(seed)
    reports = {}
    for name, model in models.items():
        model.fit(x_train, training.labels.to_numpy(dtype=int))
        probabilities = model.predict_proba(x_val)[:, 1]
        reports[name] = model_metrics(
            validation.labels.to_numpy(dtype=int),
            probabilities,
            customer_ids=validation.customer_ids,
        )
    best = max(
        (name for name in models if name != "dummy_prior"),
        key=lambda name: reports[name]["average_precision"],
    )
    predictions = x_val[["missing_ratio", "coefficient_variation", "mean_kwh"]].copy()
    predictions["customer_id"] = validation.customer_ids.to_numpy()
    predictions["true_label"] = validation.labels.to_numpy()
    predictions["model_score"] = models[best].predict_proba(x_val)[:, 1]
    return models[best], best, x_train.columns.tolist(), reports, aggregate_errors(predictions)


def _run(
    data,
    partitions,
    *,
    input_path,
    output_dir,
    sample_size,
    source,
    seed,
    experiment_id,
    feature_config,
    validation_only,
):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Choose an empty output directory; refusing to overwrite {output_dir}"
        )
    if feature_config not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature configuration: {feature_config}")
    chosen, best_name, columns, validation, analysis = _fit_candidates(
        data, partitions, config=feature_config, seed=seed
    )
    report = experiment_metadata(
        data,
        partitions,
        input_path=input_path,
        sample_size=sample_size,
        source=source,
        seed=seed,
        experiment_id=experiment_id or output_dir.name,
        feature_config=feature_config,
    )
    report.update(
        {
            "feature_columns": columns,
            "candidate_parameters": {
                name: model.named_steps["classifier"].get_params(deep=False)
                for name, model in candidate_models(seed).items()
            },
            "selected_model": best_name,
            "validation": validation,
            "test": None,
        }
    )
    report["methodology"]["test_usage"] = (
        "not evaluated; validation-only experiment"
        if validation_only
        else "final holdout evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "validation_error_analysis.json", analysis)
    manifest = {
        "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
        "partitions": {
            key: data.customer_ids.iloc[idx].astype(str).tolist() for key, idx in partitions.items()
        },
    }
    write_json(output_dir / "split_manifest.json", manifest)
    if not validation_only:
        _write_holdout(
            data, partitions, chosen, config=feature_config, report=report, output_dir=output_dir
        )
    bundle = {
        "model": chosen,
        "model_name": best_name,
        "feature_columns": columns,
        "seed": seed,
        "feature_config": feature_config,
        "experiment_id": report["experiment_id"],
        "fitted_on": "train only",
        "software": report["software"],
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    report["model_sha256"] = file_fingerprint(output_dir / "model.joblib")
    write_json(output_dir / "metrics.json", report)
    return report


def train(
    input_path: str | Path,
    *,
    output_dir: str | Path = "artifacts",
    sample_size: int | None = None,
    source: str = "custom",
    seed: int = 42,
    feature_config: str = "baseline",
    experiment_id: str | None = None,
    validation_only: bool = False,
) -> dict:
    data = sample_customers(load_consumption(input_path), size=sample_size, seed=seed)
    return _run(
        data,
        split_customers(data, seed=seed),
        input_path=input_path,
        output_dir=output_dir,
        sample_size=sample_size,
        source=source,
        seed=seed,
        feature_config=feature_config,
        experiment_id=experiment_id,
        validation_only=validation_only,
    )


def experiment(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    sample_size: int | None = None,
    source: str = "custom",
    seed: int = 42,
    experiment_id: str | None = None,
) -> dict:
    """Four fixed ablations on identical customers; never predict/evaluate test."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty experiment directory: {output_dir}")
    data = sample_customers(load_consumption(input_path), size=sample_size, seed=seed)
    partitions = split_customers(data, seed=seed)
    identifier = experiment_id or output_dir.name
    runs = {}
    for config in FEATURE_CONFIGS:
        runs[config] = _run(
            data,
            partitions,
            input_path=input_path,
            output_dir=output_dir / config,
            sample_size=sample_size,
            source=source,
            seed=seed,
            feature_config=config,
            experiment_id=f"{identifier}-{config}",
            validation_only=True,
        )
    best = max(
        runs,
        key=lambda config: runs[config]["validation"][runs[config]["selected_model"]][
            "average_precision"
        ],
    )
    result = {
        "experiment_id": identifier,
        "selection_metric": "validation average_precision",
        "selected_feature_config": best,
        "selected_model": runs[best]["selected_model"],
        "test_evaluated": False,
        "runs": runs,
    }
    write_json(output_dir / "experiment.json", result)
    write_json(output_dir / "validation_summary.json", validation_summary(result))
    return result


def finalize(
    input_path: str | Path,
    *,
    experiment_dir: str | Path,
    output_dir: str | Path,
    experiment_id: str | None = None,
) -> dict:
    """Evaluate the validation winner once, using its frozen train-only estimator."""
    experiment_dir, output_dir = Path(experiment_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty final evaluation directory: {output_dir}")
    selection = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    config = selection["selected_feature_config"]
    if config not in FEATURE_CONFIGS or selection["test_evaluated"]:
        raise ValueError("Expected a validation-only feature selection report.")
    expected_config = max(
        selection["runs"],
        key=lambda key: selection["runs"][key]["validation"][
            selection["runs"][key]["selected_model"]
        ]["average_precision"],
    )
    if config != expected_config:
        raise ValueError("Selected feature configuration is not the validation AP winner.")
    run_dir = experiment_dir / config
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if (
        report != selection["runs"][config]
        or report["test"] is not None
        or report["selected_model"] != selection["selected_model"]
    ):
        raise ValueError("Selected run differs from the validation-only experiment report.")
    if report["model_sha256"] != file_fingerprint(run_dir / "model.joblib"):
        raise ValueError("Selected model fingerprint mismatch.")
    if report["dataset"]["fingerprint_sha256"] != file_fingerprint(input_path):
        raise ValueError("Dataset fingerprint differs from the validation experiment.")
    seed = report["methodology"]["seed"]
    sample_size = report["dataset"]["requested_sample_size"]
    data = sample_customers(load_consumption(input_path), size=sample_size, seed=seed)
    partitions = split_customers(data, seed=seed)
    fingerprints = {
        key: partition_fingerprint(data.customer_ids.iloc[idx]) for key, idx in partitions.items()
    }
    if fingerprints != report["partition_fingerprints"]:
        raise ValueError("Customer partition fingerprints differ from the validation experiment.")
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
        "partitions": {
            key: data.customer_ids.iloc[idx].astype(str).tolist() for key, idx in partitions.items()
        },
    }
    if manifest != expected_manifest:
        raise ValueError("Customer split manifest differs from the validation experiment.")
    # Trusted local artifacts only, exactly as in the existing scoring workflow.
    bundle = joblib.load(run_dir / "model.joblib")
    current = experiment_metadata(
        data,
        partitions,
        input_path=input_path,
        sample_size=sample_size,
        source=report["dataset"]["source"],
        seed=seed,
        experiment_id=experiment_id or output_dir.name,
        feature_config=config,
    )
    if current["software"] != report["software"]:
        raise ValueError("Software versions differ; finalize in the experiment environment.")
    report["parent_experiment_id"] = selection["experiment_id"]
    report["experiment_id"] = current["experiment_id"]
    report["dataset"]["input_path"] = str(input_path)
    report["methodology"]["test_usage"] = "final holdout evaluation of frozen validation winner"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_manifest.json", manifest)
    analysis_path = run_dir / "validation_error_analysis.json"
    if analysis_path.is_file():
        write_json(
            output_dir / "validation_error_analysis.json",
            json.loads(analysis_path.read_text(encoding="utf-8")),
        )
    _write_holdout(
        data, partitions, bundle["model"], config=config, report=report, output_dir=output_dir
    )
    bundle["experiment_id"] = report["experiment_id"]
    joblib.dump(bundle, output_dir / "model.joblib")
    report["model_sha256"] = file_fingerprint(output_dir / "model.joblib")
    write_json(output_dir / "metrics.json", report)
    return report


def score(input_path: str | Path, model_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Score customers. Load joblib only from your own trusted model file."""
    bundle = joblib.load(model_path)
    data = load_consumption(input_path, require_labels=False)
    # Older model bundles deliberately retain the pilot's feature calculation.
    features = build_features(data, config=bundle.get("feature_config", "baseline"))
    if features.columns.tolist() != bundle["feature_columns"]:
        raise ValueError("Model feature schema differs from the incoming data.")
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    ranked = rank_customers(data.customer_ids, features, probabilities)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)
    return ranked
