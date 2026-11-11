import json

import joblib
import numpy as np
import pandas as pd
import pytest

import gridguard.training as training
from gridguard.data import load_consumption, split_customers
from gridguard.features import build_features
from gridguard.provenance import file_fingerprint
from gridguard.synthetic import save_synthetic


def test_ablations_never_extract_test_features_and_record_provenance(tmp_path, monkeypatch):
    path = save_synthetic(tmp_path / "data.csv", customers=120, days=90, seed=42)
    data = load_consumption(path)
    parts = split_customers(data, seed=42)
    test_ids = set(data.customer_ids.iloc[parts["test"]])
    original = training.build_features

    def guarded_features(subset, **kwargs):
        assert test_ids.isdisjoint(subset.customer_ids)
        return original(subset, **kwargs)

    monkeypatch.setattr(training, "build_features", guarded_features)
    output = tmp_path / "ablation"
    result = training.experiment(path, output_dir=output, source="synthetic")
    assert not result["test_evaluated"]
    assert (output / "experiment.json").is_file()
    public_summary = json.loads((output / "validation_summary.json").read_text())
    assert "input_path" not in public_summary["dataset"]
    assert "partitions" not in public_summary
    assert data.customer_ids.iloc[0] not in json.dumps(public_summary)
    runs = result["runs"]
    for config, report in runs.items():
        assert report["test"] is None
        assert not (output / config / "test_predictions.csv").exists()
        assert report["dataset"]["fingerprint_sha256"] == file_fingerprint(path)
        assert report["partition_fingerprints"] == runs["baseline"]["partition_fingerprints"]
        assert report["split_sizes"] == {"train": 72, "validation": 24, "test": 24}
        assert report["feature_config"] == config
        assert set(report["software"]) >= {"python", "scikit-learn", "numpy", "pandas"}
        assert "n_estimators" in report["candidate_parameters"]["random_forest"]
        assert data.customer_ids.iloc[0] not in json.dumps(report)
    best = result["selected_feature_config"]
    assert runs[best]["validation"][result["selected_model"]]["average_precision"] == max(
        run["validation"][run["selected_model"]]["average_precision"] for run in runs.values()
    )
    with pytest.raises(FileExistsError):
        training.experiment(path, output_dir=output)

    monkeypatch.setattr(training, "build_features", original)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Finalization must reuse the frozen estimator, never refit")

    monkeypatch.setattr(training, "_fit_candidates", forbidden_fit)
    finalized = training.finalize(path, experiment_dir=output, output_dir=tmp_path / "final")
    assert finalized["test"]["evaluated_customers"] == 24
    assert finalized["feature_config"] == best
    assert finalized["parent_experiment_id"] == "ablation"
    assert finalized["experiment_id"] == "final"
    assert finalized["partition_fingerprints"] == runs[best]["partition_fingerprints"]
    # A changed CSV (even with the same row count) cannot reuse the selection.
    with path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Dataset fingerprint"):
        training.finalize(path, experiment_dir=output, output_dir=tmp_path / "mismatched")


def test_imputation_and_scaling_fit_only_training_partition(tmp_path, monkeypatch):
    path = save_synthetic(tmp_path / "data.csv", customers=120, days=90, seed=42)
    original = load_consumption(path)
    parts = split_customers(original, seed=42)
    raw = pd.read_csv(path)
    held_out = np.concatenate([parts["validation"], parts["test"]])
    raw.iloc[held_out, 1:-1] = 1e6
    raw.to_csv(path, index=False)
    candidates = training.candidate_models

    def only_linear(seed):
        return {key: value for key, value in candidates(seed).items() if key != "random_forest"}

    monkeypatch.setattr(training, "candidate_models", only_linear)
    output = tmp_path / "trained"
    training.train(path, output_dir=output, validation_only=True)
    model = joblib.load(output / "model.joblib")["model"]
    x_train = build_features(load_consumption(path).subset(parts["train"]))
    np.testing.assert_allclose(
        model.named_steps["imputer"].statistics_, np.nanmedian(x_train, axis=0)
    )
    imputed_train = model.named_steps["imputer"].transform(x_train)
    np.testing.assert_allclose(
        model.named_steps["scaler"].mean_, imputed_train.mean(axis=0), rtol=1e-6
    )


def test_legacy_bundle_scoring_and_no_overwrite(tmp_path):
    path = save_synthetic(tmp_path / "data.csv", customers=120, days=90, seed=42)
    output = tmp_path / "trained"
    training.train(path, output_dir=output)
    with pytest.raises(FileExistsError):
        training.train(path, output_dir=output)
    bundle = joblib.load(output / "model.joblib")
    del bundle["feature_config"]
    joblib.dump(bundle, output / "legacy.joblib")
    ranked = training.score(path, output / "legacy.joblib", tmp_path / "ranked.csv")
    assert len(ranked) == 120
