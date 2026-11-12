import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import gridguard.calibration as calibration
from gridguard.dashboard import load_reliability
from gridguard.data import ConsumptionData, load_consumption, split_customers
from gridguard.features import build_features
from gridguard.provenance import experiment_metadata, write_json
from gridguard.reliability import CALIBRATION_PARTITIONS
from gridguard.synthetic import save_synthetic
from gridguard.training import score


@pytest.fixture
def reference(tmp_path):
    path = save_synthetic(tmp_path / "data.csv", customers=600, days=90, seed=42)
    data = load_consumption(path)
    parts = split_customers(data, seed=42)
    folder = tmp_path / "reference"
    folder.mkdir()
    report = experiment_metadata(
        data,
        parts,
        input_path=path,
        sample_size=None,
        source="synthetic",
        seed=42,
        experiment_id="reference",
        feature_config="baseline",
    )
    report.update(test=None, validation={})
    write_json(folder / "metrics.json", report)
    write_json(
        folder / "split_manifest.json",
        {
            "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
            "partitions": {key: data.customer_ids.iloc[idx].tolist() for key, idx in parts.items()},
        },
    )
    return path, folder


def test_partition_separation_reproducibility_and_sizes(reference):
    path, _ = reference
    data = load_consumption(path)
    parts = calibration.split_development(data, seed=12)
    for left, values in parts.items():
        np.testing.assert_array_equal(values, calibration.split_development(data, seed=12)[left])
        assert set(data.labels.iloc[values]) == {0, 1}
        for right, other in parts.items():
            if right != left:
                assert set(values).isdisjoint(other)
    assert set(np.concatenate(list(parts.values()))) == set(range(len(data)))
    assert {name: len(values) for name, values in parts.items()} == {
        "train": 300,
        "model_selection": 60,
        "calibration_fit": 120,
        "calibration_selection": 60,
        "development_evaluation": 60,
    }
    assert not np.array_equal(parts["train"], calibration.split_development(data, seed=13)["train"])


def test_workflow_excludes_holdout_and_fits_preprocessing_only_on_train(
    reference, tmp_path, monkeypatch
):
    path, folder = reference
    original = json.loads((folder / "split_manifest.json").read_text())
    holdout = set(original["partitions"]["test"])
    snapshots = {item.name: item.read_bytes() for item in folder.iterdir()}
    build = calibration.build_features
    import gridguard.training as training

    def guarded_features(data, **kwargs):
        assert holdout.isdisjoint(data.customer_ids), (
            "Final holdout features must never be extracted"
        )
        return build(data, **kwargs)

    monkeypatch.setattr(calibration, "build_features", guarded_features)
    monkeypatch.setattr(training, "build_features", guarded_features)
    output = tmp_path / "calibration"
    result = calibration.calibrate(path, reference_dir=folder, output_dir=output)
    parts = json.loads((output / "split_manifest.json").read_text())["partitions"]
    assert set(parts) == set(CALIBRATION_PARTITIONS)
    assert set(parts["excluded_holdout"]) == holdout
    assert result["evaluation_scope"] == "development_only"
    assert result["independent_evaluation"] is None
    assert result["methodology"]["independent_final_performance_established"] is False
    assert "isotonic" in result["skipped_methods"]
    assert set(result["methods"]) == {"uncalibrated", "sigmoid"}
    assert result["selected_method"] == min(
        result["methods"], key=lambda method: result["methods"][method]["selection"]["brier_score"]
    )
    data = load_consumption(path)
    positions = {customer: i for i, customer in enumerate(data.customer_ids)}
    training_data = data.subset(np.array([positions[customer] for customer in parts["train"]]))
    features = build_features(training_data)
    base = joblib.load(output / "model.joblib")["uncalibrated_model"]
    np.testing.assert_allclose(
        base.named_steps["imputer"].statistics_, np.nanmedian(features, axis=0), rtol=1e-6
    )
    if result["selected_model"] == "logistic_regression":
        imputed = base.named_steps["imputer"].transform(features)
        np.testing.assert_allclose(
            base.named_steps["scaler"].mean_, imputed.mean(axis=0), rtol=1e-6
        )
    predictions = pd.read_csv(output / "calibration_predictions.csv")
    assert set(predictions.customer_id) == set(parts["development_evaluation"])
    assert holdout.isdisjoint(predictions.customer_id)
    assert not (output / "test_predictions.csv").exists()
    assert {item.name: item.read_bytes() for item in folder.iterdir()} == snapshots
    assert load_reliability(output / "reliability.json") == result
    # New selected-model bundle retains the existing score interface; old bundles stay unchanged.
    inference = pd.read_csv(path)
    inference = inference.loc[inference.CONS_NO.isin(parts["development_evaluation"])]
    inference.to_csv(tmp_path / "development-inference.csv", index=False)
    ranked = score(
        tmp_path / "development-inference.csv", output / "model.joblib", tmp_path / "ranking.csv"
    )
    assert len(ranked) == len(parts["development_evaluation"])
    assert "true_label" not in ranked


def test_reproducible_workflow_and_no_overwrite(reference, tmp_path):
    path, folder = reference
    first = calibration.calibrate(
        path, reference_dir=folder, output_dir=tmp_path / "first", experiment_id="fixed"
    )
    second = calibration.calibrate(
        path, reference_dir=folder, output_dir=tmp_path / "second", experiment_id="fixed"
    )
    assert first == second
    assert (tmp_path / "first/calibration_predictions.csv").read_bytes() == (
        tmp_path / "second/calibration_predictions.csv"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="empty"):
        calibration.calibrate(path, reference_dir=folder, output_dir=tmp_path / "first")


def test_diagnostic_outcomes_cannot_select_mapping(reference, tmp_path, monkeypatch):
    path, folder = reference
    baseline = calibration.calibrate(path, reference_dir=folder, output_dir=tmp_path / "baseline")
    manifest = json.loads((tmp_path / "baseline/split_manifest.json").read_text())["partitions"]
    evaluation_ids = set(manifest["development_evaluation"])
    # Inspected final metrics and winner names must not drive new selection.
    parent = json.loads((folder / "metrics.json").read_text())
    parent["test"] = {"average_precision": 0.99, "roc_auc": 0.01}
    parent["selected_model"] = "arbitrary historical winner"
    write_json(folder / "metrics.json", parent)
    original_metrics = calibration.reliability_metrics

    def changed_diagnostics(labels, probabilities, *, customer_ids, **kwargs):
        if set(customer_ids) == evaluation_ids:
            labels = 1 - np.asarray(labels)
        return original_metrics(labels, probabilities, customer_ids=customer_ids, **kwargs)

    monkeypatch.setattr(calibration, "reliability_metrics", changed_diagnostics)
    changed = calibration.calibrate(path, reference_dir=folder, output_dir=tmp_path / "changed")
    assert changed["selected_method"] == baseline["selected_method"]
    assert changed["selected_model"] == baseline["selected_model"]
    for method in baseline["methods"]:
        assert changed["methods"][method]["selection"] == baseline["methods"][method]["selection"]
        assert (
            changed["methods"][method]["development_evaluation"]["brier_score"]
            != baseline["methods"][method]["development_evaluation"]["brier_score"]
        )


def test_sigmoid_and_isotonic_freeze_base_and_are_reproducible(monkeypatch):
    rng = np.random.default_rng(3)
    x_train, x_cal, x_eval = (
        rng.normal(size=(200, 3)),
        rng.normal(size=(1000, 3)),
        rng.normal(size=(30, 3)),
    )
    y_train = (x_train[:, 0] + rng.normal(size=200) > 0).astype(int)
    y_cal = (x_cal[:, 0] + rng.normal(size=1000) > 0).astype(int)
    model = LogisticRegression().fit(x_train, y_train)
    before = model.coef_.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("The fitted base estimator must not be refitted by calibration")

    monkeypatch.setattr(model, "fit", forbidden)
    args = {
        "calibration_ids": [f"cal-{i}" for i in range(1000)],
        "training_ids": [f"train-{i}" for i in range(200)],
        "evaluation_ids": [f"eval-{i}" for i in range(30)],
    }
    first, skipped = calibration.fit_calibrators(model, x_cal, y_cal, **args)
    second, _ = calibration.fit_calibrators(model, x_cal, y_cal, **args)
    assert set(first) == {"sigmoid", "isotonic"}
    assert skipped == {}
    for method in first:
        p = first[method].predict_proba(x_eval)
        np.testing.assert_array_equal(p, second[method].predict_proba(x_eval))
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_array_equal(model.coef_, before)


@pytest.mark.parametrize("overlap", ["train", "evaluate", "normalized"])
def test_calibration_rejects_customer_overlap_before_fitting(overlap):
    args = {"calibration_ids": ["a", "b"], "training_ids": ["c"], "evaluation_ids": ["d"]}
    args["training_ids" if overlap != "evaluate" else "evaluation_ids"] = [
        " a " if overlap == "normalized" else "a"
    ]
    with pytest.raises(ValueError, match="customer IDs"):
        calibration.fit_calibrators(None, np.zeros((2, 1)), [0, 1], **args)


@pytest.mark.parametrize("labels", [[0] * 20, [1] * 20, [0, 1]])
def test_insufficient_calibration_samples_have_explicit_reasons(labels):
    fitted, skipped = calibration.fit_calibrators(
        None,
        np.zeros((len(labels), 1)),
        labels,
        calibration_ids=[str(i) for i in range(len(labels))],
        training_ids=["train"],
        evaluation_ids=["eval"],
    )
    assert fitted == {}
    assert set(skipped) == {"sigmoid", "isotonic"}
    assert "1000" in skipped["isotonic"]


def test_constant_base_scores_and_isotonic_minimum():
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    for n in (999, 1000):
        fitted, skipped = calibration.fit_calibrators(
            model,
            np.zeros((n, 1)),
            np.arange(n) % 2,
            calibration_ids=[str(i) for i in range(n)],
            training_ids=["train"],
            evaluation_ids=["eval"],
        )
        assert set(fitted) == {"sigmoid"}
        assert ("1000" if n == 999 else "distinct") in skipped["isotonic"]
        assert len(np.unique(fitted["sigmoid"].predict_proba(np.zeros((4, 1)))[:, 1])) == 1


def test_calibration_feature_alignment():
    with pytest.raises(ValueError, match="aligned"):
        calibration.fit_calibrators(
            None,
            np.zeros((21, 1)),
            [0, 1] * 10,
            calibration_ids=[str(i) for i in range(20)],
            training_ids=["train"],
            evaluation_ids=["eval"],
        )


def test_isotonic_eligibility_uses_native_response_before_probability_saturation():
    model = LogisticRegression().fit([[-1], [1], [-2], [2]], [0, 1, 0, 1])
    features = np.linspace(10000, 11000, 1000).reshape(-1, 1)
    assert len(np.unique(model.predict_proba(features)[:, 1])) == 1
    fitted, skipped = calibration.fit_calibrators(
        model,
        features,
        np.arange(1000) % 2,
        calibration_ids=[str(i) for i in range(1000)],
        training_ids=["train"],
        evaluation_ids=["eval"],
    )
    assert "isotonic" in fitted
    assert skipped == {}


@pytest.mark.parametrize(
    "labels", [None, pd.Series([0] * 100), pd.Series([1] * 100), pd.Series([0, 1] * 10)]
)
def test_missing_single_class_or_small_development_is_rejected(labels):
    n = len(labels) if labels is not None else 100
    data = ConsumptionData(
        pd.Series([str(i) for i in range(n)]), pd.DataFrame({"x": [1] * n}), labels
    )
    with pytest.raises(ValueError, match="requires|requires binary"):
        calibration.split_development(data)


def test_reference_integrity_legacy_and_source_mismatch(reference, tmp_path):
    path, folder = reference
    metrics_path = folder / "metrics.json"
    report = json.loads(metrics_path.read_text())
    original = metrics_path.read_bytes()
    del report["schema_version"]
    write_json(metrics_path, report)
    with pytest.raises(ValueError, match="schema-v2"):
        calibration.calibrate(path, reference_dir=folder, output_dir=tmp_path / "legacy")
    metrics_path.write_bytes(original)
    with path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Dataset fingerprint"):
        calibration.calibrate(path, reference_dir=folder, output_dir=tmp_path / "changed")


def test_reference_overlap_fails_before_fitting(reference, tmp_path, monkeypatch):
    path, folder = reference
    manifest_path = folder / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["partitions"]["test"][0] = manifest["partitions"]["train"][0]
    write_json(manifest_path, manifest)
    monkeypatch.setattr(
        calibration, "_fit_candidates", lambda *a, **kw: pytest.fail("Must not fit")
    )
    with pytest.raises(ValueError, match="overlapping"):
        calibration.calibrate(path, reference_dir=folder, output_dir=tmp_path / "bad")


def test_calibration_cli(reference, tmp_path, capsys):
    from gridguard.cli import main

    path, folder = reference
    main(
        [
            "calibrate",
            "--input",
            str(path),
            "--reference-dir",
            str(folder),
            "--output-dir",
            str(tmp_path / "cli"),
            "--n-bins",
            "5",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["evaluation_scope"] == "development_only"
    assert (tmp_path / "cli/reliability.json").is_file()
