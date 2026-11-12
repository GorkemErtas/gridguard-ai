import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

import gridguard.anomaly as anomaly
from gridguard.dashboard import load_anomaly_benchmark
from gridguard.data import load_consumption, split_customers
from gridguard.features import build_features
from gridguard.provenance import experiment_metadata, write_json
from gridguard.reliability import summaries_match
from gridguard.synthetic import save_synthetic


@pytest.fixture
def anomaly_reference(tmp_path):
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
    report.update(test={"average_precision": 0.99}, validation={})
    write_json(folder / "metrics.json", report)
    write_json(
        folder / "split_manifest.json",
        {
            "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
            "partitions": {key: data.customer_ids.iloc[idx].tolist() for key, idx in parts.items()},
        },
    )
    return path, folder


def test_robust_deviation_formula_zero_mad_and_all_missing_features():
    fitted = anomaly.RobustMAD().fit([[0, 10], [1, 10], [2, 10]])
    assert fitted.score_samples([[2, 10]])[0] == pytest.approx(np.log1p(1 / 1.4826) / 2)
    assert fitted.score_samples([[1, 11]])[0] > fitted.score_samples([[1, 10]])[0]
    assert np.isfinite(fitted.score_samples([[1e308, -1e308]])).all()
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
        [[np.nan, 1], [np.nan, 1]]
    )
    ignored = anomaly.RobustMAD().fit(imputed, observed_features=[False, True])
    assert ignored.score_samples([[1e200, 1]])[0] == 0
    all_ignored = anomaly.RobustMAD().fit([[0, 0]], observed_features=[False, False])
    np.testing.assert_array_equal(all_ignored.score_samples([[1e308, 1e308]]), [0])


@pytest.mark.parametrize(
    "training",
    [
        [[1, 1], [1, 1]],
        [[1e308, -1e308], [-1e308, 1e308]],
        [[1e308, 1e308], [1e308, 1e308]],
    ],
)
def test_robust_extreme_and_constant_features_are_stable(training):
    fitted = anomaly.RobustMAD().fit(training)
    assert np.isfinite(fitted.center_).all() and np.isfinite(fitted.mad_).all()
    assert np.isfinite(fitted.score_samples(training)).all()
    assert np.isfinite(fitted.score_samples([[0, 0]])).all()


@pytest.mark.parametrize("values", [[[np.nan]], [[np.inf]], [], [[1], [2, 3]]])
def test_robust_invalid_feature_inputs(values):
    with pytest.raises(ValueError):
        anomaly.RobustMAD().fit(values)


def test_robust_shape_and_parameter_validation():
    with pytest.raises(ValueError, match="mask"):
        anomaly.RobustMAD().fit([[1, 2]], observed_features=[True])
    with pytest.raises(ValueError, match="Feature columns"):
        anomaly.RobustMAD().fit([[1, 2]]).score_samples([[1]])
    for value in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="positive"):
            anomaly.RobustMAD(mad_floor=value).fit([[1]])


def test_if_contamination_changes_offset_not_raw_score_ranking():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(80, 3))
    automatic = IsolationForest(n_estimators=20, random_state=4, contamination="auto").fit(features)
    fraction = IsolationForest(n_estimators=20, random_state=4, contamination=0.1).fit(features)
    np.testing.assert_array_equal(
        automatic.score_samples(features), fraction.score_samples(features)
    )
    assert automatic.offset_ != fraction.offset_


def test_benchmark_training_membership_preprocessing_and_holdout_exclusion(
    anomaly_reference, tmp_path, monkeypatch
):
    path, folder = anomaly_reference
    original = json.loads((folder / "split_manifest.json").read_text())["partitions"]
    train_ids, test_ids = set(original["train"]), set(original["test"])
    snapshot = {item.name: item.read_bytes() for item in folder.iterdir()}
    build = anomaly.build_features
    seen = {"imputer": 0, "isolation": 0, "robust": 0}

    def guarded_features(data, **kwargs):
        assert test_ids.isdisjoint(data.customer_ids)
        result = build(data, **kwargs)
        result.index = data.customer_ids
        return result

    imputer_fit, isolation_fit, robust_fit = (
        SimpleImputer.fit,
        IsolationForest.fit,
        anomaly.RobustMAD.fit,
    )

    def guard_imputer(self, X, *args, **kwargs):
        assert set(X.index) == train_ids
        seen["imputer"] += 1
        return imputer_fit(self, X, *args, **kwargs)

    def guard_isolation(self, X, y=None, **kwargs):
        assert len(X) == len(train_ids) and y is None
        seen["isolation"] += 1
        return isolation_fit(self, X, **kwargs)

    def guard_robust(self, X, y=None, **kwargs):
        assert len(X) == len(train_ids) and y is None
        seen["robust"] += 1
        return robust_fit(self, X, **kwargs)

    monkeypatch.setattr(anomaly, "build_features", guarded_features)
    monkeypatch.setattr(SimpleImputer, "fit", guard_imputer)
    monkeypatch.setattr(IsolationForest, "fit", guard_isolation)
    monkeypatch.setattr(anomaly.RobustMAD, "fit", guard_robust)
    with monkeypatch.context() as guarded:
        guarded.setattr(
            joblib, "load", lambda *a, **kw: pytest.fail("Historical model must not be loaded")
        )
        result = anomaly.benchmark(
            path, reference_dir=folder, output_dir=tmp_path / "benchmark", capacities=[1, 50, 100]
        )
    assert seen == {"imputer": 1, "isolation": 1, "robust": 1}
    parts = json.loads((tmp_path / "benchmark/split_manifest.json").read_text())["partitions"]
    assert parts["train"] == original["train"]
    assert set(parts["excluded_holdout"]) == test_ids
    assert set(parts["development_validation"]).isdisjoint(parts["development_evaluation"])
    assert set(parts["development_validation"] + parts["development_evaluation"]) == set(
        original["validation"]
    )
    assert result["evaluation_scope"] == "development_only"
    assert result["independent_evaluation"] is None
    bundle = joblib.load(tmp_path / "benchmark/anomaly_models.joblib")
    data = load_consumption(path)
    positions = {customer: i for i, customer in enumerate(data.customer_ids)}
    training = data.subset([positions[customer] for customer in original["train"]])
    features = build_features(training)
    np.testing.assert_allclose(
        bundle["imputer"].statistics_, np.nanmedian(features, axis=0), rtol=1e-6
    )
    predictions = pd.read_csv(
        tmp_path / "benchmark/anomaly_predictions.csv", float_precision="round_trip"
    )
    assert test_ids.isdisjoint(predictions.customer_id) and train_ids.isdisjoint(
        predictions.customer_id
    )
    assert set(predictions.customer_id) == set(original["validation"])
    assert result == load_anomaly_benchmark(tmp_path / "benchmark/anomaly_benchmark.json")
    assert {item.name: item.read_bytes() for item in folder.iterdir()} == snapshot
    assert not (tmp_path / "benchmark/test_predictions.csv").exists()


def test_reproducibility_and_historical_metrics_do_not_select_methods(anomaly_reference, tmp_path):
    path, folder = anomaly_reference
    first = anomaly.benchmark(
        path, reference_dir=folder, output_dir=tmp_path / "one", experiment_id="fixed"
    )
    second = anomaly.benchmark(
        path, reference_dir=folder, output_dir=tmp_path / "two", experiment_id="fixed"
    )
    assert first == second
    parent = json.loads((folder / "metrics.json").read_text())
    parent["test"] = {"average_precision": 0.001}
    parent["selected_model"] = "historical logistic regression"
    write_json(folder / "metrics.json", parent)
    third = anomaly.benchmark(path, reference_dir=folder, output_dir=tmp_path / "three")
    assert summaries_match(first["evaluations"], third["evaluations"])
    assert first["model_parameters"] == third["model_parameters"]
    assert (tmp_path / "one/anomaly_predictions.csv").read_bytes() == (
        tmp_path / "two/anomaly_predictions.csv"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="empty"):
        anomaly.benchmark(path, reference_dir=folder, output_dir=tmp_path / "one")


def test_reference_corruption_fails_before_fitting(anomaly_reference, tmp_path, monkeypatch):
    path, folder = anomaly_reference
    manifest = json.loads((folder / "split_manifest.json").read_text())
    manifest["partitions"]["test"][0] = manifest["partitions"]["train"][0]
    write_json(folder / "split_manifest.json", manifest)
    monkeypatch.setattr(
        anomaly,
        "fit_development_baseline",
        lambda *a, **kw: pytest.fail("No fitting on invalid reference"),
    )
    with pytest.raises(ValueError, match="overlapping"):
        anomaly.benchmark(path, reference_dir=folder, output_dir=tmp_path / "bad")


def test_legacy_reference_and_changed_dataset_are_rejected(anomaly_reference, tmp_path):
    path, folder = anomaly_reference
    saved = (folder / "metrics.json").read_bytes()
    report = json.loads(saved)
    del report["schema_version"]
    write_json(folder / "metrics.json", report)
    with pytest.raises(ValueError, match="schema-v2"):
        anomaly.benchmark(path, reference_dir=folder, output_dir=tmp_path / "legacy")
    (folder / "metrics.json").write_bytes(saved)
    with path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Dataset fingerprint"):
        anomaly.benchmark(path, reference_dir=folder, output_dir=tmp_path / "changed")


def test_anomaly_cli(anomaly_reference, tmp_path, capsys):
    from gridguard.cli import main

    path, folder = anomaly_reference
    main(
        [
            "benchmark-anomaly",
            "--input",
            str(path),
            "--reference-dir",
            str(folder),
            "--output-dir",
            str(tmp_path / "cli"),
            "--capacities",
            "5",
            "50",
        ]
    )
    assert json.loads(capsys.readouterr().out)["evaluation_scope"] == "development_only"
    assert (tmp_path / "cli/anomaly_benchmark.json").is_file()


@pytest.fixture
def saved_anomaly(anomaly_reference, tmp_path):
    path, reference = anomaly_reference
    folder = tmp_path / "saved-anomaly"
    anomaly.benchmark(path, reference_dir=reference, output_dir=folder)
    return folder


@pytest.mark.parametrize("filename", ["anomaly_predictions.csv", "split_manifest.json"])
def test_saved_artifact_fingerprints(saved_anomaly, filename):
    with (saved_anomaly / filename).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_anomaly_benchmark(saved_anomaly / "anomaly_benchmark.json")


def test_forged_scope_and_aggregate_metrics_are_rejected(saved_anomaly):
    path = saved_anomaly / "anomaly_benchmark.json"
    report = json.loads(path.read_text())
    report["evaluation_scope"] = "independent_final"
    write_json(path, report)
    with pytest.raises(ValueError, match="development-only"):
        load_anomaly_benchmark(path)
    report["evaluation_scope"] = "development_only"
    report["evaluations"]["development_evaluation"]["methods"]["isolation_forest"][
        "average_precision"
    ] = 0.123456
    write_json(path, report)
    with pytest.raises(ValueError, match="differ from"):
        load_anomaly_benchmark(path)


def test_substituted_customer_fails_even_with_updated_prediction_checksum(saved_anomaly):
    from gridguard.provenance import file_fingerprint

    csv = saved_anomaly / "anomaly_predictions.csv"
    frame = pd.read_csv(csv)
    frame.loc[0, "customer_id"] = "substituted"
    frame.to_csv(csv, index=False)
    report = json.loads((saved_anomaly / "anomaly_benchmark.json").read_text())
    report["predictions_sha256"] = file_fingerprint(csv)
    write_json(saved_anomaly / "anomaly_benchmark.json", report)
    with pytest.raises(ValueError, match="membership"):
        load_anomaly_benchmark(saved_anomaly / "anomaly_benchmark.json")


def test_dashboard_is_read_only_private_and_independent_of_legacy_run(
    saved_anomaly, monkeypatch, tmp_path
):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from sklearn.ensemble import RandomForestClassifier
    from streamlit.testing.v1 import AppTest

    def forbidden(*args, **kwargs):
        pytest.fail("The saved benchmark view must not load or fit any model")

    before = {path.name: path.read_bytes() for path in saved_anomaly.iterdir()}
    monkeypatch.setattr(joblib, "load", forbidden)
    for estimator in (RandomForestClassifier, IsolationForest, SimpleImputer, anomaly.RobustMAD):
        monkeypatch.setattr(estimator, "fit", forbidden)
    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(tmp_path / "no-legacy-run"))
    monkeypatch.setenv("GRIDGUARD_ANOMALY_REPORT", str(saved_anomaly / "anomaly_benchmark.json"))
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=20
    ).run()
    app.sidebar.radio[0].set_value("Anomaly Detection").run()
    assert not app.exception
    assert any("Development-only benchmark" in warning.value for warning in app.warning)
    assert any("Synthetic demonstration" in warning.value for warning in app.warning)
    assert any("not probabilities" in caption.value for caption in app.caption)
    assert len(app.get("plotly_chart")) == 4
    assert not app.get("download_button")
    assert not any("customer_id" in frame.value.columns for frame in app.dataframe)
    expected_ids = pd.read_csv(saved_anomaly / "anomaly_predictions.csv").customer_id
    text = " ".join(frame.value.to_string() for frame in app.dataframe)
    assert not any(customer in text for customer in expected_ids)
    app.selectbox(key="anomaly_partition").set_value("development_validation").run()
    assert not app.exception
    assert before == {path.name: path.read_bytes() for path in saved_anomaly.iterdir()}
    app.text_input(key="anomaly_path").set_value(str(tmp_path / "absent.json")).run()
    assert not app.exception
    assert "No valid saved anomaly benchmark" in app.info[0].value


def test_legacy_dashboard_without_optional_anomaly_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    write_json(tmp_path / "metrics.json", {"test": None, "validation": {}})
    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("GRIDGUARD_ANOMALY_REPORT", raising=False)
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=20
    ).run()
    app.sidebar.radio[0].set_value("Anomaly Detection").run()
    assert not app.exception
    assert "No valid saved anomaly benchmark" in app.info[0].value
    app.sidebar.radio[0].set_value("Overview").run()
    assert not app.exception
    assert any("validation-only" in info.value for info in app.info)
