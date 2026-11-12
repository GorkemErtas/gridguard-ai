import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import gridguard.explainability as explainability
from gridguard.dashboard import load_explainability
from gridguard.data import load_consumption, split_customers
from gridguard.features import build_features
from gridguard.provenance import experiment_metadata, file_fingerprint, write_json
from gridguard.reliability import summaries_match
from gridguard.synthetic import save_synthetic


@pytest.fixture
def reference(tmp_path):
    csv = save_synthetic(tmp_path / "demo.csv", customers=600, days=90, seed=42)
    data = load_consumption(csv)
    parts = split_customers(data, seed=42)
    folder = tmp_path / "reference"
    folder.mkdir()
    report = experiment_metadata(
        data,
        parts,
        input_path=csv,
        sample_size=None,
        source="synthetic",
        seed=42,
        experiment_id="reference",
        feature_config="baseline",
    )
    report.update(test={"average_precision": 0.9}, validation={})
    write_json(folder / "metrics.json", report)
    write_json(
        folder / "split_manifest.json",
        {
            "dataset_fingerprint_sha256": report["dataset"]["fingerprint_sha256"],
            "partitions": {
                name: data.customer_ids.iloc[idx].tolist() for name, idx in parts.items()
            },
        },
    )
    return csv, folder


@pytest.fixture
def saved(reference, tmp_path):
    csv, folder = reference
    output = tmp_path / "explainability"
    explainability.explain(csv, reference_dir=folder, output_dir=output, n_repeats=3)
    return output


def toy_model():
    features = pd.DataFrame(
        {"signal": np.linspace(-2, 2, 100), "duplicate": np.linspace(-2, 2, 100), "constant": 1.0}
    )
    labels = (features.signal > 0).astype(int)
    return LogisticRegression().fit(features, labels), features, labels


def permutation_args(n=100):
    return {
        "customer_ids": [f"importance-{i}" for i in range(n)],
        "training_ids": ["train"],
        "excluded_ids": ["holdout"],
    }


def test_deterministic_permutations_correlated_constant_and_negative_importance():
    model, features, labels = toy_model()
    first, raw = explainability.permutation_diagnostics(
        model, features, 1 - labels, n_repeats=5, **permutation_args()
    )
    second, again = explainability.permutation_diagnostics(
        model, features, 1 - labels, n_repeats=5, **permutation_args()
    )
    assert first == second and raw == again
    assert first["feature_names"] == features.columns.tolist()
    assert any(row["mean_decrease"] < 0 for row in first["features"])
    constant = next(row for row in first["features"] if row["feature"] == "constant")
    assert constant["mean_decrease"] == constant["std_decrease"] == 0


def test_feature_alignment_overlap_and_repeats_validation():
    model, features, labels = toy_model()
    with pytest.raises(ValueError, match="names/order"):
        explainability.permutation_diagnostics(
            model, features[features.columns[::-1]], labels, **permutation_args()
        )
    with pytest.raises(ValueError, match="customer IDs"):
        explainability.permutation_diagnostics(
            model, features, labels, **{**permutation_args(), "excluded_ids": ["importance-0"]}
        )
    for repeats in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="n_repeats"):
            explainability.permutation_diagnostics(
                model, features, labels, n_repeats=repeats, **permutation_args()
            )


@pytest.mark.parametrize("labels", [None, np.zeros(100), np.ones(100), np.array([])])
def test_unsupported_importance_cohorts_do_not_score_models(labels, monkeypatch):
    model, features, _ = toy_model()
    if labels is not None and not len(labels):
        features = features.iloc[:0]
    monkeypatch.setattr(model, "predict_proba", lambda *a: pytest.fail("Cannot score this cohort"))
    monkeypatch.setattr(
        explainability,
        "permutation_importance",
        lambda *a, **kw: pytest.fail("Cannot permute this cohort"),
    )
    report, repeats = explainability.permutation_diagnostics(
        model, features, labels, **permutation_args(len(features))
    )
    assert report["status"] != "available"
    assert report["baseline_average_precision"] is None and repeats["decreases"] == []
    assert all(row["mean_decrease"] is None for row in report["features"])


def test_permutation_pipeline_handles_missing_raw_features():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    features = pd.DataFrame(
        {"signal": np.r_[np.nan, np.linspace(-2, 2, 99)], "all_missing": np.nan}
    )
    labels = (features.signal.fillna(0) > 0).astype(int)
    model = make_pipeline(SimpleImputer(keep_empty_features=True), LogisticRegression()).fit(
        features, labels
    )
    report, _ = explainability.permutation_diagnostics(
        model, features, labels, n_repeats=2, **permutation_args()
    )
    assert all(np.isfinite(row["mean_decrease"]) for row in report["features"])
    assert report["features"][1]["mean_decrease"] == 0


def test_training_preprocessing_and_analysis_cohorts_are_disjoint(reference, tmp_path, monkeypatch):
    from sklearn.impute import SimpleImputer

    csv, folder = reference
    original = json.loads((folder / "split_manifest.json").read_text())["partitions"]
    train_ids, holdout = set(original["train"]), set(original["test"])
    build, fit = explainability.build_features, SimpleImputer.fit
    fits = []
    before = {path.name: path.read_bytes() for path in folder.iterdir()}

    def guarded_features(data, **kwargs):
        assert holdout.isdisjoint(data.customer_ids)
        result = build(data, **kwargs)
        result.index = data.customer_ids
        return result

    def guarded_fit(self, X, *args, **kwargs):
        assert set(X.index) == train_ids
        fits.append(len(X))
        return fit(self, X, *args, **kwargs)

    monkeypatch.setattr(explainability, "build_features", guarded_features)
    monkeypatch.setattr(SimpleImputer, "fit", guarded_fit)
    with monkeypatch.context() as guard:
        guard.setattr(
            joblib, "load", lambda *a, **kw: pytest.fail("No historical estimator loading")
        )
        result = explainability.explain(
            csv, reference_dir=folder, output_dir=tmp_path / "new", n_repeats=2
        )
    assert fits == [len(train_ids)]
    membership = json.loads((tmp_path / "new/split_manifest.json").read_text())["partitions"]
    assert membership["train"] == original["train"]
    assert set(membership["excluded_holdout"]) == holdout
    assert set(membership["development_importance"]).isdisjoint(
        membership["development_evaluation"]
    )
    assert set(membership["development_importance"] + membership["development_evaluation"]) == set(
        original["validation"]
    )
    bundle = joblib.load(tmp_path / "new/model.joblib")
    data = load_consumption(csv)
    train_features = build_features(data.subset(split_customers(data, seed=42)["train"]))
    np.testing.assert_allclose(
        bundle["model"].named_steps["imputer"].statistics_,
        np.nanmedian(train_features, axis=0),
        rtol=1e-6,
    )
    assert result == load_explainability(tmp_path / "new/explainability.json")
    assert result["methodology"]["independent_final_performance_established"] is False
    assert result["importance"]["feature_names"] == train_features.columns.tolist()
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert not (tmp_path / "new/test_predictions.csv").exists()


def test_reproducibility_no_overwrite_and_error_outcomes_do_not_choose_importance(
    reference, tmp_path, monkeypatch
):
    csv, folder = reference
    one = explainability.explain(
        csv, reference_dir=folder, output_dir=tmp_path / "one", n_repeats=2, experiment_id="fixed"
    )
    two = explainability.explain(
        csv, reference_dir=folder, output_dir=tmp_path / "two", n_repeats=2, experiment_id="fixed"
    )
    assert one == two
    with pytest.raises(FileExistsError, match="empty"):
        explainability.explain(csv, reference_dir=folder, output_dir=tmp_path / "one")
    parent = json.loads((folder / "metrics.json").read_text())
    parent["test"] = {"average_precision": 0.001}
    parent["selected_model"] = "ignored historical model"
    write_json(folder / "metrics.json", parent)
    original = explainability.error_diagnostics

    def changed_errors(frame, *args, **kwargs):
        frame = frame.copy()
        frame["true_label"] = 1 - frame.true_label
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(explainability, "error_diagnostics", changed_errors)
    three = explainability.explain(
        csv, reference_dir=folder, output_dir=tmp_path / "three", n_repeats=2
    )
    assert one["importance"] == three["importance"]
    assert one["model_parameters"] == three["model_parameters"]
    assert not summaries_match(one["errors"], three["errors"])


@pytest.mark.parametrize(
    "filename",
    ["explainability_predictions.csv", "permutation_repeats.json", "split_manifest.json"],
)
def test_artifact_checksum_failures(saved, filename):
    with (saved / filename).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_explainability(saved / "explainability.json")


def test_changed_summary_and_false_final_scope_are_rejected(saved):
    path = saved / "explainability.json"
    report = json.loads(path.read_text())
    report["evaluation_scope"] = "independent_final"
    write_json(path, report)
    with pytest.raises(ValueError, match="development-only"):
        load_explainability(path)
    report["evaluation_scope"] = "development_only"
    report["importance"]["features"][0]["mean_decrease"] = 0.987654
    write_json(path, report)
    with pytest.raises(ValueError, match="permutation summary"):
        load_explainability(path)


def test_customer_substitution_fails_with_updated_checksum(saved):
    csv = saved / "explainability_predictions.csv"
    frame = pd.read_csv(csv)
    frame.loc[0, "customer_id"] = "substituted"
    frame.to_csv(csv, index=False)
    path = saved / "explainability.json"
    report = json.loads(path.read_text())
    report["predictions_sha256"] = file_fingerprint(csv)
    write_json(path, report)
    with pytest.raises(ValueError, match="membership"):
        load_explainability(path)


def test_cli(reference, tmp_path, capsys):
    from gridguard.cli import main

    csv, folder = reference
    main(
        [
            "explain",
            "--input",
            str(csv),
            "--reference-dir",
            str(folder),
            "--output-dir",
            str(tmp_path / "cli"),
            "--n-repeats",
            "2",
        ]
    )
    assert json.loads(capsys.readouterr().out)["evaluation_scope"] == "development_only"


def test_dashboard_renders_without_loading_fitting_or_rescoring(saved, monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from streamlit.testing.v1 import AppTest

    def forbidden(*args, **kwargs):
        pytest.fail("Dashboard cannot load, fit, permute or rescore models")

    monkeypatch.setattr(joblib, "load", forbidden)
    for estimator in (RandomForestClassifier, Pipeline):
        monkeypatch.setattr(estimator, "fit", forbidden)
        monkeypatch.setattr(estimator, "predict_proba", forbidden)
    monkeypatch.setattr(SimpleImputer, "fit", forbidden)
    monkeypatch.setattr(explainability, "permutation_importance", forbidden)
    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(tmp_path / "no-legacy"))
    monkeypatch.setenv("GRIDGUARD_EXPLAINABILITY_REPORT", str(saved / "explainability.json"))
    snapshot = {path.name: file_fingerprint(path) for path in saved.iterdir()}
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=20
    ).run()
    app.sidebar.radio[0].set_value("Model Explainability").run()
    assert not app.exception
    assert any("Development-only" in warning.value for warning in app.warning)
    assert any("Synthetic demonstration" in warning.value for warning in app.warning)
    assert any("not causal effects" in item.value for item in app.caption)
    assert not app.get("download_button")
    assert not any("customer_id" in value.value.columns for value in app.dataframe)
    assert len(app.get("plotly_chart")) >= 3
    app.selectbox(key="explain_threshold").set_value(0.8).run()
    app.selectbox(key="explain_feature").set_value("missing_ratio").run()
    assert not app.exception
    assert snapshot == {path.name: file_fingerprint(path) for path in saved.iterdir()}
    app.text_input(key="explain_path").set_value(str(tmp_path / "absent.json")).run()
    assert not app.exception and "No valid saved explainability report" in app.info[0].value


def test_legacy_dashboard_remains_available_without_optional_explanations(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    write_json(tmp_path / "metrics.json", {"test": None, "validation": {}})
    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("GRIDGUARD_EXPLAINABILITY_REPORT", raising=False)
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=20
    ).run()
    app.sidebar.radio[0].set_value("Model Explainability").run()
    assert not app.exception and "No valid saved explainability report" in app.info[0].value
    app.sidebar.radio[0].set_value("Overview").run()
    assert not app.exception and any("validation-only" in row.value for row in app.info)
