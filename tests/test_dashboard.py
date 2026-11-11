import json

import pandas as pd
import pytest

from gridguard.dashboard import (
    discover_runs,
    file_signature,
    filter_queue,
    load_run,
    optional_report,
    prediction_metrics,
    private_export,
    quality_groups,
    queue_metrics,
)
from gridguard.provenance import file_fingerprint, partition_fingerprint


@pytest.fixture
def dashboard_artifacts(tmp_path):
    folder = tmp_path / "synthetic-final"
    folder.mkdir()
    pd.DataFrame(
        {
            "customer_id": ["z", "a", "m", "=external"],
            "model_score": [0.80001, 0.80001, 0.79999, 0.1],
            "true_label": [0, 1, 1, 0],
        }
    ).to_csv(folder / "test_predictions.csv", index=False)
    report = {
        "test": {"average_precision": 0.5},
        "validation": {},
        "dataset": {
            "source": "synthetic",
            "customers": 4,
            "positive_customers": 2,
            "positive_rate": 0.5,
        },
        "split_sizes": {"test": 4},
    }
    (folder / "metrics.json").write_text(json.dumps(report))
    return folder


def test_legacy_order_filters_and_consistent_queue_metrics(dashboard_artifacts):
    run = load_run(dashboard_artifacts)
    assert run.predictions.customer_id.tolist() == ["a", "z", "m", "=external"]
    assert "legacy_unverified" in run.provenance
    queue = filter_queue(run.predictions, capacity=1, bands=["High"])
    assert queue_metrics(queue, run.predictions) == {
        "customers": 1,
        "precision": 1,
        "recall": 0.5,
        "lift": 2,
    }
    assert prediction_metrics(run.predictions)["ranking_at_k"]["50"]["evaluated_k"] == 4
    assert filter_queue(run.predictions, capacity=3, bands=[], search="a").empty
    assert (
        filter_queue(run.predictions, capacity=3, bands=["Low"], search="=EXTERNAL").shape[0] == 1
    )
    assert filter_queue(run.predictions, capacity=3, bands=["Low"], search=".*").empty
    export = private_export(run.predictions).decode()
    assert "customer_id" not in export and "true_label" not in export
    assert "'=external" in private_export(run.predictions, include_ids=True).decode()


def test_schema_v2_checksum_and_membership(dashboard_artifacts):
    folder = dashboard_artifacts
    report = json.loads((folder / "metrics.json").read_text())
    parts = {
        "train": ["training"],
        "validation": ["validation"],
        "test": ["z", "a", "m", "=external"],
    }
    report.update(
        schema_version=2,
        test_predictions_sha256=file_fingerprint(folder / "test_predictions.csv"),
        dataset={"fingerprint_sha256": "fixture"},
        split_sizes={key: len(ids) for key, ids in parts.items()},
        partition_fingerprints={key: partition_fingerprint(ids) for key, ids in parts.items()},
    )
    (folder / "metrics.json").write_text(json.dumps(report))
    (folder / "split_manifest.json").write_text(
        json.dumps({"dataset_fingerprint_sha256": "fixture", "partitions": parts})
    )
    assert load_run(folder).provenance.startswith("verified:")
    (folder / "split_manifest.json").unlink()
    assert "manifest unavailable" in load_run(folder).provenance
    with (folder / "test_predictions.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_run(folder)


def test_validation_missing_reports_and_cache_keys(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"test": None, "validation": {}}))
    assert load_run(tmp_path).predictions.empty
    signature = file_signature(tmp_path)
    path.write_text(json.dumps({"test": None, "validation": {}, "experiment_id": "new"}))
    assert signature != file_signature(tmp_path)
    assert str(tmp_path) in discover_runs(tmp_path)
    assert optional_report(tmp_path / "missing.json")[0] == {}
    path.write_text("not json")
    assert optional_report(path)[1].startswith("Could not read")
    assert quality_groups({}, "missing_ratio").empty


def test_corrupt_scores_and_labels_fail_closed(dashboard_artifacts):
    path = dashboard_artifacts / "test_predictions.csv"
    df = pd.read_csv(path)
    df.loc[0, "model_score"] = 2
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="probabilities"):
        load_run(dashboard_artifacts)


def test_unlabeled_predictions_do_not_invent_queue_performance(dashboard_artifacts):
    path = dashboard_artifacts / "test_predictions.csv"
    frame = pd.read_csv(path).drop(columns="true_label")
    frame.to_csv(path, index=False)
    run = load_run(dashboard_artifacts)
    assert prediction_metrics(run.predictions) == {}
    assert queue_metrics(run.predictions, run.predictions)["precision"] is None


def test_duplicate_customer_ids_rejected(dashboard_artifacts):
    path = dashboard_artifacts / "test_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "customer_id"] = frame.loc[1, "customer_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_run(dashboard_artifacts)


def test_all_dashboard_sections_smoke(dashboard_artifacts, monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(dashboard_artifacts))
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=15
    )
    app.run()
    assert not app.exception
    for section in ["Model performance", "Inspection queue", "Customer analytics", "Data quality"]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception, section
    from gridguard.synthetic import make_synthetic

    readings = make_synthetic(customers=100, days=61, seed=42)
    readings.loc[0, "CONS_NO"] = "a"
    raw_path = dashboard_artifacts.parent / "history.csv"
    readings.to_csv(raw_path, index=False)
    app.sidebar.radio[0].set_value("Customer analytics").run()
    app.text_input[1].set_value(str(raw_path))
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Calendar days" and metric.value == "61" for metric in app.metric)
    app.sidebar.radio[0].set_value("Inspection queue").run()
    app.multiselect[0].set_value([]).run()
    assert "No customers" in app.info[0].value
