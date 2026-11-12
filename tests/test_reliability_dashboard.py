import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from gridguard.calibration import calibrate
from gridguard.dashboard import file_signature, load_reliability
from gridguard.provenance import file_fingerprint, write_json
from gridguard.synthetic import save_synthetic
from gridguard.training import train


@pytest.fixture(scope="module")
def reliability_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("reliability-dashboard")
    csv = save_synthetic(root / "demo.csv", customers=600, days=90, seed=42)
    train(csv, output_dir=root / "reference", source="synthetic", validation_only=True)
    calibrate(csv, reference_dir=root / "reference", output_dir=root / "calibrated")
    return root / "calibrated"


def test_reliability_view_renders_without_training_or_legacy_artifacts(
    reliability_artifacts, monkeypatch, tmp_path
):
    pytest.importorskip("streamlit")
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from streamlit.testing.v1 import AppTest

    def forbidden(*args, **kwargs):
        pytest.fail("The reliability view must never load estimators or fit models/calibrators")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(CalibratedClassifierCV, "fit", forbidden)
    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(tmp_path / "absent-legacy-run"))
    monkeypatch.setenv(
        "GRIDGUARD_RELIABILITY_REPORT", str(reliability_artifacts / "reliability.json")
    )
    snapshot = {path.name: file_fingerprint(path) for path in reliability_artifacts.iterdir()}
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=20
    ).run()
    app.sidebar.radio[0].set_value("Model Reliability").run()
    assert not app.exception
    assert any("Development-only" in item.value for item in app.warning)
    assert any("Synthetic demonstration" in item.value for item in app.warning)
    assert any("not confirmed theft probabilities" in item.value for item in app.caption)
    assert any(
        "Independent final performance is not established" in item.value for item in app.warning
    )
    summary = next(frame.value for frame in app.dataframe if "brier_score" in frame.value)
    assert set(summary.method) == {"uncalibrated", "sigmoid"}
    ranking = next(frame.value for frame in app.dataframe if "requested_k" in frame.value)
    assert len(ranking) == 6
    assert len(app.get("plotly_chart")) == 2
    assert not app.get("download_button")
    assert snapshot == {
        path.name: file_fingerprint(path) for path in reliability_artifacts.iterdir()
    }
    app.text_input(key="reliability_path").set_value(str(tmp_path / "missing.json")).run()
    assert not app.exception
    assert "No valid saved reliability evaluation" in app.info[0].value


@pytest.mark.parametrize("filename", ["calibration_predictions.csv", "split_manifest.json"])
def test_reliability_checks_consumed_artifact_fingerprints(
    reliability_artifacts, tmp_path, filename
):
    folder = tmp_path / "corrupt"
    shutil.copytree(reliability_artifacts, folder)
    with (folder / filename).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_reliability(folder / "reliability.json")


def test_reliability_rejects_false_final_scope_and_changed_metrics(reliability_artifacts, tmp_path):
    folder = tmp_path / "corrupt"
    shutil.copytree(reliability_artifacts, folder)
    path = folder / "reliability.json"
    report = json.loads(path.read_text())
    report["evaluation_scope"] = "independent_final"
    write_json(path, report)
    with pytest.raises(ValueError, match="development-only"):
        load_reliability(path)
    report["evaluation_scope"] = "development_only"
    report["methods"]["uncalibrated"]["development_evaluation"]["brier_score"] = 0.7
    write_json(path, report)
    with pytest.raises(ValueError, match="differ from the diagnostic"):
        load_reliability(path)


def test_substituted_diagnostic_customers_fail_even_with_updated_checksum(
    reliability_artifacts, tmp_path
):
    folder = tmp_path / "corrupt"
    shutil.copytree(reliability_artifacts, folder)
    csv = folder / "calibration_predictions.csv"
    frame = pd.read_csv(csv)
    frame.loc[0, "customer_id"] = "substituted"
    frame.to_csv(csv, index=False)
    report = json.loads((folder / "reliability.json").read_text())
    report["predictions_sha256"] = file_fingerprint(csv)
    write_json(folder / "reliability.json", report)
    with pytest.raises(ValueError, match="membership"):
        load_reliability(folder / "reliability.json")


def test_reliability_cache_signature_includes_diagnostic_files(reliability_artifacts, tmp_path):
    folder = tmp_path / "cache"
    shutil.copytree(reliability_artifacts, folder)
    names = ("reliability.json", "split_manifest.json", "calibration_predictions.csv")
    before = file_signature(folder, names=names)
    with (folder / "calibration_predictions.csv").open("a") as handle:
        handle.write("\n")
    assert before != file_signature(folder, names=names)


def test_legacy_dashboard_handles_missing_reliability_view(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    (tmp_path / "metrics.json").write_text(json.dumps({"test": None, "validation": {}}))
    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("GRIDGUARD_RELIABILITY_REPORT", raising=False)
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=20
    ).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Model Reliability").run()
    assert not app.exception
    assert "No valid saved reliability evaluation" in app.info[0].value
    app.sidebar.radio[0].set_value("Overview").run()
    assert not app.exception
    assert any("validation-only" in item.value for item in app.info)
