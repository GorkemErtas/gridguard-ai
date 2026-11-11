import json
import shutil

import pandas as pd
import pytest

from gridguard.evaluation import evaluate
from gridguard.provenance import file_fingerprint
from gridguard.synthetic import save_synthetic
from gridguard.training import train


@pytest.fixture(scope="module")
def saved_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation")
    csv = save_synthetic(root / "demo.csv", customers=180, days=90, seed=12)
    artifacts = root / "artifacts"
    train(csv, output_dir=artifacts, source="synthetic", seed=12)
    return artifacts


def test_evaluate_holdout_and_errors(tmp_path, saved_artifacts):
    output = tmp_path / "report"
    result = evaluate(saved_artifacts, output, bootstrap=20)
    assert result["test_customers"] == 36
    assert result["precision_at_k"]["50"]["evaluated_k"] == 36
    assert not (output / "errors.csv").exists()
    assert (output / "error_analysis.json").exists()
    assert result["holdout_provenance"].startswith("verified:")
    assert result["precision_at_k"]["50"]["recall"] == 1
    assert result["bootstrap"]["requested_iterations"] == 20
    assert json.loads((output / "evaluation.json").read_text())["source"] == "synthetic"
    again = evaluate(saved_artifacts, tmp_path / "again", bootstrap=20)
    assert again == result
    evaluate(saved_artifacts, tmp_path / "private", bootstrap=0, export_customer_errors=True)
    assert (tmp_path / "private" / "errors.csv").is_file()
    with pytest.raises(FileExistsError):
        evaluate(saved_artifacts, output, bootstrap=0)


def test_rejects_same_size_substituted_predictions(tmp_path, saved_artifacts):
    artifacts = tmp_path / "artifacts"
    shutil.copytree(saved_artifacts, artifacts)
    path = artifacts / "test_predictions.csv"
    predictions = pd.read_csv(path)
    predictions.loc[0, "customer_id"] = "another-customer"
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError, match="prediction fingerprint"):
        evaluate(artifacts, tmp_path / "report", bootstrap=0)
    report = json.loads((artifacts / "metrics.json").read_text())
    report["test_predictions_sha256"] = file_fingerprint(path)
    (artifacts / "metrics.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="held-out test partition"):
        evaluate(artifacts, tmp_path / "report", bootstrap=0)


def test_rejects_overlapping_manifest(tmp_path, saved_artifacts):
    artifacts = tmp_path / "artifacts"
    shutil.copytree(saved_artifacts, artifacts)
    path = artifacts / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["partitions"]["test"][0] = manifest["partitions"]["train"][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlapping"):
        evaluate(artifacts, tmp_path / "report", bootstrap=0)


def test_legacy_report_is_explicitly_unverified(tmp_path, saved_artifacts):
    artifacts = tmp_path / "artifacts"
    shutil.copytree(saved_artifacts, artifacts)
    path = artifacts / "metrics.json"
    report = json.loads(path.read_text())
    del report["schema_version"]
    path.write_text(json.dumps(report))
    result = evaluate(artifacts, tmp_path / "legacy", bootstrap=0)
    assert result["holdout_provenance"].startswith("legacy_unverified")


def test_bootstrap_validation(saved_artifacts, tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate(saved_artifacts, tmp_path / "negative", bootstrap=-1)


def test_rejects_non_holdout_file(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(json.dumps({"split_sizes": {"test": 2}}))
    pd.DataFrame(
        {"customer_id": ["x"], "true_label": [1], "model_score": [0.9], "risk_score": [90]}
    ).to_csv(artifacts / "test_predictions.csv", index=False)
    with pytest.raises(ValueError, match="test split size"):
        evaluate(artifacts, tmp_path / "report")
