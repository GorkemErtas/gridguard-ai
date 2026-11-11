import json

import pandas as pd
import pytest

from gridguard.evaluation import evaluate
from gridguard.synthetic import save_synthetic
from gridguard.training import train


def test_evaluate_holdout_and_errors(tmp_path):
    csv = save_synthetic(tmp_path / "demo.csv", customers=180, days=90, seed=12)
    artifacts = tmp_path / "artifacts"
    train(csv, output_dir=artifacts, source="synthetic", seed=12)
    output = tmp_path / "report"
    result = evaluate(artifacts, output, bootstrap=20)
    assert result["test_customers"] == 36
    assert result["precision_at_k"]["50"]["evaluated_k"] == 36
    assert (output / "errors.csv").exists()
    assert json.loads((output / "evaluation.json").read_text())["source"] == "synthetic"


def test_rejects_non_holdout_file(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(json.dumps({"split_sizes": {"test": 2}}))
    pd.DataFrame({"customer_id": ["x"], "true_label": [1], "model_score": [0.9],
                  "risk_score": [90]}).to_csv(artifacts / "test_predictions.csv", index=False)
    with pytest.raises(ValueError, match="test split size"):
        evaluate(artifacts, tmp_path / "report")
