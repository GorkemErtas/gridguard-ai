import json

import numpy as np
import pandas as pd

from gridguard.modeling import precision_at_k, risk_band
from gridguard.synthetic import save_synthetic
from gridguard.training import score, train


def test_precision_at_k_and_bands():
    assert precision_at_k(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1]), k=2) == 0.5
    assert risk_band(80) == "High"
    assert risk_band(50) == "Medium"
    assert risk_band(49) == "Low"


def test_training_and_scoring_end_to_end(tmp_path):
    data = save_synthetic(tmp_path / "demo.csv", customers=180, days=90, seed=12)
    artifacts = tmp_path / "artifacts"
    results = train(data, output_dir=artifacts, source="synthetic", seed=12)
    assert results["dataset"]["source"] == "synthetic"
    assert results["split_sizes"] == {"train": 108, "validation": 36, "test": 36}
    assert results["selected_model"] in {"logistic_regression", "random_forest"}
    assert (artifacts / "model.joblib").is_file()
    assert (artifacts / "metrics.json").is_file()
    assert (artifacts / "test_predictions.csv").is_file()
    assert results["test"]["evaluated_customers"] == 36

    saved_report = json.loads((artifacts / "metrics.json").read_text())
    assert saved_report["methodology"]["selection_metric"] == "validation average_precision"
    held_out = pd.read_csv(artifacts / "test_predictions.csv")
    assert len(held_out) == 36
    assert held_out["customer_id"].is_unique
    assert held_out["true_label"].isin([0, 1]).all()
    assert held_out["risk_score"].between(0, 100).all()
    assert held_out["model_score"].is_monotonic_decreasing
    k = min(50, len(held_out))
    assert abs(held_out["true_label"].head(k).mean() - results["test"]["precision_at_50"]) < 1e-9

    output = tmp_path / "ranked.csv"
    rankings = score(data, artifacts / "model.joblib", output)
    assert len(rankings) == 180
    assert "true_label" not in rankings
    assert rankings["risk_score"].is_monotonic_decreasing
    assert output.is_file()
