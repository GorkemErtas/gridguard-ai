import json

import numpy as np
import pandas as pd

from gridguard.error_analysis import aggregate_errors


def test_threshold_errors_ranking_errors_and_quality_groups():
    predictions = pd.DataFrame(
        {
            "customer_id": [f"fixture-{i:03}" for i in range(80)],
            "true_label": np.repeat([0, 0, 1, 1], 20),
            "model_score": np.repeat([0.9, 0.6, 0.4, 0.1], 20),
            "missing_ratio": np.repeat([0.05, 0.05, 0.5, 0.5], 20),
            "mean_kwh": [np.nan] * 5 + [50] * 75,
            "coefficient_variation": [0.5] * 80,
        }
    )
    report = aggregate_errors(predictions)
    errors = report["classification_errors_at_0_5"]
    assert errors["false_positive"] == errors["false_negative"] == 40
    assert errors["high_confidence_false_positive"] == 20
    assert errors["high_confidence_false_negative"] == 20
    assert errors["false_positive_rate"] == errors["false_negative_rate"] == 1
    assert report["ranking_errors_at_k"]["50"] == {
        "evaluated_k": 50,
        "negatives_in_queue": 40,
        "positives_outside_queue": 30,
    }
    groups = report["data_quality_groups"]["missing_ratio"]["groups"]
    negative_group = next(group for group in groups.values() if group["positives"] == 0)
    assert negative_group["average_precision"] is None
    assert negative_group["roc_auc"] is None
    assert negative_group["false_negative_rate"] is None
    assert report["data_quality_groups"]["mean_kwh"]["groups"]["unavailable"] == {
        "customers": 5,
        "suppressed": True,
    }
    assert "fixture-" not in json.dumps(report)
    assert aggregate_errors(predictions.sample(frac=1, random_state=7)) == report


def test_quality_groups_unavailable_for_legacy_predictions():
    result = aggregate_errors(
        pd.DataFrame({"customer_id": ["a", "b"], "true_label": [0, 1], "model_score": [0.9, 0.1]})
    )
    assert result["data_quality_groups"]["missing_ratio"]["status"].startswith("unavailable")
