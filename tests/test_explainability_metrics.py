import numpy as np
import pandas as pd
import pytest

from gridguard.anomaly_metrics import score_distribution
from gridguard.explainability_metrics import (
    analysis_thresholds,
    error_diagnostics,
    importance_summary,
)
from gridguard.modeling import classification_metrics, model_metrics


def frame(labels=(0, 0, 1, 1), scores=(0.1, 0.5, 0.49, 0.9)):
    return pd.DataFrame(
        {
            "customer_id": [str(i) for i in range(len(scores))],
            "true_label": labels,
            "model_score": scores,
            "x": np.arange(len(scores)),
            "missing_ratio": 0.05,
            "coefficient_variation": 0.5,
            "mean_kwh": 50,
        }
    )


def test_exact_confusion_rates_and_threshold_boundary():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.5, 0.49, 0.9], threshold=0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    for key in ("precision", "recall", "false_positive_rate", "false_negative_rate", "f1"):
        assert result[key] == 0.5
    assert classification_metrics([0, 1], [0, 1], threshold=0)["confusion_matrix"] == [
        [0, 1],
        [0, 1],
    ]
    assert classification_metrics([0, 1], [0, 1], threshold=1)["confusion_matrix"] == [
        [1, 0],
        [0, 1],
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        classification_metrics([1], [0.5], threshold=threshold)


def test_undefined_rates_and_legacy_zero_division_policy():
    result = classification_metrics([0, 0], [0.1, 0.2])
    assert result["precision"] is result["recall"] is result["false_negative_rate"] is None
    assert result["false_positive_rate"] == 0
    legacy = model_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert legacy["precision_0_5"] == legacy["recall_0_5"] == legacy["f1_0_5"] == 0
    assert classification_metrics([], [])["confusion_matrix"] == [[0, 0], [0, 0]]
    assert classification_metrics(None, [0.4])["confusion_matrix"] is None


def test_error_diagnostics_reuse_groups_and_custom_ranking_capacities():
    data = frame()
    result = error_diagnostics(
        data, ["x"], thresholds=[0.8, 0.5, 0.2], capacities=[1, 99], min_group_size=1
    )
    assert result["thresholds"][1]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["ranking"]["ranking_at_k"]["1"]["precision"] == 1
    assert result["ranking"]["ranking_at_k"]["99"]["evaluated_k"] == 4
    fixed = result["fixed_error_analysis"]
    assert set(fixed["ranking_errors_at_k"]) == {"1", "99"}
    assert fixed["classification_threshold"] == 0.5
    groups = fixed["data_quality_groups"]["missing_ratio"]["groups"]
    assert next(iter(groups.values()))["rate_status"] == "insufficient_class_support"


def test_feature_and_risk_histograms_share_edges_and_keep_missingness():
    data = frame(labels=[0, 0, 1, 1], scores=[0.1, 0.2, 0.8, 0.9])
    data["x"] = [0, 1, 9, np.nan]
    result = error_diagnostics(data, ["x"], min_group_size=1)
    distributions = [
        row
        for row in result["outcome_distributions"]
        if row["threshold"] == 0.5 and row["feature"] == "x"
    ]
    tn, tp = [row for row in distributions if row["outcome"] in ("TN", "TP")]
    assert tp["missing_values"] == 1
    assert [item["lower"] for item in tn["distribution"]["histogram"]] == [
        item["lower"] for item in tp["distribution"]["histogram"]
    ]
    assert sum(item["count"] for item in tp["distribution"]["histogram"]) == 1
    risks = [
        row
        for row in result["outcome_distributions"]
        if row["threshold"] == 0.5
        and row["feature"] == "risk_score"
        and row["status"] == "available"
    ]
    assert risks[0]["distribution"]["histogram"][0]["lower"] == 0
    assert risks[0]["distribution"]["histogram"][-1]["upper"] == 100


def test_small_groups_and_sparse_observations_do_not_expose_values():
    data = frame(labels=[0] * 30, scores=[0.1] * 30)
    data["x"] = [np.nan] * 29 + [123]
    result = error_diagnostics(data, ["x"], min_group_size=20)
    tn = next(
        row
        for row in result["outcome_distributions"]
        if row["threshold"] == 0.5 and row["feature"] == "x" and row["outcome"] == "TN"
    )
    assert tn["status"] == "insufficient_observed_values" and tn["distribution"] is None
    small = error_diagnostics(frame(), ["x"])
    assert all(row["distribution"] is None for row in small["outcome_distributions"])
    assert next(
        iter(
            small["fixed_error_analysis"]["data_quality_groups"]["missing_ratio"]["groups"].values()
        )
    )["suppressed"]


@pytest.mark.parametrize("label", [None, 0, 1])
def test_missing_single_class_and_empty_error_cohorts(label):
    data = frame()
    if label is None:
        data = data.drop(columns="true_label")
    else:
        data["true_label"] = label
    result = error_diagnostics(data, ["x"])
    if label is None:
        assert result["status"] == "labels_unavailable"
        assert result["fixed_error_analysis"] is None and result["outcome_distributions"] == []
    else:
        assert result["ranking"]["roc_auc"] is None
    empty = error_diagnostics(data.iloc[:0], ["x"])
    assert empty["customers"] == 0
    assert empty["ranking"]["ranking_at_k"] == {}


def test_repeat_summary_preserves_negative_values_and_population_sd():
    repeats = {
        "status": "available",
        "feature_names": ["negative", "constant"],
        "n_repeats": 3,
        "seed": 42,
        "customers": 10,
        "positive_customers": 5,
        "baseline_average_precision": 0.5,
        "decreases": [[-0.1, -0.2, -0.3], [0, 0, 0]],
    }
    result = importance_summary(repeats)
    assert result["features"][0]["mean_decrease"] == pytest.approx(-0.2)
    assert result["features"][0]["std_decrease"] == pytest.approx(
        np.std([-0.1, -0.2, -0.3], ddof=0)
    )
    assert result["features"][0]["negative_mean"]
    assert result["features"][1]["std_decrease"] == 0
    with pytest.raises(ValueError, match="align"):
        importance_summary({**repeats, "decreases": [[0, 0, 0]]})


def test_common_histogram_bounds_include_all_values_and_preserve_constant_cases():
    result = score_distribution([0.5, 0.5], histogram_range=(0, 1))
    assert len(result["histogram"]) == 20
    assert sum(item["count"] for item in result["histogram"]) == 2
    assert score_distribution([0.5, 0.5])["histogram"] == [{"lower": 0.5, "upper": 0.5, "count": 2}]
    for bounds in ((1, 0), (0, np.inf), (0, 0.1), (0,)):
        with pytest.raises(ValueError, match="Histogram range"):
            score_distribution([0.5], histogram_range=bounds)


def test_configuration_and_input_validation():
    assert analysis_thresholds([1, 0.5, 0, 0.5]) == [0, 0.5, 1]
    with pytest.raises(ValueError, match="primary threshold"):
        analysis_thresholds([0.3, 0.8])
    with pytest.raises(ValueError, match="Thresholds"):
        analysis_thresholds([None, 0.5])
    with pytest.raises(ValueError, match="binary"):
        classification_metrics([2], [0.5])
    with pytest.raises(ValueError, match="Missing diagnostic"):
        error_diagnostics(frame().drop(columns="x"), ["x"])
