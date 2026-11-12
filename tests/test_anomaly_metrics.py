import numpy as np
import pandas as pd
import pytest

from gridguard.anomaly_metrics import (
    ANOMALY_METHODS,
    benchmark_summary,
    compare_rankings,
    inspection_capacities,
    score_distribution,
)
from gridguard.modeling import model_metrics, ranking_metrics, ranking_order, score_metrics


def test_native_score_order_and_exact_top_k_metrics():
    y, scores, ids = [0, 1, 1, 0], [1000, 1000, -2, -100], ["z", " a ", "m", "n"]
    assert ranking_order(scores, ids, score_domain="unbounded").tolist() == [1, 0, 2, 3]
    metrics = ranking_metrics(y, scores, 2, customer_ids=ids, score_domain="unbounded")
    assert metrics == {
        "evaluated_k": 2,
        "positives_found": 1,
        "precision": 0.5,
        "recall": 0.5,
        "lift": 1,
    }
    assert ranking_metrics(y, scores, 99, customer_ids=ids, score_domain="unbounded")["recall"] == 1
    with pytest.raises(ValueError, match="probabilities"):
        ranking_order(scores, ids)


def test_probability_metrics_are_unchanged_and_scale_does_not_change_ranking():
    y, p, ids = [1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], list("dcba")
    old = model_metrics(y, np.array(p), customer_ids=ids)
    native = score_metrics(y, np.array(p) * 100 - 500, customer_ids=ids, score_domain="unbounded")
    for key in ("average_precision", "roc_auc", "ranking_at_k"):
        assert native[key] == old[key]


def test_overlap_jaccard_nonoverlap_rates_and_tie_aware_correlation():
    result = compare_rankings(
        [4, 3, 2, 1],
        [3, 4, 1, 2],
        customer_ids=list("abcd"),
        labels=[1, 0, 1, 0],
        capacities=[1, 2, 99],
        min_group_size=1,
    )
    first, second, full = result["top_k"]
    assert first["intersection"] == 0 and first["jaccard"] == 0
    assert first["left_only"]["positive_rate"] == 1
    assert first["right_only"]["positive_rate"] == 0
    assert second["intersection"] == 2 and second["jaccard"] == 1
    assert full["evaluated_k"] == 4 and full["intersection"] == 4
    assert full["left_only"]["positive_rate"] is None
    assert result["spearman_rho"] == pytest.approx(0.6)
    tied = compare_rankings([1, 1, 2], [1, 2, 2], customer_ids=list("zab"), capacities=[1])
    # Pearson correlation of average ranks [1.5,1.5,3] / [1,2.5,2.5].
    assert tied["spearman_rho"] == pytest.approx(0.5)
    swapped_ids = compare_rankings([1, 1, 2], [1, 2, 2], customer_ids=list("baz"), capacities=[1])
    assert swapped_ids["spearman_rho"] == tied["spearman_rho"]


def test_empty_singleton_and_constant_score_comparisons():
    for scores, ids in (([], []), ([1], ["a"]), ([1, 1], ["a", "b"])):
        result = compare_rankings(scores, scores, customer_ids=ids, capacities=[1, 20])
        assert result["spearman_rho"] is None
        assert result["top_k"][0]["jaccard"] == (1 if scores else None)
    assert compare_rankings([1, 2], [2, 1], customer_ids=["a", "b"], capacities=[1])[
        "spearman_rho"
    ] == pytest.approx(-1)


def test_small_nonoverlap_groups_suppress_label_rates():
    result = compare_rankings(
        [2, 1], [1, 2], customer_ids=["a", "b"], labels=[1, 0], capacities=[1]
    )
    assert result["top_k"][0]["left_only"]["customers"] == 1
    assert result["top_k"][0]["left_only"]["positive_customers"] is None
    assert result["top_k"][0]["right_only"]["status"] == "small_or_empty"


def test_distribution_statistics_and_extreme_values_stay_finite():
    result = score_distribution([-2, 0, 2], bins=2)
    assert result["median"] == result["mean"] == 0
    assert result["std"] == pytest.approx(np.std([-2, 0, 2]))
    assert sum(item["count"] for item in result["histogram"]) == 3
    for values in ([1e308, 1e308], [-1e308, 1e308], [0], []):
        result = score_distribution(values)
        assert all(
            value is None or np.isfinite(value)
            for key, value in result.items()
            if key != "histogram"
        )
        assert sum(item["count"] for item in result["histogram"]) == len(values)
    assert score_distribution([])["mean"] is None


@pytest.mark.parametrize("label", [None, 0, 1])
def test_missing_or_single_class_labels(label):
    frame = pd.DataFrame(
        {"customer_id": list("abcd"), **{method: [4, 3, 2, 1] for method in ANOMALY_METHODS}}
    )
    if label is not None:
        frame["true_label"] = label
    result = benchmark_summary(frame, capacities=[1, 10])
    assert result["methods"]["robust_mad"]["roc_auc"] is None
    if label is None:
        assert result["methods"]["robust_mad"]["ranking_at_k"] == {}
    else:
        metrics = result["methods"]["robust_mad"]
        assert metrics["average_precision"] == (1 if label else None)
        assert metrics["ranking_at_k"]["1"]["recall"] == (0.25 if label else None)
    assert result["warnings"]


def test_quality_groups_use_existing_bins_and_suppress_weak_support():
    n = 80
    frame = pd.DataFrame(
        {
            "customer_id": [str(i) for i in range(n)],
            "true_label": [0, 1] * 40,
            "missing_ratio": [0.05] * 40 + [0.2] * 5 + [0.3] * 30 + [0.7] * 5,
            **{method: np.arange(n) for method in ANOMALY_METHODS},
        }
    )
    frame.loc[45:74, "true_label"] = 0
    result = benchmark_summary(frame)
    assert [group["status"] for group in result["missing_reading_groups"]] == [
        "available",
        "small_group",
        "insufficient_class_support",
        "small_group",
    ]
    assert result["missing_reading_groups"][0]["methods"]["isolation_forest"]["roc_auc"] is not None
    assert result["missing_reading_groups"][1]["positive_customers"] is None
    assert result["missing_reading_groups"][2]["methods"] is None


def test_permutation_invariance_with_raw_score_ties():
    frame = pd.DataFrame(
        {
            "customer_id": ["z", "a", "m", "n"],
            "true_label": [0, 1, 1, 0],
            **{method: [4, 4, -10, -20] for method in ANOMALY_METHODS},
        }
    )
    result = benchmark_summary(frame, capacities=[1, 2, 3], min_group_size=1)
    assert result["methods"]["robust_mad"]["ranking_at_k"]["1"]["precision"] == 1
    from gridguard.reliability import summaries_match

    assert summaries_match(
        result,
        benchmark_summary(
            frame.sample(frac=1, random_state=4), capacities=[3, 2, 1], min_group_size=1
        ),
    )


@pytest.mark.parametrize("values", [[], [0], [-1], [1.5], [True], [None]])
def test_invalid_capacities(values):
    with pytest.raises(ValueError, match="positive integers"):
        inspection_capacities(values)


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [[1]]])
def test_invalid_unbounded_scores(scores):
    with pytest.raises(ValueError, match="finite"):
        ranking_order(scores, score_domain="unbounded")


def test_invalid_inputs_do_not_create_metrics():
    with pytest.raises(ValueError, match="customer IDs"):
        compare_rankings([1, 2], [1], customer_ids=["a", "b"])
    with pytest.raises(ValueError, match="Customer IDs"):
        compare_rankings([1], [1], customer_ids=None)
    with pytest.raises(ValueError, match="binary"):
        compare_rankings([1], [1], customer_ids=["a"], labels=[2])
    with pytest.raises(ValueError, match="all three"):
        benchmark_summary(pd.DataFrame({"customer_id": ["a"]}))
    with pytest.raises(ValueError, match="score_domain"):
        ranking_order([0.5], score_domain="unknown")
