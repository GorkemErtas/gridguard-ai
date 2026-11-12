import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from gridguard.modeling import model_metrics
from gridguard.reliability import reliability_metrics, summaries_match


def test_probability_scores_bin_boundaries_and_ece_correctness():
    y, p = [0, 0, 1, 1, 1], [0, 0.25, 0.5, 0.75, 1]
    metrics = reliability_metrics(y, p, customer_ids=list("abcde"), n_bins=4)
    assert metrics["brier_score"] == pytest.approx(0.075)
    assert metrics["brier_score"] == brier_score_loss(y, p)
    assert metrics["log_loss"] == log_loss(y, p, labels=[0, 1])
    assert metrics["ece"] == pytest.approx(0.2)
    assert [item["count"] for item in metrics["bins"]] == [2, 1, 1, 1]
    fractions, means = calibration_curve(y, p, n_bins=4, strategy="uniform")
    np.testing.assert_allclose([item["mean_probability"] for item in metrics["bins"]], means)
    np.testing.assert_allclose([item["positive_fraction"] for item in metrics["bins"]], fractions)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_constant_probabilities_and_empty_bins(label):
    result = reliability_metrics([label] * 4, [0.5] * 4, customer_ids=list("dcba"), n_bins=10)
    assert result["brier_score"] == 0.25
    assert result["log_loss"] == pytest.approx(np.log(2))
    assert result["ece"] == 0.5
    assert result["roc_auc"] is None
    assert result["average_precision"] == (1 if label else None)
    assert len([item for item in result["bins"] if item["count"] == 0]) == 9
    assert all(item["positive_fraction"] is None for item in result["bins"] if not item["count"])
    assert result["ranking_at_k"]["50"]["recall"] == (1 if label else None)
    assert len(result["warnings"]) == 3


def test_extreme_probabilities_are_finite_and_only_log_loss_is_clipped():
    correct = reliability_metrics([0, 1], [0, 1], customer_ids=["a", "b"])
    incorrect = reliability_metrics([1, 0], [0, 1], customer_ids=["a", "b"])
    assert correct["brier_score"] == correct["ece"] == 0
    assert incorrect["brier_score"] == incorrect["ece"] == 1
    assert incorrect["log_loss"] == pytest.approx(-np.log(np.finfo(np.float64).eps))
    assert 0 <= correct["log_loss"] < 1e-12
    assert np.isfinite(incorrect["log_loss"])


def test_missing_and_empty_labels_are_explicit():
    missing = reliability_metrics(None, [0.9], customer_ids=["a"])
    assert missing["status"] == "labels_unavailable"
    assert missing["positive_customers"] is None
    assert missing["brier_score"] is None
    assert missing["ranking_at_k"] == {}
    empty = reliability_metrics([], [], customer_ids=[])
    assert empty["status"] == "empty"
    assert empty["customers"] == empty["positive_customers"] == 0
    assert empty["bins"] == []


def test_ranking_metrics_reuse_raw_probabilities_and_customer_id_ties():
    ids = [f"{99 - i:03}" for i in range(100)]
    y, raw = np.r_[np.ones(50), np.zeros(50)], np.linspace(0.9, 0.1, 100)
    baseline = reliability_metrics(y, raw, customer_ids=ids)
    tied = reliability_metrics(y, np.full(100, 0.5), customer_ids=ids)
    for key in ("average_precision", "roc_auc", "ranking_at_k"):
        assert baseline[key] == model_metrics(y, raw, customer_ids=ids)[key]
    assert baseline["ranking_at_k"]["50"]["precision"] == 1
    assert tied["ranking_at_k"]["50"]["precision"] == 0
    assert tied["ranking_at_k"]["50"]["recall"] == 0
    assert tied["distinct_probabilities"] == 1
    perm = np.random.default_rng(5).permutation(100)
    shuffled = reliability_metrics(y[perm], np.full(100, 0.5), customer_ids=np.array(ids)[perm])
    assert shuffled == tied


@pytest.mark.parametrize("n_bins", [0, -1, 51, 1.5, True, "10", None])
def test_invalid_bin_counts(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        reliability_metrics([1], [0.5], customer_ids=["a"], n_bins=n_bins)


@pytest.mark.parametrize("probabilities", [[-0.1], [1.1], [np.nan], [np.inf], [[0.5]]])
def test_invalid_probability_inputs(probabilities):
    with pytest.raises(ValueError):
        reliability_metrics([1], probabilities, customer_ids=["a"])


@pytest.mark.parametrize("labels", [[np.nan], [None], [2], [0.5], [1, 0]])
def test_invalid_or_misaligned_labels(labels):
    with pytest.raises(ValueError, match="aligned"):
        reliability_metrics(labels, [0.5], customer_ids=["a"])


@pytest.mark.parametrize("ids", [["a", " a "], ["a"], ["", "b"], [None, "b"]])
def test_invalid_customer_membership(ids):
    with pytest.raises(ValueError, match="customer IDs"):
        reliability_metrics([1, 0], [0.5, 0.5], customer_ids=ids)


def test_summary_comparison_respects_nulls_shapes_and_nonfinite_values():
    assert summaries_match({"a": [1.0, None]}, {"a": [1 + 1e-13, None]})
    assert not summaries_match({"a": [1.0]}, {"a": [1.1]})
    assert not summaries_match([1], [1, 2])
    assert not summaries_match({"a": 1}, {"b": 1})
    assert not summaries_match(None, 0)
    assert not summaries_match(float("nan"), 0)


def test_customer_ids_are_required_and_numpy_integer_bins_work():
    with pytest.raises(ValueError, match="require customer IDs"):
        reliability_metrics([1], [0.5], customer_ids=None)
    result = reliability_metrics([1], [0.5], customer_ids=["a"], n_bins=np.int64(2))
    assert len(result["bins"]) == 2
