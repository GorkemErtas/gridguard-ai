import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from gridguard.data import ConsumptionData, load_consumption, sample_customers, split_customers
from gridguard.modeling import model_metrics, precision_at_k, ranking_metrics, ranking_order
from gridguard.synthetic import save_synthetic


def test_split_preserves_original_algorithm_and_customer_separation(tmp_path):
    data = load_consumption(save_synthetic(tmp_path / "data.csv", customers=180, days=90, seed=42))
    sampled = sample_customers(data, size=120, seed=42)
    parts = split_customers(sampled, seed=42)
    labels = sampled.labels.to_numpy()
    development, test = train_test_split(
        np.arange(120), test_size=0.2, random_state=42, stratify=labels
    )
    train, validation = train_test_split(
        development, test_size=0.25, random_state=42, stratify=labels[development]
    )
    for key, expected in zip(parts, (train, validation, test)):
        np.testing.assert_array_equal(parts[key], expected)
        np.testing.assert_array_equal(parts[key], split_customers(sampled, seed=42)[key])
    assert len(np.unique(np.concatenate(list(parts.values())))) == len(sampled)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(sampled.customer_ids.iloc[parts[left]]).isdisjoint(
            sampled.customer_ids.iloc[parts[right]]
        )
    duplicated = ConsumptionData(
        pd.Series(["same"] * len(sampled)), sampled.readings, sampled.labels
    )
    with pytest.raises(ValueError, match="unique"):
        split_customers(duplicated, seed=42)


def test_ties_use_same_ids_in_metrics_and_queue():
    ids = pd.Series(["z", "a", "m"])
    y, p = np.array([0, 1, 1]), np.array([0.50001, 0.50001, 0.50000])
    assert ranking_order(p, ids).tolist() == [1, 0, 2]
    assert precision_at_k(y, p, 1, customer_ids=ids) == 1
    result = ranking_metrics(y, p, 2, customer_ids=ids)
    assert result == {"evaluated_k": 2, "positives_found": 1,
                      "precision": 0.5, "recall": 0.5, "lift": 0.75}
    assert ranking_order(p[[2, 1, 0]], ids.iloc[[2, 1, 0]]).tolist() == [1, 2, 0]
    assert precision_at_k(y, p, 99) == pytest.approx(2 / 3)
    assert precision_at_k(np.array([]), np.array([])) == 0
    assert model_metrics(np.zeros(3), p)["roc_auc"] is None


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k(np.array([1]), np.array([0.7]), k)


@pytest.mark.parametrize("p", [[np.nan], [np.inf], [1.1], [-0.1]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError, match="probabilities"):
        precision_at_k(np.array([1]), np.array(p))


def test_metric_length_and_label_validation():
    with pytest.raises(ValueError, match="aligned"):
        precision_at_k(np.array([1, 0]), np.array([0.7]))
    with pytest.raises(ValueError, match="binary"):
        model_metrics(np.array([0.2]), np.array([0.7]))
