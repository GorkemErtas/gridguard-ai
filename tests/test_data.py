import numpy as np
import pandas as pd
import pytest

from gridguard.data import load_consumption, sample_customers
from gridguard.features import build_features, review_signal
from gridguard.synthetic import make_synthetic


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "synthetic.csv"
    make_synthetic(customers=120, days=80, seed=7).to_csv(path, index=False)
    return path


def test_load_data_sorted_and_no_target_leakage(csv_path):
    data = load_consumption(csv_path)
    assert len(data) == 120
    assert len(data.readings.columns) == 80
    assert data.readings.columns[0] == "2025-01-01"
    assert "FLAG" not in data.readings
    assert data.labels is not None
    features = build_features(data)
    assert features.shape == (120, 15)
    assert "FLAG" not in features.columns
    assert "recent_drop_ratio" in features
    assert isinstance(review_signal(features.iloc[0]), str)


def test_sampling_is_stratified_and_reproducible(csv_path):
    data = load_consumption(csv_path)
    sample_a = sample_customers(data, size=80, seed=3)
    sample_b = sample_customers(data, size=80, seed=3)
    assert sample_a.customer_ids.tolist() == sample_b.customer_ids.tolist()
    assert len(sample_a) == 80
    assert sample_a.labels is not None
    assert sample_a.labels.sum() > 0


def test_invalid_labels_rejected(csv_path):
    df = pd.read_csv(csv_path)
    df.loc[0, "FLAG"] = 2
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="FLAG"):
        load_consumption(csv_path)


def test_duplicate_customers_rejected(csv_path):
    df = pd.read_csv(csv_path)
    df.loc[1, "CONS_NO"] = df.loc[0, "CONS_NO"]
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_consumption(csv_path)


def test_invalid_negative_and_missing_values(csv_path):
    df = pd.read_csv(csv_path)
    col = next(column for column in df if column not in ("CONS_NO", "FLAG"))
    df.loc[0, col] = -99.0
    df.loc[1, col] = np.nan
    df.to_csv(csv_path, index=False)
    loaded = load_consumption(csv_path)
    assert pd.isna(loaded.readings.iat[0, 0])
    assert pd.isna(loaded.readings.iat[1, 0])


def test_unlabeled_csv_can_be_scored(csv_path):
    df = pd.read_csv(csv_path).drop(columns=["FLAG"])
    df.to_csv(csv_path, index=False)
    loaded = load_consumption(csv_path, require_labels=False)
    assert loaded.labels is None
    assert build_features(loaded).shape[0] == 120
