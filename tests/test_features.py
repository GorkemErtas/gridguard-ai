import numpy as np
import pandas as pd
import pytest

from gridguard.data import ConsumptionData, load_consumption
from gridguard.features import FEATURE_CONFIGS, build_features, calendar_readings
from gridguard.synthetic import make_synthetic


def test_calendar_windows_streaks_and_missing_days(tmp_path):
    dates = pd.date_range("2025-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    readings = pd.DataFrame(np.ones((3, 61)), columns=dates)
    readings.iloc[0, -31] = 100  # Outside last 30 calendar days.
    readings.iloc[1, 5:9] = np.nan
    readings.iloc[1, 9:14] = 0
    readings.iloc[2] = np.nan
    readings = readings.drop(columns=dates[-2])
    data = ConsumptionData(pd.Series(["a", "b", "c"]), readings, None)
    daily = calendar_readings(data)
    assert daily.shape == (3, 61)
    assert daily[dates[-2]].isna().all()
    legacy, improved = build_features(data), build_features(data, config="combined")
    assert legacy.loc[0, "recent_kwh"] > 1
    assert improved.loc[0, "recent_kwh"] == 1
    assert improved.loc[1, "longest_missing_days"] == 4
    assert improved.loc[1, "longest_zero_days"] == 5
    assert improved.loc[2, "longest_missing_days"] == 61
    assert improved.loc[2, "longest_zero_days"] == 0
    assert pd.isna(improved.loc[2, "mean_kwh"])
    assert improved.loc[0, "iqr_kwh"] == 0


def test_dates_sorted_duplicate_dates_and_extreme_readings(tmp_path):
    frame = make_synthetic(customers=100, days=61, seed=7)
    frame.iloc[0, 1] = 1e12
    frame = frame.drop(columns=frame.columns[5])
    frame = frame.loc[:, frame.columns[::-1]]
    path = tmp_path / "dates.csv"
    frame.to_csv(path, index=False)
    data = load_consumption(path)
    assert data.readings.columns.is_monotonic_increasing
    assert data.readings.iloc[0, 0] == pytest.approx(1e12)
    for config in FEATURE_CONFIGS:
        original = build_features(data, config=config)
        pd.testing.assert_frame_equal(original, build_features(data, config=config))
        # Features depend only on that customer's readings, never labels/population.
        pd.testing.assert_frame_equal(original.iloc[[0]].reset_index(drop=True),
                                      build_features(data.subset(np.array([0])), config=config))
        assert np.isfinite(original.loc[0, "maximum_kwh"])
    first = data.readings.columns[0]
    frame[pd.Timestamp(first).strftime("%Y/%m/%d")] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate daily"):
        load_consumption(path)


def test_missing_day_breaks_zero_streak():
    dates = pd.date_range("2025-01-01", periods=61).strftime("%Y-%m-%d")
    readings = pd.DataFrame(np.zeros((1, 61)), columns=dates).drop(columns=dates[30])
    data = ConsumptionData(pd.Series(["a"]), readings, None)
    assert build_features(data, config="quality").loc[0, "longest_zero_days"] == 30
