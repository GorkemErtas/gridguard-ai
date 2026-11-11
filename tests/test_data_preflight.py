import pandas as pd
import pytest

from gridguard.data_preflight import inspect_csv
from gridguard.synthetic import make_synthetic


def test_data_preflight_counts_and_dates(tmp_path):
    frame = make_synthetic(customers=120, days=80, seed=4)
    frame = frame.rename(
        columns={
            col: f"{pd.Timestamp(col).year}/{pd.Timestamp(col).month}/{pd.Timestamp(col).day}"
            for col in frame.columns
            if col not in ("CONS_NO", "FLAG")
        }
    )
    path = tmp_path / "consumption.csv"
    frame.to_csv(path, index=False)
    summary = inspect_csv(path, chunk_size=19)
    assert summary["customers"] == 120
    assert summary["positive_labels"] == int(frame.FLAG.sum())
    assert summary["reading_columns"] == 80
    assert summary["missing_calendar_days"] == 0


def test_data_preflight_rejects_duplicates_across_chunks(tmp_path):
    frame = make_synthetic(customers=120, days=80, seed=4)
    frame.loc[10, "CONS_NO"] = frame.loc[0, "CONS_NO"]
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="repeated"):
        inspect_csv(path, chunk_size=8)


def test_data_preflight_reports_negative_readings(tmp_path):
    frame = make_synthetic(customers=120, days=80, seed=4)
    frame.iloc[0, 1] = -50
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    summary = inspect_csv(path)
    assert summary["negative_readings_masked_in_training"] == 1


def test_preflight_distinguishes_nonnumeric_and_nonfinite(tmp_path):
    frame = make_synthetic(customers=120, days=80, seed=4)
    column = frame.columns[1]
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = "not-a-reading"
    frame.loc[1, column] = float("inf")
    frame.loc[2, column] = -10
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    summary = inspect_csv(path)
    assert summary["nonnumeric_nonempty_cells"] == 1
    assert summary["nonfinite_readings"] == 1
    assert summary["training_missing_reading_rate"] > summary["missing_reading_rate"]
