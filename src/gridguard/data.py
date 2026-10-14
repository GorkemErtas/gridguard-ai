"""Validated, customer-aligned consumption data loading.

Input: one row per customer, daily reading columns and optional binary FLAG.
No target label is included in the readings frame.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ConsumptionData:
    customer_ids: pd.Series
    readings: pd.DataFrame
    labels: pd.Series | None

    def subset(self, indices: np.ndarray) -> "ConsumptionData":
        return ConsumptionData(
            customer_ids=self.customer_ids.iloc[indices].reset_index(drop=True),
            readings=self.readings.iloc[indices].reset_index(drop=True),
            labels=None if self.labels is None else self.labels.iloc[indices].reset_index(drop=True),
        )

    def __len__(self) -> int:
        return len(self.customer_ids)


def _parse_date(column: str) -> datetime | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(column.strip(), fmt)
        except ValueError:
            continue
    return None


def load_consumption(path: str | Path, *, require_labels: bool = True) -> ConsumptionData:
    """Read SGCC-style CSV; enforce unique IDs and chronological date columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Consumption file not found: {path}")

    raw = pd.read_csv(
        path, low_memory=False, dtype={"CONS_NO": "string", "customer_id": "string"}
    )
    id_column = next((col for col in ("CONS_NO", "customer_id") if col in raw), None)
    label_column = next((col for col in ("FLAG", "flag") if col in raw), None)
    if id_column is None:
        raise ValueError("Expected a CONS_NO or customer_id column.")
    if require_labels and label_column is None:
        raise ValueError("Training requires a binary FLAG (0/1) column.")

    ids = raw[id_column].astype("string").str.strip().reset_index(drop=True)
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Customer IDs must be nonempty and unique.")

    dates: list[tuple[datetime, str]] = []
    for col in raw.columns:
        if col == id_column or col == label_column:
            continue
        parsed = _parse_date(str(col))
        if parsed is None:
            raise ValueError(f"Unexpected column {col!r}; expected daily reading dates.")
        dates.append((parsed, col))
    if len(dates) < 60:
        raise ValueError("At least 60 daily reading columns are required.")
    dates.sort()
    if len({date for date, _ in dates}) != len(dates):
        raise ValueError("Duplicate daily reading dates are not allowed.")

    readings = raw.loc[:, [col for _, col in dates]].apply(
        pd.to_numeric, errors="coerce"
    ).astype("float32")
    readings.columns = [date.strftime("%Y-%m-%d") for date, _ in dates]
    readings = readings.replace([np.inf, -np.inf], np.nan)
    # A negative kWh daily reading is invalid for this dataset contract.
    readings = readings.mask(readings < 0).reset_index(drop=True)

    labels = None
    if label_column is not None:
        labels = pd.to_numeric(raw[label_column], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("FLAG must contain only 0 or 1, with no missing values.")
        labels = labels.astype("int8").reset_index(drop=True)

    return ConsumptionData(customer_ids=ids, readings=readings, labels=labels)


def sample_customers(
    data: ConsumptionData, *, size: int | None, seed: int
) -> ConsumptionData:
    """Reproducible stratified subsample, preserving minority labels."""
    if size is None or size >= len(data):
        return data
    if size < 80:
        raise ValueError("Use at least 80 customers for train/validation/test evaluation.")
    indices = np.arange(len(data))
    chosen, _ = train_test_split(
        indices,
        train_size=size,
        stratify=data.labels if data.labels is not None else None,
        random_state=seed,
    )
    return data.subset(np.sort(chosen))
