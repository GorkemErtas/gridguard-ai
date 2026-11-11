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
            labels=None
            if self.labels is None
            else self.labels.iloc[indices].reset_index(drop=True),
        )

    def __len__(self) -> int:
        return len(self.customer_ids)


def _parse_date(column: str) -> datetime | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(column.strip(), fmt)
        except ValueError:
            continue
    return None


def load_consumption(path: str | Path, *, require_labels: bool = True) -> ConsumptionData:
    """Read customer-by-day CSV; enforce unique IDs and chronological dates."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Consumption file not found: {path}")

    raw = pd.read_csv(path, low_memory=False, dtype={"CONS_NO": "string", "customer_id": "string"})
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

    readings = (
        raw.loc[:, [col for _, col in dates]]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float32")
    )
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


def load_customer_history(path: str | Path, customer_id: str, *, chunk_size: int = 250):
    """Bounded dashboard lookup using the loader's dates and invalid-reading policy."""
    for chunk in pd.read_csv(
        path, chunksize=chunk_size, dtype={"CONS_NO": "string", "customer_id": "string"}
    ):
        id_column = next((col for col in ("CONS_NO", "customer_id") if col in chunk), None)
        if id_column is None:
            raise ValueError("Expected a CONS_NO or customer_id column.")
        selected = chunk.loc[chunk[id_column].astype("string").str.strip() == customer_id]
        if selected.empty:
            continue
        dates = sorted((date, col) for col in chunk if (date := _parse_date(str(col))) is not None)
        if not dates:
            return pd.DataFrame(columns=["date", "kwh"])
        if len({date for date, _ in dates}) != len(dates):
            raise ValueError("Duplicate daily reading dates are not allowed.")
        values = pd.to_numeric(selected.iloc[0][[col for _, col in dates]], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).mask(values < 0)
        history = pd.Series(values.to_numpy(dtype=float), index=[date for date, _ in dates])
        return (
            history.reindex(pd.date_range(dates[0][0], dates[-1][0], freq="D"))
            .rename_axis("date")
            .reset_index(name="kwh")
        )
    return pd.DataFrame(columns=["date", "kwh"])


def sample_customers(data: ConsumptionData, *, size: int | None, seed: int) -> ConsumptionData:
    """Reproducible stratified subsample, preserving minority labels."""
    _validate_customers(data)
    if size is not None and size < 80:
        raise ValueError("Use at least 80 customers for train/validation/test evaluation.")
    if size is None or size >= len(data):
        return data
    indices = np.arange(len(data))
    chosen, _ = train_test_split(
        indices,
        train_size=size,
        stratify=data.labels if data.labels is not None else None,
        random_state=seed,
    )
    return data.subset(np.sort(chosen))


def _validate_customers(data: ConsumptionData) -> None:
    ids = data.customer_ids.astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Customer IDs must be nonempty and unique before partitioning.")
    if len(data.readings) != len(ids) or (data.labels is not None and len(data.labels) != len(ids)):
        raise ValueError("Customer IDs, readings and labels must be aligned.")


def split_customers(data: ConsumptionData, *, seed: int) -> dict[str, np.ndarray]:
    """Original positional 60/20/20 split, preserving the pilot's membership/order."""
    _validate_customers(data)
    if data.labels is None or not data.labels.isin([0, 1]).all():
        raise ValueError("Splitting requires binary labels.")
    labels = data.labels.to_numpy(dtype=int)
    positives = int(labels.sum())
    if len(labels) < 80 or positives < 10 or len(labels) - positives < 10:
        raise ValueError("At least 80 customers with >=10 samples per class are needed.")
    development, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        development, test_size=0.25, random_state=seed, stratify=labels[development]
    )
    return {"train": train_idx, "validation": val_idx, "test": test_idx}
