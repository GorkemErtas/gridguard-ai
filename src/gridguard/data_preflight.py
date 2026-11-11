"""Inspect a local electricity-consumption CSV without publishing customer-level records."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.data import _parse_date


def inspect_csv(path: str | Path, *, chunk_size: int = 500) -> dict:
    """Stream over customers to summarize schema, labels and reading quality."""
    path = Path(path)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    id_col = next((col for col in ("CONS_NO", "customer_id") if col in header), None)
    label_col = next((col for col in ("FLAG", "flag") if col in header), None)
    if id_col is None or label_col is None:
        raise ValueError("Expected CONS_NO/customer_id and FLAG/flag columns")
    date_columns = [col for col in header if col not in (id_col, label_col)]
    dates = [_parse_date(str(col)) for col in date_columns]
    if len(dates) < 60 or any(date is None for date in dates):
        raise ValueError("Expected at least 60 parseable reading date columns")
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate calendar dates in header")

    ordered = sorted(dates)
    gaps = sum((b - a).days - 1 for a, b in zip(ordered[:-1], ordered[1:]))
    seen_ids = set()
    rows = positives = missing = zero = negative = invalid = nonfinite = 0
    all_readings = 0
    positive_min = positive_max = None

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False,
                             dtype={id_col: "string"}):
        ids = chunk[id_col].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
            raise ValueError("Missing or duplicate customer ID")
        id_list = ids.astype(str).tolist()
        if any(cid in seen_ids for cid in id_list):
            raise ValueError("Customer ID repeated across CSV chunks")
        seen_ids.update(id_list)

        labels = pd.to_numeric(chunk[label_col], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("FLAG contains missing or nonbinary values")
        values = chunk[date_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        missing_mask = ~np.isfinite(values)
        rows += len(chunk)
        positives += int((labels == 1).sum())
        missing += int(missing_mask.sum())
        negative += int(((values < 0) & np.isfinite(values)).sum())
        zero += int((values == 0).sum())
        invalid += int((chunk[date_columns].notna().to_numpy() & np.isnan(values)).sum())
        nonfinite += int(np.isinf(values).sum())
        all_readings += values.size
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size:
            lo, hi = float(positive.min()), float(positive.max())
            positive_min = lo if positive_min is None else min(positive_min, lo)
            positive_max = hi if positive_max is None else max(positive_max, hi)

    return {
        "file_name": path.name,
        "customers": rows,
        "reading_columns": len(date_columns),
        "date_start": ordered[0].date().isoformat(),
        "date_end": ordered[-1].date().isoformat(),
        "missing_calendar_days": gaps,
        "positive_labels": positives,
        "positive_rate": positives / rows if rows else None,
        "missing_or_nonfinite_readings": missing,
        "missing_reading_rate": missing / all_readings if all_readings else None,
        "zero_readings": zero,
        "negative_readings_masked_in_training": negative,
        "nonnumeric_nonempty_cells": invalid,
        "nonfinite_readings": nonfinite,
        "training_missing_reading_rate": (missing + negative) / all_readings if all_readings else None,
        "positive_kwh_min": positive_min,
        "positive_kwh_max": positive_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()
    summary = inspect_csv(args.input, chunk_size=args.chunk_size)
    payload = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
