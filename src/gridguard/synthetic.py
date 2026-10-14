"""Synthetic data ONLY for local demonstration, not performance claims."""

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(customers: int = 600, days: int = 120, seed: int = 42) -> pd.DataFrame:
    if customers < 100 or days < 60:
        raise ValueError("Synthetic demo requires >=100 customers and >=60 days.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    base = rng.lognormal(mean=3.3, sigma=0.5, size=(customers, 1))
    seasonality = 1.0 + 0.13 * np.sin(np.arange(days) * 2 * np.pi / 7)
    readings = base * seasonality[None, :] * rng.lognormal(
        mean=0.0, sigma=0.18, size=(customers, days)
    )
    labels = np.zeros(customers, dtype="int8")
    suspicious = rng.choice(customers, size=max(10, round(customers * 0.09)), replace=False)
    labels[suspicious] = 1
    # Inject a clear pattern into the last 30-50 days; this creates an easier
    # problem than real-world theft detection and must be reported as synthetic.
    for customer in suspicious:
        onset = int(rng.integers(days - min(days // 2, 50), days - 24))
        readings[customer, onset:] *= rng.uniform(0.06, 0.35)
        if rng.random() < 0.4:
            zeros = rng.choice(np.arange(onset, days), size=8, replace=False)
            readings[customer, zeros] = 0.0

    missing = rng.random((customers, days)) < 0.025
    readings[missing] = np.nan
    table = pd.DataFrame(readings.round(3), columns=dates.strftime("%m/%d/%Y"))
    table.insert(0, "CONS_NO", [f"DEMO-{i:05d}" for i in range(customers)])
    table["FLAG"] = labels
    return table


def save_synthetic(path: str | Path, *, customers: int, days: int, seed: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic(customers, days, seed).to_csv(path, index=False)
    return path
