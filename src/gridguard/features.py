"""Deterministic, customer-level features and descriptive review signals."""

import numpy as np
import pandas as pd

from gridguard.data import ConsumptionData


def build_features(data: ConsumptionData) -> pd.DataFrame:
    """Derive features from readings only, in chronological order.

    Missingness is retained as its own feature; remaining feature NaNs are
    imputed strictly inside the fitted training pipeline.
    """
    readings = data.readings
    observed = readings.notna().sum(axis=1).clip(lower=1)
    changes = readings.diff(axis=1)
    previous = readings.iloc[:, -60:-30].mean(axis=1)
    recent = readings.iloc[:, -30:].mean(axis=1)
    weekend = pd.to_datetime(readings.columns).dayofweek >= 5
    weekends = readings.loc[:, weekend].mean(axis=1)
    weekdays = readings.loc[:, ~weekend].mean(axis=1)

    features = pd.DataFrame(
        {
            "mean_kwh": readings.mean(axis=1),
            "median_kwh": readings.median(axis=1),
            "std_kwh": readings.std(axis=1, ddof=0),
            "p95_kwh": readings.quantile(0.95, axis=1),
            "maximum_kwh": readings.max(axis=1),
            "missing_ratio": readings.isna().mean(axis=1),
            "zero_ratio": readings.eq(0).sum(axis=1) / observed,
            "active_ratio": readings.gt(0).sum(axis=1) / observed,
            "recent_kwh": recent,
            "previous_kwh": previous,
            "recent_previous_ratio": (recent + 1) / (previous + 1),
            "recent_drop_ratio": ((previous - recent) / (previous + 1)).clip(0, 1),
            "mean_daily_change": changes.abs().mean(axis=1),
            "weekend_weekday_ratio": (weekends + 1) / (weekdays + 1),
        }
    )
    features["coefficient_variation"] = features["std_kwh"] / (features["mean_kwh"] + 1)
    return features.replace([np.inf, -np.inf], np.nan).astype("float32")


def risk_band(score: float) -> str:
    """Illustrative UI bands; not a validated operational decision threshold."""
    return "High" if score >= 80 else "Medium" if score >= 50 else "Low"


def review_signal(row: pd.Series) -> str:
    """Heuristic reading summary, not a feature-attribution explanation."""
    reasons = []
    if row["recent_drop_ratio"] >= 0.35:
        reasons.append("Recent consumption drop")
    if row["missing_ratio"] >= 0.25:
        reasons.append("High missing-reading rate")
    if row["coefficient_variation"] >= 1.2:
        reasons.append("Unusually variable consumption")
    if row["zero_ratio"] >= 0.3:
        reasons.append("Frequent zero readings")
    return "; ".join(reasons[:3]) if reasons else "No strong rule-based indicator"
