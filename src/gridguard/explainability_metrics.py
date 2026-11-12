"""Pure saved-score error diagnostics; no estimator loading, fitting or scoring."""

from numbers import Real

import numpy as np
import pandas as pd

from gridguard.anomaly_metrics import inspection_capacities, score_distribution
from gridguard.error_analysis import aggregate_errors
from gridguard.modeling import classification_metrics, ranking_order, score_metrics

EXPLAINABILITY_PARTITIONS = (
    "train",
    "development_importance",
    "development_evaluation",
    "excluded_holdout",
)
OUTCOMES = ("TN", "FP", "FN", "TP")
IMPORTANCE_NOTICE = (
    "Permutation importance measures model-specific AP sensitivity, not causal effects or "
    "individual attribution. Correlated features can share/mask importance, and independent "
    "shuffling can create unrealistic combinations. Negative decreases are retained. "
    "Repeat standard deviations describe permutation randomness, not confidence intervals "
    "or uncertainty from sampling/training. Values near machine precision are numerical "
    "noise and do not support feature rankings. Small cohorts can give unstable estimates."
)


def analysis_thresholds(values):
    values = list(values)
    if not values or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError("Thresholds must be finite numbers in [0, 1].")
    result = sorted(set(float(value) for value in values))
    if 0.5 not in result:
        raise ValueError("Thresholds must include the predefined primary threshold 0.5.")
    return result


def importance_summary(repeats):
    """Validate/summarize saved permutation decreases without calling an estimator."""
    names = repeats["feature_names"]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("Expected nonempty unique original feature names.")
    count = repeats["n_repeats"]
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("n_repeats must be a positive integer.")
    if repeats["status"] not in ("available", "empty", "single_class", "labels_unavailable"):
        raise ValueError("Unsupported permutation-importance status.")
    values = np.asarray(repeats["decreases"], dtype=float)
    if repeats["status"] == "available":
        if (
            values.shape != (len(names), count)
            or not np.isfinite(values).all()
            or ((values < -1) | (values > 1)).any()
        ):
            raise ValueError(
                "Permutation decreases must align with features and repeats in [-1, 1]."
            )
        baseline = repeats["baseline_average_precision"]
        if not isinstance(baseline, Real) or not np.isfinite(baseline) or not 0 <= baseline <= 1:
            raise ValueError("Invalid permutation baseline AP.")
        means, deviations = values.mean(axis=1), values.std(axis=1)
    else:
        if repeats["decreases"] != [] or repeats["baseline_average_precision"] is not None:
            raise ValueError("Unavailable permutation importance must not invent scores/decreases.")
        means = deviations = [None] * len(names)
    return {
        key: repeats[key]
        for key in (
            "status",
            "feature_names",
            "n_repeats",
            "seed",
            "customers",
            "positive_customers",
            "baseline_average_precision",
        )
    } | {
        "metric": "average_precision",
        "notice": IMPORTANCE_NOTICE,
        "features": [
            {
                "feature": name,
                "mean_decrease": float(mean) if mean is not None else None,
                "std_decrease": float(std) if std is not None else None,
                "negative_mean": bool(mean < 0) if mean is not None else None,
            }
            for name, mean, std in zip(names, means, deviations)
        ],
    }


def error_diagnostics(
    predictions,
    feature_names,
    *,
    thresholds=(0.2, 0.5, 0.8),
    capacities=(50, 100, 200),
    min_group_size=20,
):
    """Reuse legacy fixed-0.5 group errors; add declared thresholds and distributions.

    Histograms use shared per-feature edges across outcomes, and fixed [0,1]/
    [0,100] score axes. Small outcome cohorts expose counts, never feature values.
    Missing raw features stay missing in plots rather than becoming imputed zero.
    """
    thresholds, capacities = analysis_thresholds(thresholds), inspection_capacities(capacities)
    if (
        isinstance(min_group_size, (bool, np.bool_))
        or not isinstance(min_group_size, (int, np.integer))
        or min_group_size < 1
    ):
        raise ValueError("min_group_size must be a positive integer.")
    if not {"customer_id", "model_score", *feature_names}.issubset(predictions):
        raise ValueError("Missing diagnostic scores, IDs or original feature columns.")
    if len(feature_names) != len(set(feature_names)) or set(feature_names) & {
        "model_score",
        "risk_score",
        "customer_id",
        "true_label",
        "partition",
    }:
        raise ValueError("Feature names must be unique and separate from diagnostic fields.")
    p = predictions.model_score.to_numpy(dtype=float)
    y = predictions.true_label.to_numpy() if "true_label" in predictions else None
    ranking_order(p, predictions.customer_id)
    ranking = score_metrics(y, p, customer_ids=predictions.customer_id, capacities=capacities)
    profiles = [
        classification_metrics(y, p, threshold=value, customer_ids=predictions.customer_id)
        for value in thresholds
    ]
    fixed = (
        aggregate_errors(predictions, min_group_size=min_group_size, capacities=capacities)
        if y is not None and len(p)
        else None
    )
    if fixed:
        for feature in fixed["data_quality_groups"].values():
            for group in feature.get("groups", {}).values():
                if group.get("suppressed"):
                    group["rate_status"] = "small_group"
                else:
                    group["rate_status"] = (
                        "available"
                        if min(group["positives"], group["customers"] - group["positives"]) >= 5
                        else "insufficient_class_support"
                    )
    distributions = []
    if y is not None:
        for threshold in thresholds:
            positive = p >= threshold
            outcomes = np.where(
                np.asarray(y) == 1, np.where(positive, "TP", "FN"), np.where(positive, "FP", "TN")
            )
            for name in [*feature_names, "model_score", "risk_score"]:
                values = (
                    p
                    if name == "model_score"
                    else np.rint(p * 100)
                    if name == "risk_score"
                    else pd.to_numeric(predictions[name], errors="raise").to_numpy(dtype=float)
                )
                finite = values[np.isfinite(values)]
                bounds = (
                    (0, 1)
                    if name == "model_score"
                    else (0, 100)
                    if name == "risk_score"
                    else (
                        (float(finite.min()), float(finite.max()))
                        if len(finite) and finite.min() < finite.max()
                        else None
                    )
                )
                for outcome in OUTCOMES:
                    selected = values[outcomes == outcome]
                    observed = selected[np.isfinite(selected)]
                    available = len(observed) >= min_group_size
                    distributions.append(
                        {
                            "threshold": threshold,
                            "feature": name,
                            "outcome": outcome,
                            "customers": len(selected),
                            "observed_values": len(observed),
                            "missing_values": len(selected) - len(observed),
                            "status": "available"
                            if available
                            else "small_or_empty"
                            if len(selected) < min_group_size
                            else "all_missing"
                            if not len(observed)
                            else "insufficient_observed_values",
                            "distribution": score_distribution(observed, histogram_range=bounds)
                            if available
                            else None,
                        }
                    )
    return {
        "status": "labels_unavailable" if y is None else "available" if len(p) else "empty",
        "customers": len(p),
        "positive_customers": int(np.asarray(y).sum()) if y is not None else None,
        "ranking": ranking,
        "thresholds": profiles,
        "fixed_error_analysis": fixed,
        "outcome_distributions": distributions,
        "min_group_size": int(min_group_size),
        "group_min_class_support": 5,
        "warnings": [
            "Development diagnostics are descriptive; subgroup associations are not causal effects."
        ]
        + (
            ["Fewer than 100 evaluation customers; rates/distributions may be unstable."]
            if len(p) < 100
            else []
        )
        + (
            ["Single-class labels; some ranking and error rates are undefined."]
            if y is not None and len(np.unique(y)) < 2
            else []
        ),
    }
