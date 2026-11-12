"""Pure probability diagnostics; no estimator loading or fitting."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from gridguard.modeling import model_metrics, ranking_metrics, ranking_order

CALIBRATION_PARTITIONS = (
    "train",
    "model_selection",
    "calibration_fit",
    "calibration_selection",
    "development_evaluation",
    "excluded_holdout",
)
METHODS = ("uncalibrated", "sigmoid", "isotonic")
ECE_NOTICE = (
    "ECE uses fixed equal-width bins and sample-weighted absolute bin gaps. "
    "It depends on bin edges and sample size, hides within-bin errors, and is not a proper "
    "scoring rule or proof of calibration. Empty bins have no estimated event rate. "
    "Brier score and log loss assess overall probability quality, not calibration alone."
)


def validate_bins(n_bins: int) -> None:
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError("n_bins must be an integer from 1 to 50.")
    if not 1 <= n_bins <= 50:
        raise ValueError("n_bins must be an integer from 1 to 50.")


def reliability_metrics(y_true, probabilities, *, customer_ids, n_bins=10) -> dict:
    """Binary-label diagnostics using the existing deterministic ranking contract.

    Uniform bins follow sklearn.calibration_curve: [0, first edge], then
    (left, right], including p=1 in the final bin. Empty bins are retained with
    null means/rates. Log loss uses sklearn's float64 machine-epsilon clipping;
    Brier, ranking and binning use the original probabilities, including 0/1.
    Missing labels or empty cohorts yield explicit unavailable metrics. Single-
    class cohorts retain proper scores but undefined ROC-AUC and existing AP/
    recall conventions. Small cohorts and constant predictions carry warnings.
    """
    validate_bins(n_bins)
    n_bins = int(n_bins)
    if customer_ids is None:
        raise ValueError("Reliability diagnostics require customer IDs for deterministic ties.")
    p = np.asarray(probabilities, dtype=np.float64)
    ranking_order(p, customer_ids)
    result = {
        "status": "labels_unavailable" if y_true is None else "empty",
        "customers": len(p),
        "positive_customers": None,
        "distinct_probabilities": int(len(np.unique(p))),
        "brier_score": None,
        "log_loss": None,
        "ece": None,
        "average_precision": None,
        "roc_auc": None,
        "ranking_at_k": {},
        "bins": [],
        "warnings": [],
    }
    if y_true is None:
        return result
    # Canonical binary-label/alignment validation, also for empty cohorts.
    ranking_metrics(y_true, p, 1, customer_ids=customer_ids)
    y = np.asarray(y_true, dtype=int)
    result["positive_customers"] = int(y.sum())
    if not len(p):
        return result
    computed = model_metrics(y, p, customer_ids=customer_ids)
    edges = np.linspace(0, 1, n_bins + 1)
    counts = np.bincount(np.searchsorted(edges[1:-1], p), minlength=n_bins)
    rates, means = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    populated = iter(zip(means, rates))
    bins = []
    for index, count in enumerate(counts):
        mean, rate = next(populated) if count else (None, None)
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(count),
                "mean_probability": float(mean) if count else None,
                "positive_fraction": float(rate) if count else None,
            }
        )
    warnings = []
    if len(p) < 100:
        warnings.append("Fewer than 100 diagnostic customers; bin rates can be unstable.")
    if len(np.unique(y)) < 2:
        warnings.append("Single-class labels: discrimination/recall metrics may be undefined.")
    if len(np.unique(p)) < 2:
        warnings.append("Constant probabilities: all customers tie; ranking uses customer IDs.")
    return {
        **result,
        "status": "available",
        "brier_score": float(brier_score_loss(y, p, pos_label=1)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(
            sum(
                item["count"] * abs(item["mean_probability"] - item["positive_fraction"])
                for item in bins
                if item["count"]
            )
            / len(p)
        ),
        **{key: computed[key] for key in ("average_precision", "roc_auc", "ranking_at_k")},
        "bins": bins,
        "warnings": warnings,
    }


def summaries_match(saved, computed) -> bool:
    """Allow numerical roundoff across supported sklearn/NumPy versions."""
    if isinstance(saved, dict) and isinstance(computed, dict):
        return saved.keys() == computed.keys() and all(
            summaries_match(saved[key], computed[key]) for key in saved
        )
    if isinstance(saved, list) and isinstance(computed, list):
        return len(saved) == len(computed) and all(
            summaries_match(left, right) for left, right in zip(saved, computed)
        )
    if isinstance(saved, (int, float)) and isinstance(computed, (int, float)):
        return bool(np.isclose(saved, computed, rtol=1e-10, atol=1e-12))
    return saved == computed
