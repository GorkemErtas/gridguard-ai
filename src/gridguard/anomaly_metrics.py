"""Pure, threshold-free anomaly ranking and complementarity diagnostics."""

from itertools import combinations

import numpy as np
import pandas as pd

from gridguard.error_analysis import QUALITY_BINS
from gridguard.modeling import ranking_order, score_metrics

ANOMALY_METHODS = ("supervised_rf", "isolation_forest", "robust_mad")
ANOMALY_PARTITIONS = (
    "train",
    "development_validation",
    "development_evaluation",
    "excluded_holdout",
)


def inspection_capacities(values):
    values = list(values)
    if not values or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1
        for k in values
    ):
        raise ValueError("Inspection capacities must be nonempty positive integers.")
    return sorted(set(int(k) for k in values))


def score_distribution(scores, *, bins=20):
    """Native-score summaries/histogram; rescale arithmetic only to avoid overflow."""
    scores = np.asarray(scores, dtype=float)
    ranking_order(scores, score_domain="unbounded")
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("Histogram bins must be a positive integer.")
    result = {
        "customers": len(scores),
        "distinct_scores": int(len(np.unique(scores))),
        **{key: None for key in ("minimum", "p05", "median", "p95", "maximum", "mean", "std")},
        "histogram": [],
    }
    if not len(scores):
        return result
    scale = max(1.0, float(np.abs(scores).max()))
    normalized = scores / scale
    quantiles = np.quantile(normalized, [0, 0.05, 0.5, 0.95, 1]) * scale
    result.update(zip(("minimum", "p05", "median", "p95", "maximum"), map(float, quantiles)))
    result.update(mean=float(normalized.mean() * scale), std=float(normalized.std() * scale))
    if result["distinct_scores"] == 1:
        result["histogram"] = [
            {"lower": float(scores[0]), "upper": float(scores[0]), "count": len(scores)}
        ]
    else:
        counts, edges = np.histogram(normalized, bins=bins)
        result["histogram"] = [
            {"lower": float(left * scale), "upper": float(right * scale), "count": int(count)}
            for left, right, count in zip(edges[:-1], edges[1:], counts)
        ]
    return result


def compare_rankings(
    left, right, *, customer_ids, labels=None, capacities=(50, 100, 200), min_group_size=20
):
    """Paired same-cohort comparison, average-rank Spearman ties, ID-broken top-K.

    No p-values or assertion that disagreement establishes an error. Rates in
    small/empty nonoverlap groups are suppressed; counts always remain visible.
    """
    capacities = inspection_capacities(capacities)
    if customer_ids is None:
        raise ValueError("Customer IDs are required for deterministic top-K ties.")
    if (
        not isinstance(min_group_size, (int, np.integer))
        or isinstance(min_group_size, bool)
        or min_group_size < 1
    ):
        raise ValueError("min_group_size must be a positive integer.")
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    a = ranking_order(left, customer_ids, score_domain="unbounded")
    b = ranking_order(right, customer_ids, score_domain="unbounded")
    score_metrics(labels, left, customer_ids=customer_ids, capacities=(), score_domain="unbounded")
    y = None if labels is None else np.asarray(labels, dtype=int)
    rho = None
    if len(left) >= 2 and len(np.unique(left)) > 1 and len(np.unique(right)) > 1:
        ranks = [pd.Series(values).rank(method="average").to_numpy() for values in (left, right)]
        rho = float(np.clip(np.corrcoef(*ranks)[0, 1], -1, 1))

    def group(indices):
        count = len(indices)
        available = y is not None and count >= min_group_size
        positives = int(y[list(indices)].sum()) if available else None
        return {
            "customers": count,
            "positive_customers": positives,
            "positive_rate": positives / count if available else None,
            "status": "available"
            if available
            else ("labels_unavailable" if y is None else "small_or_empty"),
        }

    rows = []
    for k in capacities:
        selected_a, selected_b = set(a[:k]), set(b[:k])
        intersection, union = selected_a & selected_b, selected_a | selected_b
        rows.append(
            {
                "requested_k": k,
                "evaluated_k": len(selected_a),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else None,
                "overlap_fraction": len(intersection) / len(selected_a) if selected_a else None,
                "left_only": group(selected_a - selected_b),
                "right_only": group(selected_b - selected_a),
            }
        )
    return {
        "spearman_rho": rho,
        "rank_ties": "average ranks, without customer-ID tie breaking",
        "top_k": rows,
    }


def benchmark_summary(predictions, *, capacities=(50, 100, 200), min_group_size=20):
    """Aggregate paired diagnostics without returning customer records.

    Quality group discrimination is withheld below min_group_size or five
    customers per label class. These safeguards are not power calculations.
    """
    capacities = inspection_capacities(capacities)
    if not {"customer_id", *ANOMALY_METHODS}.issubset(predictions):
        raise ValueError("Benchmark predictions require IDs and all three method scores.")
    if (
        isinstance(min_group_size, (bool, np.bool_))
        or not isinstance(min_group_size, (int, np.integer))
        or min_group_size < 1
    ):
        raise ValueError("min_group_size must be a positive integer.")
    y = predictions.true_label.to_numpy() if "true_label" in predictions else None
    methods = {}
    for method in ANOMALY_METHODS:
        methods[method] = {
            **score_metrics(
                y,
                predictions[method],
                customer_ids=predictions.customer_id,
                capacities=capacities,
                score_domain="unbounded",
            ),
            "distribution": score_distribution(predictions[method]),
        }
    pairs = [
        {
            "left_method": left,
            "right_method": right,
            **compare_rankings(
                predictions[left],
                predictions[right],
                customer_ids=predictions.customer_id,
                labels=y,
                capacities=capacities,
                min_group_size=min_group_size,
            ),
        }
        for left, right in combinations(ANOMALY_METHODS, 2)
    ]
    groups = []
    if "missing_ratio" in predictions:
        values = pd.to_numeric(predictions.missing_ratio, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        names = (
            pd.cut(values, QUALITY_BINS["missing_ratio"], right=False)
            .astype("string")
            .fillna("unavailable")
        )
        for name in sorted(names.unique()):
            mask = names.eq(name).to_numpy(dtype=bool)
            count = int(mask.sum())
            positives = int(np.asarray(y)[mask].sum()) if y is not None else None
            supported = (
                y is not None and count >= min_group_size and min(positives, count - positives) >= 5
            )
            status = (
                "available"
                if supported
                else "labels_unavailable"
                if y is None
                else "small_group"
                if count < min_group_size
                else "insufficient_class_support"
            )
            metrics = (
                {
                    method: score_metrics(
                        np.asarray(y)[mask],
                        predictions.loc[mask, method],
                        customer_ids=predictions.loc[mask, "customer_id"],
                        capacities=(),
                        score_domain="unbounded",
                    )
                    for method in ANOMALY_METHODS
                }
                if supported
                else None
            )
            groups.append(
                {
                    "group": str(name),
                    "customers": count,
                    "positive_customers": positives if count >= min_group_size else None,
                    "status": status,
                    "methods": metrics,
                }
            )
    warnings = []
    if y is None:
        warnings.append("Labels unavailable; only score/overlap diagnostics are available.")
    elif len(np.unique(y)) < 2:
        warnings.append("Single-class or empty cohort; AP/ROC-AUC/recall may be undefined.")
    if len(predictions) < 100:
        warnings.append("Fewer than 100 evaluation customers; descriptive rates can be unstable.")
    if any(methods[name]["distribution"]["distinct_scores"] <= 1 for name in ANOMALY_METHODS):
        warnings.append(
            "Constant scores: top-K may reflect customer IDs; Spearman is undefined for that method."
        )
    return {
        "customers": len(predictions),
        "positive_customers": int(np.asarray(y).sum()) if y is not None else None,
        "methods": methods,
        "complementarity": pairs,
        "missing_reading_groups": groups,
        "min_group_size": int(min_group_size),
        "min_class_support": 5,
        "warnings": warnings,
    }
