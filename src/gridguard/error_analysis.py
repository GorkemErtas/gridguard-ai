"""Aggregate classification and ranking errors, without customer-level exports."""

import numpy as np
import pandas as pd

from gridguard.modeling import model_metrics, ranking_order

QUALITY_BINS = {
    "missing_ratio": [0, 0.1, 0.25, 0.5, np.inf],
    "coefficient_variation": [0, 0.5, 1, 2, np.inf],
    "mean_kwh": [0, 10, 100, 1000, np.inf],
}


def _classification_summary(labels, scores) -> dict:
    metrics = model_metrics(labels, scores)
    tn, fp, fn, tp = np.asarray(metrics["confusion_matrix"]).ravel().tolist()
    return {
        "customers": len(labels),
        "positives": int(labels.sum()),
        "positive_prevalence": metrics["positive_prevalence"],
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "high_confidence_false_positive": int(((labels == 0) & (scores >= 0.8)).sum()),
        "high_confidence_false_negative": int(((labels == 1) & (scores <= 0.2)).sum()),
    }


def aggregate_errors(
    predictions: pd.DataFrame, *, min_group_size: int = 20, capacities=(50, 100, 200)
) -> dict:
    """Fixed descriptive bins/cutoffs; never optimize thresholds on these errors.

    Small-group rates are suppressed for cautious sharing, not a formal privacy guarantee.
    The same global inspection queue is used for every group's ranking error counts.
    """
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    y = predictions.true_label.to_numpy()
    p = predictions.model_score.to_numpy(dtype=float)
    order = ranking_order(p, predictions.customer_id)
    overall = _classification_summary(y, p)
    queues = {}
    ranking_errors = {}
    for k in capacities:
        mask = np.zeros(len(y), dtype=bool)
        mask[order[:k]] = True
        queues[str(k)] = mask
        ranking_errors[str(k)] = {
            "evaluated_k": int(mask.sum()),
            "negatives_in_queue": int(((y == 0) & mask).sum()),
            "positives_outside_queue": int(((y == 1) & ~mask).sum()),
        }
    groups = {}
    for column, edges in QUALITY_BINS.items():
        if column not in predictions:
            groups[column] = {"status": "unavailable in legacy predictions"}
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        bins = pd.cut(values, bins=edges, right=False).astype("string").fillna("unavailable")
        summaries = {}
        for name in sorted(bins.unique()):
            selected = bins.eq(name).to_numpy(dtype=bool)
            count = int(selected.sum())
            if count < min_group_size:
                summaries[str(name)] = {"customers": count, "suppressed": True}
                continue
            summary = _classification_summary(y[selected], p[selected])
            summary["global_ranking_errors"] = {
                k: {
                    "customers_in_queue": int((selected & mask).sum()),
                    "negatives_in_queue": int((selected & mask & (y == 0)).sum()),
                    "positives_outside_queue": int((selected & ~mask & (y == 1)).sum()),
                }
                for k, mask in queues.items()
            }
            summaries[str(name)] = summary
        groups[column] = {
            "bin_edges": [None if np.isinf(x) else x for x in edges],
            "last_edge": "unbounded",
            "groups": summaries,
        }
    return {
        "classification_threshold": 0.5,
        "high_confidence_cutoffs": {
            "false_positive_min_score": 0.8,
            "false_negative_max_score": 0.2,
        },
        "min_group_size": min_group_size,
        "classification_errors_at_0_5": overall,
        "ranking_errors_at_k": ranking_errors,
        "data_quality_groups": groups,
    }
