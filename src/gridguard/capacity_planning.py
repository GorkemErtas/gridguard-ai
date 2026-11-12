"""Read-only capacity scenarios from saved rankings, never prospective forecasts.

Artifact loading and provenance validation belong to ``dashboard.load_run``.
This module accepts an in-memory cohort and never loads or changes a model.
"""

from collections.abc import Iterable
from numbers import Real

import numpy as np
import pandas as pd

from gridguard.modeling import ranking_metrics, ranking_order

PLAN_COLUMNS = [
    "requested_k",
    "evaluated_k",
    "positives_found",
    "precision",
    "recall",
    "lift",
    "random_expected_positives",
    "random_precision",
    "random_recall",
    "marginal_inspections",
    "marginal_positives",
    "marginal_yield",
]

FINANCIAL_NOTICE = (
    "Illustrative scenario only, not validated financial forecasts. "
    "Hypothetical recovery assumes the same user-supplied value per observed positive; "
    "historical labels do not establish recoverable losses or confirmed electricity theft."
)


def capacity_plan(
    predictions: pd.DataFrame, capacities: Iterable[int] = (50, 100, 200)
) -> pd.DataFrame:
    """Compare unfiltered top-K selections within one fixed customer cohort.

    Requires unique customer IDs and raw ``model_score`` values. Scores are
    ranking signals, not calibrated probabilities. ``true_label`` is optional;
    absent labels leave all outcome-based quantities unavailable.

    Capacities are nonnegative integers, sorted and deduplicated by requested K.
    Requests above cohort size are retained with ``evaluated_k`` capped. The
    first marginal interval starts at zero inspections; later intervals start
    at the previous row's actual capacity. Marginal yield is additional observed
    positives / additional inspections, undefined for a zero-width interval.

    At zero actual capacity, precision and lift are undefined. Recall is zero
    only if positives exist. Zero-positive cohorts have undefined recall/lift,
    zero observed positives and a zero random expected count. An empty labeled
    cohort follows those same conventions. No input frame is modified.
    """
    required = {"customer_id", "model_score"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(predictions))}")
    requested = list(capacities)
    if any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0
        for k in requested
    ):
        raise ValueError("Capacities must be nonnegative integers.")
    requested = sorted(set(int(k) for k in requested))
    scores, ids = predictions.model_score.to_numpy(), predictions.customer_id
    labeled = "true_label" in predictions
    labels = predictions.true_label.to_numpy() if labeled else None
    # Reuse the canonical label validation even for zero/empty capacity lists.
    first = ranking_metrics(labels, scores, 1, customer_ids=ids) if labeled else None
    if not labeled:
        ranking_order(scores, ids)
    positives = int(labels.sum()) if labeled else None
    size = len(predictions)
    prevalence = positives / size if labeled and size else None
    previous_k, previous_found = 0, 0
    rows = []
    for k in requested:
        actual = min(k, size)
        metrics = {"positives_found": None, "precision": None, "recall": None, "lift": None}
        if labeled:
            if actual:
                current = first if k == 1 else ranking_metrics(labels, scores, k, customer_ids=ids)
                metrics.update({key: current[key] for key in metrics})
            else:
                metrics.update(positives_found=0, recall=0.0 if positives else None)
        extra = actual - previous_k
        extra_found = metrics["positives_found"] - previous_found if labeled else None
        rows.append(
            {
                "requested_k": k,
                "evaluated_k": actual,
                **metrics,
                "random_expected_positives": actual * prevalence
                if size and labeled
                else (0.0 if labeled else None),
                "random_precision": prevalence if actual else None,
                "random_recall": actual / size if positives else None,
                "marginal_inspections": extra,
                "marginal_positives": extra_found,
                "marginal_yield": extra_found / extra if labeled and extra else None,
            }
        )
        previous_k, previous_found = actual, metrics["positives_found"]
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def financial_scenarios(
    plan: pd.DataFrame, *, inspection_cost: float, recovery_per_positive: float
) -> pd.DataFrame:
    """Illustrative arithmetic for a frame returned by :func:`capacity_plan`.

    All inputs use the same user-chosen currency units. Each actual inspection
    incurs the same cost; each observed positive is assigned the same hypothetical
    recovery. Random selection incurs the same inspection cost. Missing outcome
    counts stay unavailable, never inferred from scores. No ROI, probability
    calibration or validated recovery forecast is implied. The input is unchanged.
    """
    for name, value in (
        ("inspection_cost", inspection_cost),
        ("recovery_per_positive", recovery_per_positive),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number.")
    required = ["requested_k", "evaluated_k", "positives_found", "random_expected_positives"]
    if not set(required).issubset(plan):
        raise ValueError("Financial scenarios require a capacity_plan result.")
    result = plan[required].copy()
    # Float conversion preserves missing outcomes as NaN, including all-None columns.
    with np.errstate(over="ignore", invalid="ignore"):
        result["inspection_cost_total"] = result.evaluated_k.astype(float) * inspection_cost
        result["hypothetical_model_recovery"] = (
            result.positives_found.astype(float) * recovery_per_positive
        )
        result["hypothetical_random_recovery"] = (
            result.random_expected_positives.astype(float) * recovery_per_positive
        )
        result["illustrative_model_net"] = (
            result.hypothetical_model_recovery - result.inspection_cost_total
        )
        result["illustrative_random_net"] = (
            result.hypothetical_random_recovery - result.inspection_cost_total
        )
    if np.isinf(result.select_dtypes(include="number").to_numpy()).any():
        raise ValueError("Financial results exceed the finite numeric range.")
    result["notice"] = FINANCIAL_NOTICE
    return result
