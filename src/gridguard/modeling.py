"""Classifier specifications, reproducible metrics, and risk ranking."""

from numbers import Real

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.features import review_signal, risk_band


def candidate_models(seed: int) -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("classifier", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=1500)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=12,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def ranking_order(
    scores: np.ndarray, customer_ids=None, *, score_domain="probability"
) -> np.ndarray:
    """Descending scores, then ID; legacy probability validation stays default.

    The explicit unbounded domain supports finite anomaly scores without
    normalizing them into apparent probabilities. No calibration is implied.
    """
    scores = np.asarray(scores, dtype=float)
    if score_domain not in ("probability", "unbounded"):
        raise ValueError("score_domain must be probability or unbounded.")
    invalid = scores.ndim != 1 or not np.isfinite(scores).all()
    if score_domain == "probability" and (invalid or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Expected a one-dimensional array of finite probabilities in [0, 1].")
    if invalid:
        raise ValueError("Expected one-dimensional finite scores.")
    if customer_ids is None:
        return np.argsort(-scores, kind="stable")
    ids = pd.Series(customer_ids, dtype="string").reset_index(drop=True).str.strip()
    if len(ids) != len(scores) or ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Expected aligned, nonempty, unique customer IDs.")
    return np.lexsort((ids.to_numpy(dtype=str), -scores))


def _metric_inputs(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected aligned one-dimensional binary labels and probabilities.")
    return labels.astype(int), scores


def ranking_metrics(
    y_true, scores, k: int = 50, *, customer_ids=None, score_domain="probability"
) -> dict:
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    labels, scores = _metric_inputs(y_true, scores)
    order = ranking_order(scores, customer_ids, score_domain=score_domain)
    count = min(k, len(labels))
    found = int(labels[order[:count]].sum())
    precision = found / count if count else 0.0
    positives = int(labels.sum())
    prevalence = positives / len(labels) if len(labels) else 0.0
    return {
        "evaluated_k": count,
        "positives_found": found,
        "precision": precision,
        "recall": found / positives if positives else None,
        "lift": precision / prevalence if prevalence else None,
    }


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 50, *, customer_ids=None):
    return ranking_metrics(y_true, scores, k, customer_ids=customer_ids)["precision"]


def score_metrics(
    y_true, scores, *, customer_ids, capacities=(50, 100, 200), score_domain="probability"
):
    """Threshold-free metrics shared by probability and anomaly evaluations."""
    scores = np.asarray(scores, dtype=float)
    ranking_order(scores, customer_ids, score_domain=score_domain)
    result = {"average_precision": None, "roc_auc": None, "ranking_at_k": {}}
    if y_true is None:
        return result
    labels, scores = _metric_inputs(y_true, scores)
    if not len(labels):
        return result
    result["ranking_at_k"] = {
        str(k): ranking_metrics(
            labels, scores, k, customer_ids=customer_ids, score_domain=score_domain
        )
        for k in capacities
    }
    if len(labels):
        result["average_precision"] = (
            float(average_precision_score(labels, scores)) if labels.any() else None
        )
        result["roc_auc"] = (
            float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
        )
    return result


def classification_metrics(y_true, scores, *, threshold=0.5, customer_ids=None):
    """Confusion counts and rates at a declared threshold; equality is positive.

    Undefined rates are null. Legacy model_metrics retains its established
    zero_division=0 precision/recall/F1 policy and historical output keys.
    """
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Threshold must be finite and in [0, 1].")
    ranking_order(scores, customer_ids)
    count = len(scores)
    result = {
        "threshold": float(threshold),
        "customers": count,
        "positive_customers": None,
        "confusion_matrix": None,
        **{
            key: None
            for key in (
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "false_negative_rate",
            )
        },
    }
    if y_true is None:
        return result
    labels, values = _metric_inputs(y_true, scores)
    matrix = (
        confusion_matrix(labels, values >= threshold, labels=[0, 1])
        if count
        else np.zeros((2, 2), dtype=int)
    )
    tn, fp, fn, tp = matrix.ravel().tolist()
    return {
        **result,
        "positive_customers": int(labels.sum()),
        "confusion_matrix": matrix.tolist(),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
    }


def model_metrics(y_true: np.ndarray, scores: np.ndarray, *, customer_ids=None) -> dict:
    labels, scores = _metric_inputs(y_true, scores)
    ranking_order(scores, customer_ids)
    if not len(labels):
        raise ValueError("Metrics require at least one customer.")
    predicted = (scores >= 0.5).astype(int)
    matrix = np.asarray(
        classification_metrics(labels, scores, customer_ids=customer_ids)["confusion_matrix"]
    )
    ranking = score_metrics(labels, scores, customer_ids=customer_ids)
    return {
        "average_precision": ranking["average_precision"],
        "roc_auc": ranking["roc_auc"],
        **{
            f"precision_at_{k}": ranking["ranking_at_k"][str(k)]["precision"]
            for k in (50, 100, 200)
        },
        "ranking_at_k": ranking["ranking_at_k"],
        "precision_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "f1_0_5": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "evaluated_customers": int(len(labels)),
        "positive_prevalence": float(labels.mean()),
    }


def rank_customers(
    ids: pd.Series, features: pd.DataFrame, probabilities: np.ndarray
) -> pd.DataFrame:
    order = ranking_order(probabilities, ids)
    if len(features) != len(ids):
        raise ValueError("Features and customer IDs must be aligned.")
    scores = np.rint(np.clip(probabilities, 0, 1) * 100).astype(int)
    result = pd.DataFrame(
        {
            "customer_id": ids.to_numpy(dtype=str),
            "risk_score": scores,
            "model_score": np.asarray(probabilities, dtype=float),
            "risk_band": [risk_band(score) for score in scores],
            "review_signal": features.apply(review_signal, axis=1).to_numpy(),
        }
    )
    return result.iloc[order].reset_index(drop=True)
