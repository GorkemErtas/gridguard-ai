"""Classifier specifications, reproducible metrics, and risk ranking."""

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


def ranking_order(scores: np.ndarray, customer_ids=None) -> np.ndarray:
    """Rank raw probabilities; ties use customer ID (or stable input order)."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected a one-dimensional array of finite probabilities in [0, 1].")
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


def ranking_metrics(y_true, scores, k: int = 50, *, customer_ids=None) -> dict:
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    labels, scores = _metric_inputs(y_true, scores)
    order = ranking_order(scores, customer_ids)
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


def model_metrics(y_true: np.ndarray, scores: np.ndarray, *, customer_ids=None) -> dict:
    labels, scores = _metric_inputs(y_true, scores)
    ranking_order(scores, customer_ids)
    if not len(labels):
        raise ValueError("Metrics require at least one customer.")
    predicted = (scores >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "average_precision": float(average_precision_score(labels, scores)) if labels.any() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        **{f"precision_at_{k}": precision_at_k(labels, scores, k, customer_ids=customer_ids)
           for k in (50, 100, 200)},
        "ranking_at_k": {str(k): ranking_metrics(labels, scores, k, customer_ids=customer_ids)
                         for k in (50, 100, 200)},
        "precision_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "f1_0_5": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "evaluated_customers": int(len(labels)),
        "positive_prevalence": float(labels.mean()),
    }


def rank_customers(ids: pd.Series, features: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
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
