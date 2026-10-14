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

from gridguard.features import risk_band, review_signal


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


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 50) -> float:
    top_count = min(k, len(y_true))
    if top_count == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")[:top_count]
    return float(np.asarray(y_true)[order].mean())


def model_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int | list]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision_at_50": precision_at_k(labels, scores, k=50),
        "precision_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "f1_0_5": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "evaluated_customers": int(len(labels)),
        "positive_prevalence": float(labels.mean()),
    }


def rank_customers(ids: pd.Series, features: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
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
    return result.sort_values(
        ["model_score", "customer_id"], ascending=[False, True]
    ).reset_index(drop=True)
