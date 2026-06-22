from typing import Any

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_positive_class_scores(
    model: Any,
    X_test: pd.DataFrame,
) -> list[float] | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1].tolist()

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test).tolist()

    return None


def validate_evaluation_inputs(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test length mismatch: {len(X_test)} != {len(y_test)}"
        )

    if X_test.empty:
        raise ValueError("X_test is empty.")

    if y_test.empty:
        raise ValueError("y_test is empty.")

    target_values = set(y_test.dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(f"Invalid target values for binary classification: {target_values}")


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    validate_evaluation_inputs(X_test, y_test)

    try:
        y_pred = model.predict(X_test)
        scores = get_positive_class_scores(model, X_test)
    except NotFittedError as exc:
        raise ValueError("Model must be fitted before evaluation.") from exc

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": None,
        "pr_auc": None,
    }

    if scores is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
        metrics["pr_auc"] = float(average_precision_score(y_test, scores))

    return metrics