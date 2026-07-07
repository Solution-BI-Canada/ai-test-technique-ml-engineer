import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


def select_threshold(y_true, y_proba, target_recall: float = 0.95) -> float:
    """
    Select the highest threshold that keeps recall above the business target.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        return 0.5

    valid_indexes = np.where(recall[:-1] >= target_recall)[0]

    if len(valid_indexes) == 0:
        return 0.5

    return float(thresholds[valid_indexes[-1]])


def evaluate_classifier(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y_true, y_proba),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }