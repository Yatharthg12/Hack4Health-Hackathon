"""Consistent metrics required by the hackathon evaluation document."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    truth: Sequence[int],
    predictions: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    """Calculate aggregate, class-wise, confusion, and multiclass AUC metrics."""

    truth_array = np.asarray(truth)
    prediction_array = np.asarray(predictions)
    labels = np.arange(len(class_names))
    report = classification_report(
        truth_array,
        prediction_array,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(truth_array, prediction_array)),
        "balanced_accuracy": float(balanced_accuracy_score(truth_array, prediction_array)),
        "macro_precision": float(precision_score(truth_array, prediction_array, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth_array, prediction_array, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth_array, prediction_array, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth_array, prediction_array, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth_array, prediction_array, labels=labels).tolist(),
        "per_class": {name: report[name] for name in class_names},
        "support": int(len(truth_array)),
    }
    try:
        binary_truth = label_binarize(truth_array, classes=labels)
        result["roc_auc_macro_ovr"] = float(
            roc_auc_score(binary_truth, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        result["roc_auc_macro_ovr"] = None
    return result


def regression_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    target_names: Sequence[str],
) -> dict[str, object]:
    """Calculate all requested regression metrics per target and as aggregates."""

    truth = np.asarray(truth, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    per_target: dict[str, dict[str, float]] = {}
    for index, target in enumerate(target_names):
        mse = mean_squared_error(truth[:, index], predictions[:, index])
        per_target[target] = {
            "mae": float(mean_absolute_error(truth[:, index], predictions[:, index])),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(truth[:, index], predictions[:, index])),
            "explained_variance": float(explained_variance_score(truth[:, index], predictions[:, index])),
        }
    keys = ("mae", "mse", "rmse", "r2", "explained_variance")
    aggregate = {
        key: float(np.mean([metrics[key] for metrics in per_target.values()])) for key in keys
    }
    return {"per_target": per_target, "aggregate": aggregate, "support": int(len(truth))}

