"""Evaluation module for regression and classification models.

Computes standardized performance metrics, confusion matrices, per-class
breakdowns, and generates visualization artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
)

# Backward-compatible RMSE calculation for scikit-learn < 1.4 and >= 1.4
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from src.config import (
    CLASS_LABELS,
    CONFUSION_MATRIX_PATH,
    EVALUATION_RESULTS_PATH,
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate standard regression metrics on test predictions.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary containing MAE, RMSE, and R2 score.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate multiclass classification metrics on test predictions.

    Args:
        y_true: Ground truth category labels.
        y_pred: Model predictions.
        labels: Explicit category labels order. Defaults to config.CLASS_LABELS.

    Returns:
        Dictionary containing accuracy, macro & weighted precision/recall/F1,
        per-class report, and confusion matrix.
    """
    target_labels = labels or CLASS_LABELS

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, labels=target_labels, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, labels=target_labels, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=target_labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=target_labels).tolist()
    report = classification_report(y_true, y_pred, labels=target_labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "labels": target_labels,
        "classification_report": report,
    }


def generate_confusion_matrix_plot(
    cm: List[List[int]],
    labels: List[str],
    output_path: Optional[Path] = None
) -> Path:
    """Generate and save a confusion matrix heatmap image.

    Args:
        cm: 2D confusion matrix list.
        labels: Class labels in order.
        output_path: Destination image path. Defaults to config.CONFUSION_MATRIX_PATH.

    Returns:
        Path to saved figure.
    """
    out = output_path or CONFUSION_MATRIX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title("Classification Confusion Matrix (Test Set)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    return out


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Path:
    """Save model evaluation metrics dictionary to JSON format.

    Args:
        results: Dictionary containing regression & classification metrics.
        output_path: File path to save. Defaults to config.EVALUATION_RESULTS_PATH.

    Returns:
        Path to the saved JSON file.
    """
    out = output_path or EVALUATION_RESULTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return out


def load_evaluation_results(
    file_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Load offline evaluation results from JSON file.

    Args:
        file_path: Path to evaluation results JSON. Defaults to config.EVALUATION_RESULTS_PATH.

    Returns:
        Evaluation dictionary.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    path = file_path or EVALUATION_RESULTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found at {path}. Please train the models first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
