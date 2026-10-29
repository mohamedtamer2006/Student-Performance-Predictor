"""Unit and integration tests for regression and classification model training & evaluation.
"""

from pathlib import Path
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    CLASS_LABELS,
    CLASSIFICATION_MODEL_PATH,
    EVALUATION_RESULTS_PATH,
    REGRESSION_MODEL_PATH,
)
from src.evaluate import (
    evaluate_classification,
    evaluate_regression,
    load_evaluation_results,
)
from src.train_classification import train_classification_models
from src.train_regression import train_regression_models


def test_evaluate_regression_logic():
    """Verify regression metrics computation with known synthetic numbers."""
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    y_pred = np.array([10.0, 11.0, 15.0, 16.0])

    res = evaluate_regression(y_true, y_pred)
    assert "mae" in res
    assert "rmse" in res
    assert "r2" in res
    assert res["mae"] == 0.5
    assert res["r2"] > 0.8


def test_evaluate_classification_logic():
    """Verify multiclass classification evaluation with known synthetic labels."""
    y_true = np.array(["Low", "Medium", "High", "Medium"])
    y_pred = np.array(["Low", "Medium", "High", "High"])

    res = evaluate_classification(y_true, y_pred, labels=CLASS_LABELS)
    assert res["accuracy"] == 0.75
    assert 0.0 <= res["f1_macro"] <= 1.0
    assert len(res["confusion_matrix"]) == 3
    assert "Low" in res["classification_report"]


def test_trained_regression_pipeline():
    """Verify regression training produces fitted pipeline and valid test metrics."""
    pipe, res = train_regression_models(save_model=False)

    assert isinstance(pipe, Pipeline)
    assert res["selected_model"] is not None
    assert "test_metrics" in res
    assert res["test_metrics"]["mae"] > 0.0
    assert np.isfinite(res["test_metrics"]["r2"])
    assert "baseline_metrics" in res


def test_trained_classification_pipeline():
    """Verify classification training produces fitted pipeline and valid test metrics."""
    pipe, res = train_classification_models(save_model=False, save_plot=False)

    assert isinstance(pipe, Pipeline)
    assert res["selected_model"] is not None
    assert "test_metrics" in res
    assert 0.0 <= res["test_metrics"]["accuracy"] <= 1.0
    assert 0.0 <= res["test_metrics"]["f1_macro"] <= 1.0
    assert len(res["test_metrics"]["confusion_matrix"]) == 3


def test_saved_model_files_exist():
    """Verify serialized model files and report exist on disk."""
    assert REGRESSION_MODEL_PATH.exists()
    assert CLASSIFICATION_MODEL_PATH.exists()
    assert EVALUATION_RESULTS_PATH.exists()


def test_evaluation_results_schema():
    """Verify evaluation_results.json conforms to expected schema."""
    data = load_evaluation_results()

    assert "metadata" in data
    assert "regression" in data
    assert "classification" in data
    assert "selected_model" in data["regression"]
    assert "selected_model" in data["classification"]
    assert "test_metrics" in data["regression"]
    assert "test_metrics" in data["classification"]
