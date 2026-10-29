"""Unit and integration tests for inference and prediction services.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.config import CLASS_LABELS
from src.data_loader import get_features_and_targets, load_dataset
from src.predict import (
    load_classification_pipeline,
    load_regression_pipeline,
    predict_batch,
    predict_student,
)


@pytest.fixture
def sample_student_dict() -> dict:
    """Fixture providing a realistic student feature dictionary."""
    df = load_dataset()
    X, _, _ = get_features_and_targets(df)
    return X.iloc[0].to_dict()


def test_load_pipelines_success():
    """Verify saved pipelines load without error."""
    reg_pipe = load_regression_pipeline()
    clf_pipe = load_classification_pipeline()

    assert reg_pipe is not None
    assert clf_pipe is not None


def test_load_pipeline_missing_file(tmp_path: Path):
    """Verify loading a nonexistent pipeline path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_regression_pipeline(tmp_path / "missing.joblib")


def test_predict_student_output_format(sample_student_dict: dict):
    """Verify single-student prediction returns the expected dictionary structure and types."""
    result = predict_student(sample_student_dict)

    assert "predicted_grade" in result
    assert "predicted_category" in result
    assert "category_description" in result
    assert "class_probabilities" in result

    # Numerical grade must be finite and within 0-20
    assert 0.0 <= result["predicted_grade"] <= 20.0

    # Categorical prediction must be one of Low, Medium, High
    assert result["predicted_category"] in CLASS_LABELS

    # Probabilities should sum to approximately 1.0
    if result["class_probabilities"] is not None:
        probs = result["class_probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-2


def test_predict_batch():
    """Verify batch inference appends expected prediction columns."""
    df = load_dataset()
    X, _, _ = get_features_and_targets(df)
    batch = X.head(5).copy()

    predictions = predict_batch(batch)

    assert "predicted_grade" in predictions.columns
    assert "predicted_category" in predictions.columns
    assert len(predictions) == 5
    assert predictions["predicted_category"].isin(CLASS_LABELS).all()


def test_predict_invalid_data_fails(sample_student_dict: dict):
    """Verify invalid input data in predict_student raises appropriate ValueError."""
    bad_data = sample_student_dict.copy()
    bad_data["age"] = -5

    with pytest.raises(ValueError):
        predict_student(bad_data)
