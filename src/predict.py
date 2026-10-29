"""Inference and prediction module for student performance.

Provides safe artifact loading, input validation, and unified prediction
functions for both regression (final numerical grade) and classification
(Low / Medium / High performance category).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURE_COLUMNS,
    CLASS_DESCRIPTIONS,
    CLASSIFICATION_MODEL_PATH,
    REGRESSION_MODEL_PATH,
)
from src.preprocessing import validate_input_data

logger = logging.getLogger(__name__)


def load_regression_pipeline(model_path: Optional[Path] = None) -> Pipeline:
    """Load the trained regression pipeline from disk.

    Args:
        model_path: Path to joblib file. Defaults to config.REGRESSION_MODEL_PATH.

    Returns:
        Fitted Scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If pipeline file does not exist.
    """
    path = model_path or REGRESSION_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Regression model not found at '{path}'. "
            "Please train the model first by running: python -m src.train"
        )
    return joblib.load(path)


def load_classification_pipeline(model_path: Optional[Path] = None) -> Pipeline:
    """Load the trained classification pipeline from disk.

    Args:
        model_path: Path to joblib file. Defaults to config.CLASSIFICATION_MODEL_PATH.

    Returns:
        Fitted Scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If pipeline file does not exist.
    """
    path = model_path or CLASSIFICATION_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Classification model not found at '{path}'. "
            "Please train the model first by running: python -m src.train"
        )
    return joblib.load(path)


def predict_student(
    student_data: Union[pd.DataFrame, Dict[str, Any]],
    reg_pipeline: Optional[Pipeline] = None,
    clf_pipeline: Optional[Pipeline] = None
) -> Dict[str, Any]:
    """Generate both regression and classification predictions for a single student or batch.

    Args:
        student_data: Dictionary of feature values or a single-row DataFrame.
        reg_pipeline: Pre-loaded regression pipeline. If None, loaded from disk.
        clf_pipeline: Pre-loaded classification pipeline. If None, loaded from disk.

    Returns:
        Dictionary containing:
        - predicted_grade: float (clipped to valid 0-20 scale)
        - predicted_category: str ('Low', 'Medium', or 'High')
        - category_description: str
        - class_probabilities: Dict[str, float] (if model supports predict_proba)
        - input_features: validated dictionary of features
    """
    # Standardize input to DataFrame
    if isinstance(student_data, dict):
        df_input = pd.DataFrame([student_data])
    elif isinstance(student_data, pd.DataFrame):
        df_input = student_data.copy()
    else:
        raise TypeError(f"student_data must be a dict or DataFrame, got {type(student_data).__name__}")

    # Validate schema, ranges, categories, and column order
    df_clean = validate_input_data(df_input)

    # Load pipelines if not provided
    reg_pipe = reg_pipeline or load_regression_pipeline()
    clf_pipe = clf_pipeline or load_classification_pipeline()

    # Predict numerical grade
    raw_grade_preds = reg_pipe.predict(df_clean)
    # Clip to valid 0-20 Portuguese scale
    clipped_grades = np.clip(raw_grade_preds, 0.0, 20.0)

    # Predict performance class
    class_preds = clf_pipe.predict(df_clean)

    # Compute probabilities if supported
    probabilities = None
    if hasattr(clf_pipe, "predict_proba"):
        try:
            probs = clf_pipe.predict_proba(df_clean)[0]
            classes = clf_pipe.classes_
            probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
        except Exception:
            probabilities = None

    grade = round(float(clipped_grades[0]), 2)
    category = str(class_preds[0])
    description = CLASS_DESCRIPTIONS.get(category, "")

    return {
        "predicted_grade": grade,
        "predicted_category": category,
        "category_description": description,
        "class_probabilities": probabilities,
        "input_features": df_clean.iloc[0].to_dict(),
    }


def predict_batch(
    df: pd.DataFrame,
    reg_pipeline: Optional[Pipeline] = None,
    clf_pipeline: Optional[Pipeline] = None
) -> pd.DataFrame:
    """Predict performance for multiple student records.

    Args:
        df: Input DataFrame with student features.
        reg_pipeline: Pre-loaded regression pipeline.
        clf_pipeline: Pre-loaded classification pipeline.

    Returns:
        DataFrame with original features plus 'predicted_grade' and 'predicted_category'.
    """
    df_clean = validate_input_data(df)
    reg_pipe = reg_pipeline or load_regression_pipeline()
    clf_pipe = clf_pipeline or load_classification_pipeline()

    raw_grades = reg_pipe.predict(df_clean)
    clipped_grades = np.clip(raw_grades, 0.0, 20.0)
    classes = clf_pipe.predict(df_clean)

    result_df = df_clean.copy()
    result_df["predicted_grade"] = np.round(clipped_grades, 2)
    result_df["predicted_category"] = classes

    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_dataset, get_features_and_targets

    print("Testing prediction on sample student from dataset...")
    df_raw = load_dataset()
    X, _, _ = get_features_and_targets(df_raw)
    sample_student = X.iloc[0].to_dict()

    result = predict_student(sample_student)
    print("\nPrediction Result:")
    print(f"Predicted Grade:    {result['predicted_grade']} / 20.0")
    print(f"Predicted Category: {result['predicted_category']} ({result['category_description']})")
    print(f"Probabilities:      {result['class_probabilities']}")
