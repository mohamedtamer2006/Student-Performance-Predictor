"""Unit tests for dataset loading, schema verification, and feature-target separation.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.config import (
    ALL_FEATURE_COLUMNS,
    CLASS_LABELS,
    DATA_FILE,
    FORBIDDEN_FEATURES,
    REGRESSION_TARGET,
)
from src.data_loader import (
    get_features_and_targets,
    load_dataset,
    summarize_dataset,
)


def test_dataset_loading_succeeds():
    """Verify that dataset loads properly and has non-empty rows and columns."""
    df = load_dataset(DATA_FILE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 395
    assert df.shape[1] == 33


def test_missing_dataset_path_raises_error(tmp_path: Path):
    """Verify that a nonexistent path without auto_download raises FileNotFoundError."""
    fake_path = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(fake_path, auto_download=False)


def test_required_columns_exist():
    """Verify all required predictors and target exist in the dataset."""
    df = load_dataset()
    assert REGRESSION_TARGET in df.columns
    for col in ALL_FEATURE_COLUMNS:
        assert col in df.columns


def test_feature_and_target_separation():
    """Verify separation produces X, y_reg, and y_clf with identical lengths."""
    df = load_dataset()
    X, y_reg, y_clf = get_features_and_targets(df)

    assert len(X) == len(df)
    assert len(y_reg) == len(df)
    assert len(y_clf) == len(df)
    assert X.shape[1] == len(ALL_FEATURE_COLUMNS)


def test_no_forbidden_target_columns_in_inputs():
    """Strictly assert no future grades (G1, G2, G3) appear in feature matrix X."""
    df = load_dataset()
    X, _, _ = get_features_and_targets(df)

    for forbidden in FORBIDDEN_FEATURES:
        assert forbidden not in X.columns, f"Data leakage: {forbidden} found in X!"


def test_regression_target_bounds():
    """Verify all regression targets are within the legitimate Portuguese scale [0, 20]."""
    df = load_dataset()
    _, y_reg, _ = get_features_and_targets(df)

    assert y_reg.min() >= 0.0
    assert y_reg.max() <= 20.0
    assert not y_reg.isnull().any()


def test_classification_target_distribution():
    """Verify classification categories only contain defined labels Low, Medium, High."""
    df = load_dataset()
    _, _, y_clf = get_features_and_targets(df)

    unique_classes = set(y_clf.unique())
    assert unique_classes.issubset(set(CLASS_LABELS))
    # All 3 classes should be represented
    assert set(CLASS_LABELS) == unique_classes


def test_summarize_dataset():
    """Verify summary helper returns correct structure and expected metrics."""
    df = load_dataset()
    summary = summarize_dataset(df)

    assert summary["n_rows"] == 395
    assert summary["n_features"] == 30
    assert summary["missing_values_total"] == 0
    assert "Low" in summary["classification_target_distribution"]
