"""Unit tests for preprocessing pipeline, imputation, scaling, and input validation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.config import ALL_FEATURE_COLUMNS
from src.data_loader import get_features_and_targets, load_dataset
from src.preprocessing import create_preprocessor, validate_input_data


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Fixture providing a clean sample DataFrame with all 30 features."""
    df = load_dataset()
    X, _, _ = get_features_and_targets(df)
    return X.head(10).copy()


def test_create_preprocessor_structure():
    """Verify ColumnTransformer is properly structured with numerical and categorical branches."""
    preprocessor = create_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in names
    assert "cat" in names


def test_preprocessor_fit_transform(sample_feature_df: pd.DataFrame):
    """Verify preprocessor transforms tabular DataFrame into clean numeric array."""
    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(sample_feature_df)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_feature_df)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_missing_values(sample_feature_df: pd.DataFrame):
    """Verify median and most_frequent imputers safely handle NaN inputs."""
    corrupted_df = sample_feature_df.copy()
    corrupted_df.loc[0, "age"] = np.nan
    corrupted_df.loc[1, "absences"] = np.nan
    corrupted_df.loc[2, "Mjob"] = np.nan

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(corrupted_df)

    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unknown_categories(sample_feature_df: pd.DataFrame):
    """Verify that unseen categorical labels at inference time do not raise errors."""
    preprocessor = create_preprocessor()
    preprocessor.fit(sample_feature_df)

    novel_df = sample_feature_df.copy()
    novel_df.loc[0, "Mjob"] = "astronaut"
    novel_df.loc[0, "reason"] = "teleportation"

    # Should not raise ValueError because handle_unknown='ignore'
    transformed = preprocessor.transform(novel_df)
    assert transformed.shape[0] == len(novel_df)
    assert not np.isnan(transformed).any()


def test_validate_input_data_success(sample_feature_df: pd.DataFrame):
    """Verify validation passes for standard records and returns proper column alignment."""
    validated = validate_input_data(sample_feature_df)
    assert list(validated.columns) == ALL_FEATURE_COLUMNS


def test_validate_input_data_missing_column(sample_feature_df: pd.DataFrame):
    """Verify validation raises ValueError if a mandatory feature is missing."""
    incomplete_df = sample_feature_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="Input is missing required features"):
        validate_input_data(incomplete_df)


def test_validate_input_data_out_of_bounds(sample_feature_df: pd.DataFrame):
    """Verify validation flags values violating domain bounds."""
    invalid_df = sample_feature_df.copy()
    invalid_df.loc[0, "age"] = 99  # Allowed age is 15-22

    with pytest.raises(ValueError, match="outside allowed range"):
        validate_input_data(invalid_df)


def test_validate_input_data_invalid_category(sample_feature_df: pd.DataFrame):
    """Verify validation flags invalid categorical values."""
    invalid_df = sample_feature_df.copy()
    invalid_df.loc[0, "sex"] = "NonBinary"  # Allowed is 'F' or 'M'

    with pytest.raises(ValueError, match="not in allowed categories"):
        validate_input_data(invalid_df)


def test_validate_empty_dataframe():
    """Verify validation rejects empty DataFrames."""
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        validate_input_data(pd.DataFrame())
