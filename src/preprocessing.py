"""Preprocessing module for the Student Performance Prediction system.

Builds leak-free ColumnTransformer pipelines conforming to Scikit-learn standards,
handling imputation, scaling, and safe categorical one-hot encoding.
Also provides validation routines for single-student inference records.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    CATEGORICAL_OPTIONS,
    FEATURE_METADATA,
    NUMERICAL_FEATURES,
)


def create_preprocessor(
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    scale_numeric: bool = True
) -> ColumnTransformer:
    """Create a leak-free Scikit-learn ColumnTransformer for tabular features.

    Implements:
    - Numerical: Median imputation + (optional) Standard scaling.
    - Categorical: Most-frequent imputation + One-Hot Encoding with handle_unknown='ignore'.

    Args:
        categorical_features: List of categorical column names. Defaults to config.CATEGORICAL_FEATURES.
        numerical_features: List of numerical column names. Defaults to config.NUMERICAL_FEATURES.
        scale_numeric: Whether to apply StandardScaler to numerical features.

    Returns:
        Fittable ColumnTransformer pipeline component.
    """
    cat_cols = categorical_features or CATEGORICAL_FEATURES
    num_cols = numerical_features or NUMERICAL_FEATURES

    # Numerical sub-pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numerical_pipeline = Pipeline(steps=num_steps)

    # Categorical sub-pipeline
    # handle_unknown='ignore' ensures any novel categorical values at inference time are zero-encoded safely
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def validate_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """Validate inference input DataFrame against the expected schema and value domains.

    Args:
        input_df: DataFrame containing one or more student rows.

    Returns:
        Cleaned, type-aligned DataFrame with exact column ordering.

    Raises:
        ValueError: If required columns are missing or values violate domain constraints.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for inference input, got {type(input_df).__name__}")

    if input_df.empty:
        raise ValueError("Input DataFrame is empty; cannot make predictions on zero records.")

    # 1. Check for missing columns
    missing_cols = [col for col in ALL_FEATURE_COLUMNS if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required features: {missing_cols}")

    # Reorder columns to match training schema exactly
    df_clean = input_df[ALL_FEATURE_COLUMNS].copy()

    # 2. Validate numerical bounds
    for col in NUMERICAL_FEATURES:
        meta = FEATURE_METADATA.get(col, {})
        min_val = meta.get("min", -np.inf)
        max_val = meta.get("max", np.inf)

        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Numerical feature '{col}' must contain valid numbers: {exc}") from exc

        # Range checks
        out_of_bounds = df_clean[(df_clean[col] < min_val) | (df_clean[col] > max_val)]
        if not out_of_bounds.empty:
            bad_val = out_of_bounds[col].iloc[0]
            raise ValueError(
                f"Feature '{col}' value {bad_val} is outside allowed range [{min_val}, {max_val}]."
            )

    # 3. Validate categorical options
    for col in CATEGORICAL_FEATURES:
        valid_options = CATEGORICAL_OPTIONS.get(col, [])
        df_clean[col] = df_clean[col].astype(str)
        invalid_rows = df_clean[~df_clean[col].isin(valid_options)]
        if not invalid_rows.empty:
            bad_cat = invalid_rows[col].iloc[0]
            raise ValueError(
                f"Feature '{col}' value '{bad_cat}' is not in allowed categories: {valid_options}"
            )

    return df_clean
