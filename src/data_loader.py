"""Data loading and ingestion module for the Student Performance dataset.

Handles reproducible automatic retrieval from UCI Machine Learning Repository,
local file fallback, schema validation, and safe feature/target extraction.
"""

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    CLASS_LABELS,
    DATA_FILE,
    FORBIDDEN_FEATURES,
    GRADE_BINS,
    NUMERICAL_FEATURES,
    REGRESSION_TARGET,
    UCI_DATASET_URL,
    UCI_LANDING_PAGE,
)

logger = logging.getLogger(__name__)


def download_dataset(destination_path: Optional[Path] = None, force: bool = False) -> Path:
    """Download the official UCI Student Performance dataset if not present locally.

    Args:
        destination_path: Path where student-mat.csv should be stored.
        force: If True, redownload even if the file exists.

    Returns:
        Path to the saved CSV file.

    Raises:
        RuntimeError: If download fails and no local dataset is found.
    """
    dest = destination_path or DATA_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info(f"Dataset already exists locally at: {dest}")
        return dest

    logger.info(f"Downloading Student Performance dataset from {UCI_DATASET_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        req = urllib.request.Request(UCI_DATASET_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            outer_zip_bytes = io.BytesIO(response.read())

        with zipfile.ZipFile(outer_zip_bytes) as outer_zip:
            # The UCI bundle contains 'student.zip'
            if "student.zip" in outer_zip.namelist():
                inner_zip_bytes = io.BytesIO(outer_zip.read("student.zip"))
                with zipfile.ZipFile(inner_zip_bytes) as inner_zip:
                    csv_content = inner_zip.read("student-mat.csv")
            elif "student-mat.csv" in outer_zip.namelist():
                csv_content = outer_zip.read("student-mat.csv")
            else:
                raise ValueError("student-mat.csv not found inside downloaded archive.")

        with open(dest, "wb") as f:
            f.write(csv_content)

        logger.info(f"Successfully downloaded and saved dataset to {dest}")
        return dest

    except Exception as exc:
        if dest.exists():
            logger.warning(f"Download failed ({exc}), but existing local file will be used: {dest}")
            return dest
        raise RuntimeError(
            f"Failed to automatically download dataset from {UCI_DATASET_URL}: {exc}\n"
            f"Please download the dataset manually from {UCI_LANDING_PAGE} and place 'student-mat.csv' in '{dest.parent}'."
        ) from exc


def load_dataset(file_path: Optional[Path] = None, auto_download: bool = True) -> pd.DataFrame:
    """Load the student performance dataset and validate its schema.

    Args:
        file_path: Path to student-mat.csv. Defaults to config.DATA_FILE.
        auto_download: Whether to attempt automatic download if missing.

    Returns:
        DataFrame containing the raw student performance data.

    Raises:
        FileNotFoundError: If the dataset file does not exist and auto_download is False or fails.
        ValueError: If essential columns or target are missing.
    """
    path = file_path or DATA_FILE

    if not path.exists():
        if auto_download:
            path = download_dataset(path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Enable auto_download or place student-mat.csv at that path."
            )

    # UCI student-mat.csv is semicolon-delimited
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] <= 1:
            # Fallback to comma if semicolon didn't split columns
            df = pd.read_csv(path, sep=",")
    except Exception as exc:
        raise ValueError(f"Could not read dataset CSV from {path}: {exc}") from exc

    # Validate essential target and features
    if REGRESSION_TARGET not in df.columns:
        raise ValueError(f"Required target column '{REGRESSION_TARGET}' not found in dataset columns.")

    missing_cols = [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols}")

    return df


def get_features_and_targets(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Strictly separate input features from targets, preventing data leakage.

    In the early-prediction scenario, previous grades ('G1', 'G2') and final
    grade ('G3') are strictly excluded from input features.

    Args:
        df: Input raw DataFrame.

    Returns:
        Tuple of (X, y_regression, y_classification):
        - X: DataFrame containing only permitted predictor features.
        - y_regression: Series containing numerical target G3 (0 to 20).
        - y_classification: Series containing categorical target ('Low', 'Medium', 'High').
    """
    # Defensive check: ensure forbidden features are not present in X
    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c not in FORBIDDEN_FEATURES]
    X = df[feature_cols].copy()

    for forbidden in FORBIDDEN_FEATURES:
        if forbidden in X.columns:
            raise ValueError(f"Data leakage detected! Forbidden column '{forbidden}' found in feature matrix X.")

    # Target extraction
    y_regression = df[REGRESSION_TARGET].astype(float).copy()

    # Classification target derivation
    # [-0.1, 9.5] -> Low, (9.5, 13.5] -> Medium, (13.5, 20.0] -> High
    y_classification = pd.cut(
        y_regression,
        bins=GRADE_BINS,
        labels=CLASS_LABELS,
        right=True
    ).astype(str)

    return X, y_regression, y_classification


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Generate an exploratory summary of the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing dataset summary statistics.
    """
    X, y_reg, y_clf = get_features_and_targets(df)
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_features": int(X.shape[1]),
        "n_categorical_features": len(CATEGORICAL_FEATURES),
        "n_numerical_features": len(NUMERICAL_FEATURES),
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "regression_target_stats": {
            "min": float(y_reg.min()),
            "max": float(y_reg.max()),
            "mean": round(float(y_reg.mean()), 2),
            "median": round(float(y_reg.median()), 2),
            "std": round(float(y_reg.std()), 2)
        },
        "classification_target_distribution": y_clf.value_counts().to_dict()
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Executing data loader module...")
    dataset = load_dataset()
    info = summarize_dataset(dataset)
    import json
    print(json.dumps(info, indent=2))
