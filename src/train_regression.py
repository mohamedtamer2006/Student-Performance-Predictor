"""Regression model training and evaluation module.

Implements model selection via cross-validation strictly on training data,
compares candidates against a DummyRegressor baseline, fits the final pipeline,
evaluates on held-out test data, and persists the pipeline artifact.
"""

import logging
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_SEED,
    REGRESSION_MODEL_PATH,
    TEST_SIZE,
)
from src.data_loader import get_features_and_targets, load_dataset
from src.evaluate import evaluate_regression
from src.preprocessing import create_preprocessor

logger = logging.getLogger(__name__)


def train_regression_models(
    save_model: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Execute complete training workflow for student final grade regression.

    Strictly adheres to:
    - Feature/target separation without forbidden features (G1, G2, G3).
    - Train/test splitting before any preprocessing fit.
    - 5-fold cross-validation inside training set for model selection.
    - Evaluation on held-out test set.
    - Pipeline persistence with Joblib.

    Args:
        save_model: Whether to save the fitted pipeline to disk.

    Returns:
        Tuple of (fitted_pipeline, evaluation_metrics_dict).
    """
    logger.info("Loading dataset for regression training...")
    df = load_dataset()
    X, y_reg, y_clf = get_features_and_targets(df)

    # Train/test split (stratified by classification target for consistent splits across tasks)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_clf
    )
    logger.info("Split dataset: %d train rows, %d test rows.", len(X_train), len(X_test))

    # Candidate models for cross-validation on training data only
    candidates = {
        "DecisionTree": DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=5, random_state=RANDOM_SEED
        ),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_SEED),
        "LinearRegression": LinearRegression(),
        "KNeighbors": KNeighborsRegressor(n_neighbors=7),
    }

    # Baseline model
    baseline_model = DummyRegressor(strategy="mean")
    baseline_pipe = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", baseline_model),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Cross-validation comparison strictly on X_train
    logger.info("Running 5-fold cross-validation for regression model selection...")
    cv_results = {}
    best_candidate_name = None
    best_cv_mae = float("inf")

    for name, model in candidates.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("model", model),
            ]
        )
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["neg_mean_absolute_error", "r2", "neg_root_mean_squared_error"],
        )
        mean_mae = float(-np.mean(scores["test_neg_mean_absolute_error"]))
        mean_r2 = float(np.mean(scores["test_r2"]))
        mean_rmse = float(-np.mean(scores["test_neg_root_mean_squared_error"]))

        cv_results[name] = {
            "cv_mae": round(mean_mae, 4),
            "cv_r2": round(mean_r2, 4),
            "cv_rmse": round(mean_rmse, 4),
        }
        logger.info("Candidate %s: CV MAE=%.4f, CV R2=%.4f", name, mean_mae, mean_r2)

        # Minimize MAE for model selection
        if mean_mae < best_cv_mae:
            best_cv_mae = mean_mae
            best_candidate_name = name

    logger.info("Selected best regression model based on CV MAE: %s", best_candidate_name)

    # Build and fit the winning pipeline on full X_train
    best_pipe = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", candidates[best_candidate_name]),
        ]
    )

    best_pipe.fit(X_train, y_train)
    baseline_pipe.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred_best = best_pipe.predict(X_test)
    y_pred_base = baseline_pipe.predict(X_test)

    best_test_metrics = evaluate_regression(y_test.values, y_pred_best)
    baseline_test_metrics = evaluate_regression(y_test.values, y_pred_base)

    results = {
        "selected_model": best_candidate_name,
        "cv_comparison": cv_results,
        "test_metrics": best_test_metrics,
        "baseline_metrics": baseline_test_metrics,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    if save_model:
        REGRESSION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipe, REGRESSION_MODEL_PATH)
        logger.info("Saved regression pipeline to %s", REGRESSION_MODEL_PATH)

    return best_pipe, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipe, res = train_regression_models(save_model=True)
    print("\n--- Regression Training Results ---")
    print(f"Selected Model: {res['selected_model']}")
    print(f"Test MAE:       {res['test_metrics']['mae']} (Baseline: {res['baseline_metrics']['mae']})")
    print(f"Test RMSE:      {res['test_metrics']['rmse']} (Baseline: {res['baseline_metrics']['rmse']})")
    print(f"Test R2:        {res['test_metrics']['r2']} (Baseline: {res['baseline_metrics']['r2']})")
