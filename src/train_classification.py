"""Classification model training and evaluation module.

Derives performance categories (Low, Medium, High) from final grades,
performs cross-validation model selection on training data, compares against
a DummyClassifier baseline, evaluates on held-out test data, saves the fitted
pipeline artifact, and generates evaluation visualizations.
"""

import logging
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CLASS_LABELS,
    CLASSIFICATION_MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.data_loader import get_features_and_targets, load_dataset
from src.evaluate import (
    evaluate_classification,
    generate_confusion_matrix_plot,
)
from src.preprocessing import create_preprocessor

logger = logging.getLogger(__name__)


def train_classification_models(
    save_model: bool = True,
    save_plot: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Execute complete training workflow for student performance classification.

    Follows strict best practices:
    - Independent target derivation (Low / Medium / High).
    - Early-prediction feature matrix without G1, G2, or G3.
    - Stratified train/test splitting before preprocessing fit.
    - Stratified 5-fold cross-validation on training data for model selection.
    - Full evaluation on held-out test data (accuracy, macro/weighted F1, confusion matrix).
    - Joblib pipeline persistence.

    Args:
        save_model: Whether to persist the winning pipeline to disk.
        save_plot: Whether to generate and save confusion matrix heatmap.

    Returns:
        Tuple of (fitted_pipeline, evaluation_metrics_dict).
    """
    logger.info("Loading dataset for classification training...")
    df = load_dataset()
    X, _, y_clf = get_features_and_targets(df)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_clf
    )
    logger.info("Split dataset: %d train rows, %d test rows.", len(X_train), len(X_test))

    # Candidate models for cross-validation on training data
    candidates = {
        "LogisticRegression_C1": LogisticRegression(
            C=1.0, max_iter=500, random_state=RANDOM_SEED
        ),
        "LogisticRegression_C01": LogisticRegression(
            C=0.1, max_iter=500, random_state=RANDOM_SEED
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=5, random_state=RANDOM_SEED
        ),
        "KNeighbors": KNeighborsClassifier(n_neighbors=7),
    }

    # Baseline model
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_pipe = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", baseline_model),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Cross-validation comparison strictly on X_train
    logger.info("Running 5-fold Stratified cross-validation for classifier selection...")
    cv_results = {}
    best_candidate_name = None
    best_cv_f1 = -1.0

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
            cv=skf,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
        )
        mean_acc = float(np.mean(scores["test_accuracy"]))
        mean_f1_macro = float(np.mean(scores["test_f1_macro"]))
        mean_f1_weighted = float(np.mean(scores["test_f1_weighted"]))

        cv_results[name] = {
            "cv_accuracy": round(mean_acc, 4),
            "cv_f1_macro": round(mean_f1_macro, 4),
            "cv_f1_weighted": round(mean_f1_weighted, 4),
        }
        logger.info("Candidate %s: CV Acc=%.4f, CV F1-Macro=%.4f", name, mean_acc, mean_f1_macro)

        # Select by highest macro F1
        if mean_f1_macro > best_cv_f1:
            best_cv_f1 = mean_f1_macro
            best_candidate_name = name

    logger.info("Selected best classification model based on CV F1-Macro: %s", best_candidate_name)

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

    best_test_metrics = evaluate_classification(y_test.values, y_pred_best, labels=CLASS_LABELS)
    baseline_test_metrics = evaluate_classification(y_test.values, y_pred_base, labels=CLASS_LABELS)

    results = {
        "selected_model": best_candidate_name,
        "cv_comparison": cv_results,
        "test_metrics": best_test_metrics,
        "baseline_metrics": baseline_test_metrics,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_labels": CLASS_LABELS,
    }

    if save_model:
        CLASSIFICATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipe, CLASSIFICATION_MODEL_PATH)
        logger.info("Saved classification pipeline to %s", CLASSIFICATION_MODEL_PATH)

    if save_plot:
        generate_confusion_matrix_plot(best_test_metrics["confusion_matrix"], CLASS_LABELS)
        logger.info("Generated confusion matrix plot.")

    return best_pipe, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipe, res = train_classification_models(save_model=True, save_plot=True)
    print("\n--- Classification Training Results ---")
    print(f"Selected Model:   {res['selected_model']}")
    print(f"Test Accuracy:    {res['test_metrics']['accuracy']} (Baseline: {res['baseline_metrics']['accuracy']})")
    print(f"Test F1-Macro:    {res['test_metrics']['f1_macro']} (Baseline: {res['baseline_metrics']['f1_macro']})")
    print(f"Test F1-Weighted: {res['test_metrics']['f1_weighted']} (Baseline: {res['baseline_metrics']['f1_weighted']})")
