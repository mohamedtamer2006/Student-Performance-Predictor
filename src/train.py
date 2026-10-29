"""Master training script for the Student Performance System.

Orchestrates full training, validation, evaluation, and serialization
for both regression and classification pipelines. Produces the consolidated
'evaluation_results.json' artifact and confusion matrix plot.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.config import (
    CLASS_DESCRIPTIONS,
    CLASS_LABELS,
    EVALUATION_RESULTS_PATH,
    GRADE_BINS,
    RANDOM_SEED,
    TEST_SIZE,
    UCI_DATASET_URL,
    UCI_LANDING_PAGE,
)
from src.data_loader import load_dataset, summarize_dataset
from src.evaluate import save_evaluation_results
from src.train_classification import train_classification_models
from src.train_regression import train_regression_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_training_pipeline() -> Dict[str, Any]:
    """Execute complete end-to-end training and evaluation workflow.

    Returns:
        Consolidated dictionary of metadata and offline evaluation metrics.
    """
    logger.info("Starting Student Performance end-to-end training workflow...")

    # 1. Dataset verification and summary
    raw_df = load_dataset()
    data_summary = summarize_dataset(raw_df)

    # 2. Train regression pipeline
    logger.info("--- Training Regression Model ---")
    reg_pipe, reg_results = train_regression_models(save_model=True)

    # 3. Train classification pipeline
    logger.info("--- Training Classification Model ---")
    clf_pipe, clf_results = train_classification_models(save_model=True, save_plot=True)

    # 4. Consolidate results
    master_results = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": "UCI Student Performance (Mathematics)",
            "dataset_url": UCI_LANDING_PAGE,
            "dataset_download_url": UCI_DATASET_URL,
            "random_seed": RANDOM_SEED,
            "test_split_ratio": TEST_SIZE,
            "dataset_summary": data_summary,
            "classification_boundaries": {
                "bins": GRADE_BINS,
                "labels": CLASS_LABELS,
                "descriptions": CLASS_DESCRIPTIONS,
            },
        },
        "regression": reg_results,
        "classification": clf_results,
    }

    # 5. Persist evaluation report
    save_path = save_evaluation_results(master_results, EVALUATION_RESULTS_PATH)
    logger.info("Consolidated evaluation report saved to: %s", save_path)

    return master_results


if __name__ == "__main__":
    results = run_full_training_pipeline()
    print("\n" + "=" * 60)
    print("MASTER TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Regression Model:     {results['regression']['selected_model']}")
    print(f"  Test MAE:           {results['regression']['test_metrics']['mae']:.4f}")
    print(f"  Test RMSE:          {results['regression']['test_metrics']['rmse']:.4f}")
    print(f"  Test R²:            {results['regression']['test_metrics']['r2']:.4f}")
    print(f"Classification Model: {results['classification']['selected_model']}")
    print(f"  Test Accuracy:      {results['classification']['test_metrics']['accuracy']:.4f}")
    print(f"  Test F1 (Macro):    {results['classification']['test_metrics']['f1_macro']:.4f}")
    print(f"  Test F1 (Weighted): {results['classification']['test_metrics']['f1_weighted']:.4f}")
    print("=" * 60)
