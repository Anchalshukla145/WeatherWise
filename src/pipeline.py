"""
End-to-End Pipeline Script for WeatherWise Analytics.
Executes data loading, cleaning, target construction, model training,
evaluation, persistence, metric exports, and academic report generation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict
import pandas as pd

from config import (
    CLASSIFICATION_METRICS_PATH,
    CLASSIFICATION_TEST_PREDICTIONS_PATH,
    MODEL_COMPARISON_CSV_PATH,
    REGRESSION_METRICS_PATH,
    REGRESSION_TEST_PREDICTIONS_PATH,
)
from src.classification import train_classification_models
from src.data_loader import load_dataset
from src.evaluation import (
    build_comparison_dataframe,
    evaluate_all_classifiers,
    evaluate_all_regressors,
)
from src.preprocessing import (
    construct_ml_dataset,
    prepare_classification_data,
    prepare_regression_data,
)
from src.regression import train_regression_models
from src.report_generator import generate_all_reports
from src.utils import logger, save_json_file


def run_full_pipeline() -> Dict[str, Any]:
    """
    Executes the entire data science and machine learning workflow:
    1. Load and clean historical Delhi weather dataset
    2. Construct chronological next-day targets
    3. Perform chronological train/test splits
    4. Fit and serialize 3 classification models
    5. Fit and serialize 3 regression models
    6. Evaluate models on unseen test period
    7. Save evaluation metrics JSON and CSV tables
    8. Generate academic reports (PDF and DOCX)
    """
    logger.info("Starting WeatherWise Analytics full pipeline execution...")

    # 1. Load and clean dataset
    cleaned_df, audit_stats = load_dataset()
    logger.info(f"Loaded and cleaned {len(cleaned_df)} records.")

    # 2. Construct ML targets
    ml_df = construct_ml_dataset(cleaned_df)

    # 3. Prepare data splits
    clf_data = prepare_classification_data(ml_df)
    reg_data = prepare_regression_data(ml_df)

    # 4. Train classifiers
    clf_models = train_classification_models(clf_data["X_train"], clf_data["y_train"], save_models=True)

    # 5. Train regressors
    reg_models = train_regression_models(reg_data["X_train"], reg_data["y_train"], save_primary_model=True)

    # 6. Evaluate classifiers
    clf_results = evaluate_all_classifiers(clf_models, clf_data["X_test"], clf_data["y_test"])

    # 7. Evaluate regressors
    reg_results = evaluate_all_regressors(reg_models, reg_data["X_test"], reg_data["y_test"])

    # 8. Build and save comparison DataFrame
    df_comp = build_comparison_dataframe(clf_results)
    MODEL_COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)
    logger.info(f"Saved model comparison table to {MODEL_COMPARISON_CSV_PATH}")

    # 9. Save metrics JSON files
    save_json_file(clf_results, CLASSIFICATION_METRICS_PATH)
    save_json_file(reg_results, REGRESSION_METRICS_PATH)
    logger.info(f"Saved classification metrics to {CLASSIFICATION_METRICS_PATH}")
    logger.info(f"Saved regression metrics to {REGRESSION_METRICS_PATH}")

    # 10. Save test predictions CSV for exploration
    test_pred_df = clf_data["test_df"].copy()
    for name, res in clf_results.items():
        test_pred_df[f"Pred_Class_{name}"] = res["y_pred"]
        test_pred_df[f"Pred_Prob_{name}"] = res["y_prob"]
    test_pred_df.to_csv(CLASSIFICATION_TEST_PREDICTIONS_PATH, index=False)

    reg_test_df = reg_data["test_df"].copy()
    for name, rres in reg_results.items():
        reg_test_df[f"Pred_TempMax_{name}"] = reg_models[name].predict(reg_data["X_test"])
    reg_test_df.to_csv(REGRESSION_TEST_PREDICTIONS_PATH, index=False)

    # 11. Generate PDF and DOCX reports
    reports = generate_all_reports(audit_stats, clf_results, reg_results)
    logger.info(f"Generated reports: {reports}")

    logger.info("Pipeline execution completed successfully!")

    return {
        "audit_stats": audit_stats,
        "clf_results": clf_results,
        "reg_results": reg_results,
        "df_comp": df_comp,
        "reports": reports,
    }


if __name__ == "__main__":
    from typing import Any, Dict
    run_full_pipeline()
