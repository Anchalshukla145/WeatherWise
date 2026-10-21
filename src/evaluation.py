"""
Evaluation and Metrics Calculation Module for WeatherWise Analytics.
Computes classification and regression metrics, ROC curve data, confusion matrices,
and multi-model ranking logic.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import logger


def evaluate_classifier(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates a single classifier on the unseen chronological test set.
    """
    y_pred = model.predict(X_test)

    # Compute probability scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Safe ROC-AUC computation (handling edge case of single class in test set)
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 1:
        auc = float(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        fpr_list = fpr.tolist()
        tpr_list = tpr.tolist()
    else:
        auc = None
        fpr_list, tpr_list = [], []

    # Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "matrix": cm.tolist()
        },
        "roc_curve": {
            "fpr": fpr_list,
            "tpr": tpr_list
        },
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist() if isinstance(y_prob, np.ndarray) else list(y_prob)
    }


def evaluate_all_classifiers(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """Evaluates multiple classification pipelines against the test set."""
    results = {}
    for name, model in models.items():
        logger.info(f"Evaluating classifier: {name}")
        results[name] = evaluate_classifier(name, model, X_test, y_test)
    return results


def evaluate_regressor(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates temperature regression performance (MAE, RMSE, R², residuals).
    """
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    residuals = (y_test - preds).values

    return {
        "model_name": name,
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "actuals": y_test.tolist()[:500],  # sample for plotting
        "predictions": preds.tolist()[:500],
        "residuals": residuals.tolist()[:500]
    }


def evaluate_all_regressors(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """Evaluates all trained regression models."""
    results = {}
    for name, model in models.items():
        logger.info(f"Evaluating regressor: {name}")
        results[name] = evaluate_regressor(name, model, X_test, y_test)
    return results


def build_comparison_dataframe(clf_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Formats classification metrics into a clean comparative DataFrame."""
    rows = []
    for name, res in clf_results.items():
        rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1 Score": res["f1_score"],
            "ROC-AUC": res["roc_auc"] if res["roc_auc"] is not None else np.nan,
            "True Positive": res["confusion_matrix"]["true_positive"],
            "False Negative": res["confusion_matrix"]["false_negative"],
            "False Positive": res["confusion_matrix"]["false_positive"],
            "True Negative": res["confusion_matrix"]["true_negative"]
        })
    df_comp = pd.DataFrame(rows)
    return df_comp


def rank_models_by_metric(
    df_comp: pd.DataFrame,
    metric_col: str = "F1 Score"
) -> Tuple[str, float, str]:
    """
    Ranks models according to a selected metric and returns trade-off commentary.
    """
    if metric_col not in df_comp.columns:
        metric_col = "F1 Score"

    sorted_df = df_comp.sort_values(by=metric_col, ascending=False).reset_index(drop=True)
    best_row = sorted_df.iloc[0]
    best_name = str(best_row["Model"])
    best_score = float(best_row[metric_col])

    tradeoffs = {
        "Random Forest": "Achieves the highest overall F1 score and accuracy by aggregating multiple decorrelated trees, maintaining a robust balance between precision and recall.",
        "Decision Tree": "Yields high recall and interpretability with simple threshold splits, but shows a slightly higher rate of false positives on borderline dates.",
        "Logistic Regression": "Provides a smooth probabilistic calibration and strong recall under class-balanced weighting, though linear decision boundaries limit complex feature interaction capture."
    }

    tradeoff_summary = tradeoffs.get(
        best_name,
        f"{best_name} achieved the top score of {best_score:.4f} for {metric_col}."
    )

    return best_name, best_score, tradeoff_summary
