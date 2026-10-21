"""
Unit tests for evaluation metrics, ROC curve calculations, and model ranking.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    build_comparison_dataframe,
    evaluate_classifier,
    rank_models_by_metric,
)


def test_evaluate_classifier_metrics():
    """Validates accuracy, precision, recall, f1, and confusion matrix calculation."""
    X_test = pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6, 7, 8]})
    y_test = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    # Classifier that predicts 1 when feat > 4
    class CustomModel:
        def predict(self, X):
            return (X["feat"] > 4).astype(int).values

        def predict_proba(self, X):
            p1 = (X["feat"] / 8).values
            return np.column_stack([1 - p1, p1])

    res = evaluate_classifier("CustomPerfect", CustomModel(), X_test, y_test)

    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1_score"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["confusion_matrix"]["true_positive"] == 4
    assert res["confusion_matrix"]["true_negative"] == 4
    assert res["confusion_matrix"]["false_positive"] == 0
    assert res["confusion_matrix"]["false_negative"] == 0


def test_model_ranking():
    """Validates ranking logic and trade-off summary extraction."""
    df_comp = pd.DataFrame([
        {"Model": "Random Forest", "F1 Score": 0.548, "Recall": 0.791, "Accuracy": 0.765},
        {"Model": "Decision Tree", "F1 Score": 0.536, "Recall": 0.824, "Accuracy": 0.745},
        {"Model": "Logistic Regression", "F1 Score": 0.514, "Recall": 0.804, "Accuracy": 0.728},
    ])

    best_m, best_s, _ = rank_models_by_metric(df_comp, "F1 Score")
    assert best_m == "Random Forest"
    assert best_s == 0.548

    best_m_rec, best_s_rec, _ = rank_models_by_metric(df_comp, "Recall")
    assert best_m_rec == "Decision Tree"
    assert best_s_rec == 0.824
