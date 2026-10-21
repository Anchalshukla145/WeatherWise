"""
Unit tests for classification model building, training, and inference.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import build_models, predict_next_day_rain, train_classification_models


def test_build_models_contains_required_classifiers():
    """Ensures Logistic Regression, Decision Tree, and Random Forest are instantiated."""
    models = build_models()
    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models


def test_train_and_predict_classification():
    """Tests fitting models on synthetic sample and generating valid probability predictions."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "Temp Max": np.random.uniform(20, 40, n),
        "Temp Min": np.random.uniform(5, 25, n),
        "Temp_diff": np.random.uniform(5, 18, n),
        "Month": np.random.randint(1, 13, n),
        "Day": np.random.randint(1, 29, n),
        "Week": np.random.randint(1, 53, n),
    })
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.8, 0.2]))

    trained = train_classification_models(X, y, save_models=False)
    assert len(trained) == 3

    # Test single-point inference with Random Forest
    rf_model = trained["Random Forest"]
    pred_res = predict_next_day_rain(
        rf_model,
        temp_max=34.5,
        temp_min=24.0,
        temp_diff=10.5,
        month=7,
        day=15,
        week=29
    )

    assert pred_res["prediction_class"] in [0, 1]
    assert pred_res["prediction_label"] in ["Rain", "No Rain"]
    assert 0.0 <= pred_res["probability_rain"] <= 1.0
    assert 0.0 <= pred_res["probability_no_rain"] <= 1.0
    assert pytest.approx(pred_res["probability_rain"] + pred_res["probability_no_rain"], 0.01) == 1.0
