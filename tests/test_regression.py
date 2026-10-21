"""
Unit tests for next-day maximum temperature regression.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.regression import (
    build_regression_models,
    predict_next_day_temperature,
    train_regression_models,
)


def test_build_regression_models():
    """Validates regression model architectures."""
    models = build_regression_models()
    assert "Random Forest Regressor" in models
    assert "Linear Regression" in models
    assert "Decision Tree Regressor" in models


def test_train_and_predict_regression():
    """Validates training and temperature inference."""
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        "Temp Max": np.random.uniform(20, 42, n),
        "Temp Min": np.random.uniform(10, 25, n),
        "Temp_diff": np.random.uniform(5, 17, n),
        "Month": np.random.randint(1, 13, n),
        "Day": np.random.randint(1, 29, n),
        "Week": np.random.randint(1, 53, n),
    })
    # Synthetic target with strong correlation to current Temp Max
    y = X["Temp Max"] * 0.95 + np.random.normal(0, 1, n)

    trained = train_regression_models(X, y, save_primary_model=False)
    rf_reg = trained["Random Forest Regressor"]

    pred_temp = predict_next_day_temperature(
        rf_reg,
        temp_max=32.0,
        temp_min=20.0,
        temp_diff=12.0,
        month=5,
        day=10,
        week=19
    )

    assert isinstance(pred_temp, float)
    assert 15.0 <= pred_temp <= 45.0
