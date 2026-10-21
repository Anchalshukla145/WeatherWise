"""
Temperature Regression Module for WeatherWise Analytics.
Predicts next-day maximum temperature using Random Forest, Decision Tree, and Linear Regression.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import (
    MODELS_DIR,
    RANDOM_SEED,
    TEMP_REGRESSOR_MODEL_PATH,
)
from src.utils import logger


def build_regression_models() -> Dict[str, Any]:
    """Builds regression models for next-day maximum temperature."""
    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=6,
            min_samples_leaf=20,
            random_state=RANDOM_SEED
        ),
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ]),
    }


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_primary_model: bool = True
) -> Dict[str, Any]:
    """Fits all regression models on training data and persists the primary Random Forest model."""
    models = build_regression_models()
    trained = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger.info(f"Training regressor: {name}")
        model.fit(X_train, y_train)
        trained[name] = model

    if save_primary_model and "Random Forest Regressor" in trained:
        joblib.dump(trained["Random Forest Regressor"], TEMP_REGRESSOR_MODEL_PATH)
        logger.info(f"Saved primary temperature regressor to {TEMP_REGRESSOR_MODEL_PATH}")

    return trained


def load_primary_regressor() -> Optional[Any]:
    """Loads the serialized primary temperature regressor from disk."""
    if TEMP_REGRESSOR_MODEL_PATH.exists():
        try:
            return joblib.load(TEMP_REGRESSOR_MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load temperature regressor: {e}")
    return None


def predict_next_day_temperature(
    model: Any,
    temp_max: float,
    temp_min: float,
    temp_diff: float,
    month: int,
    day: int,
    week: int
) -> float:
    """Predicts next-day maximum temperature in °C given day t weather observations."""
    input_df = pd.DataFrame([{
        "Temp Max": float(temp_max),
        "Temp Min": float(temp_min),
        "Temp_diff": float(temp_diff),
        "Month": int(month),
        "Day": int(day),
        "Week": int(week),
    }])
    pred_temp = float(model.predict(input_df)[0])
    return round(pred_temp, 2)
