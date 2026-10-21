"""
Rainfall Classification Module for WeatherWise Analytics.
Trains, persists, and evaluates Logistic Regression, Decision Tree, and Random Forest models.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    DECISION_TREE_MODEL_PATH,
    LOGISTIC_REGRESSION_MODEL_PATH,
    MODELS_DIR,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_SEED,
)
from src.utils import logger


def build_models() -> Dict[str, Any]:
    """Instantiates the three required classification models."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                random_state=RANDOM_SEED,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=15,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1
        )
    }
    return models


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Fits all classification pipelines on training data only and optionally persists them.
    """
    models = build_models()
    trained_models = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    path_map = {
        "Logistic Regression": LOGISTIC_REGRESSION_MODEL_PATH,
        "Decision Tree": DECISION_TREE_MODEL_PATH,
        "Random Forest": RANDOM_FOREST_MODEL_PATH,
    }

    for name, model in models.items():
        logger.info(f"Training classifier: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

        if save_models and name in path_map:
            joblib.dump(model, path_map[name])
            logger.info(f"Saved {name} model to {path_map[name]}")

    return trained_models


def load_classification_models() -> Dict[str, Any]:
    """Loads previously trained model artifacts if present on disk."""
    path_map = {
        "Logistic Regression": LOGISTIC_REGRESSION_MODEL_PATH,
        "Decision Tree": DECISION_TREE_MODEL_PATH,
        "Random Forest": RANDOM_FOREST_MODEL_PATH,
    }
    loaded = {}
    for name, path in path_map.items():
        if path.exists():
            try:
                loaded[name] = joblib.load(path)
            except Exception as e:
                logger.error(f"Failed to load {name} from {path}: {e}")
    return loaded


def predict_next_day_rain(
    model: Any,
    temp_max: float,
    temp_min: float,
    temp_diff: float,
    month: int,
    day: int,
    week: int
) -> Dict[str, Any]:
    """
    Predicts probability and label for next-day rainfall using a single observation vector.
    """
    input_df = pd.DataFrame([{
        "Temp Max": float(temp_max),
        "Temp Min": float(temp_min),
        "Temp_diff": float(temp_diff),
        "Month": int(month),
        "Day": int(day),
        "Week": int(week),
    }])

    pred_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    prob_no_rain = float(probabilities[0])
    prob_rain = float(probabilities[1])

    return {
        "prediction_label": "Rain" if pred_class == 1 else "No Rain",
        "prediction_class": pred_class,
        "probability_rain": prob_rain,
        "probability_no_rain": prob_no_rain,
    }
