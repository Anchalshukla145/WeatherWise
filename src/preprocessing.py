"""
Preprocessing and Feature Engineering Module for WeatherWise Analytics.
Constructs next-day predictive targets, enforces strict anti-leakage filters,
executes chronological train/test splits, and defines transformation pipelines.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CHRONOLOGICAL_SPLIT_RATIO,
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)
from src.utils import logger


def construct_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs next-day classification and regression targets from chronologically sorted data.
    
    Target Logic:
    - On day t: Observe features (Temp Max, Temp Min, Temp_diff, Month, Day, Week)
    - Next-day classification target: whether it will rain on day t+1
    - Next-day regression target: maximum temperature on day t+1
    
    Anti-Leakage Safeguards:
    - Shifts target backward by 1 record (-1) AFTER chronological sorting
    - Excludes current-day Rain and Has_rained from predictive features
    - Drops the final row because its next-day outcome is unknown
    """
    df_sorted = df.copy()
    if "Date_Parsed" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("Date_Parsed").reset_index(drop=True)

    # Classification target: rain status on day t+1
    if "Has_rained_Binary" in df_sorted.columns:
        df_sorted[TARGET_CLASSIFICATION] = df_sorted["Has_rained_Binary"].shift(-1)
    else:
        raise ValueError("Missing 'Has_rained_Binary' column for classification target construction.")

    # Regression target: Temp Max on day t+1
    if "Temp Max" in df_sorted.columns:
        df_sorted[TARGET_REGRESSION] = df_sorted["Temp Max"].shift(-1)
    else:
        raise ValueError("Missing 'Temp Max' column for regression target construction.")

    return df_sorted


def prepare_classification_data(
    df: pd.DataFrame,
    split_ratio: float = CHRONOLOGICAL_SPLIT_RATIO
) -> Dict[str, Any]:
    """
    Filters valid rows, validates features, and performs a strict chronological train-test split.
    """
    required_cols = CLASSIFICATION_FEATURES + [TARGET_CLASSIFICATION, "Date_Parsed"]
    valid_df = df.dropna(subset=required_cols).copy()
    valid_df = valid_df.sort_values("Date_Parsed").reset_index(drop=True)

    split_idx = int(len(valid_df) * split_ratio)
    train_df = valid_df.iloc[:split_idx].copy()
    test_df = valid_df.iloc[split_idx:].copy()

    X_train = train_df[CLASSIFICATION_FEATURES]
    y_train = train_df[TARGET_CLASSIFICATION].astype(int)

    X_test = test_df[CLASSIFICATION_FEATURES]
    y_test = test_df[TARGET_CLASSIFICATION].astype(int)

    logger.info(
        f"Classification data prepared: Train {len(train_df)} rows ({train_df['Date_Parsed'].min().date()} to {train_df['Date_Parsed'].max().date()}), "
        f"Test {len(test_df)} rows ({test_df['Date_Parsed'].min().date()} to {test_df['Date_Parsed'].max().date()})"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "features": CLASSIFICATION_FEATURES,
        "split_ratio": split_ratio,
        "split_date": str(test_df["Date_Parsed"].min().date()),
    }


def prepare_regression_data(
    df: pd.DataFrame,
    split_ratio: float = CHRONOLOGICAL_SPLIT_RATIO
) -> Dict[str, Any]:
    """
    Filters valid rows for temperature regression and performs a strict chronological train-test split.
    """
    required_cols = REGRESSION_FEATURES + [TARGET_REGRESSION, "Date_Parsed"]
    valid_df = df.dropna(subset=required_cols).copy()
    valid_df = valid_df.sort_values("Date_Parsed").reset_index(drop=True)

    split_idx = int(len(valid_df) * split_ratio)
    train_df = valid_df.iloc[:split_idx].copy()
    test_df = valid_df.iloc[split_idx:].copy()

    X_train = train_df[REGRESSION_FEATURES]
    y_train = train_df[TARGET_REGRESSION].astype(float)

    X_test = test_df[REGRESSION_FEATURES]
    y_test = test_df[TARGET_REGRESSION].astype(float)

    logger.info(
        f"Regression data prepared: Train {len(train_df)} rows, Test {len(test_df)} rows"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "features": REGRESSION_FEATURES,
        "split_ratio": split_ratio,
        "split_date": str(test_df["Date_Parsed"].min().date()),
    }


def create_scaler_pipeline(estimator: Any) -> Pipeline:
    """Wraps an estimator in a StandardScaler pipeline to prevent data leakage during scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])
