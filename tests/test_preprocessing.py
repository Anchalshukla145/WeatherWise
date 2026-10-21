"""
Unit tests for target construction, anti-leakage filters, and chronological splitting.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSIFICATION_FEATURES, TARGET_CLASSIFICATION, TARGET_REGRESSION
from src.preprocessing import construct_ml_dataset, prepare_classification_data, prepare_regression_data


def test_construct_ml_dataset_target_shift():
    """Verifies that next-day targets match day t+1 observations."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    sample_df = pd.DataFrame({
        "Date_Parsed": dates,
        "Temp Max": [20.0, 22.0, 21.0, 25.0, 24.0],
        "Temp Min": [5.0, 6.0, 7.0, 8.0, 7.5],
        "Temp_diff": [15.0, 16.0, 14.0, 17.0, 16.5],
        "Has_rained_Binary": [0, 1, 0, 0, 1],
        "Month": [1, 1, 1, 1, 1],
        "Day": [1, 2, 3, 4, 5],
        "Week": [1, 1, 1, 1, 1],
    })

    ml_df = construct_ml_dataset(sample_df)

    # Next-day rain target for index 0 should equal Has_rained_Binary at index 1 (which is 1)
    assert ml_df[TARGET_CLASSIFICATION].iloc[0] == 1
    assert ml_df[TARGET_CLASSIFICATION].iloc[1] == 0
    # Next-day Temp Max target for index 0 should equal Temp Max at index 1 (which is 22.0)
    assert ml_df[TARGET_REGRESSION].iloc[0] == 22.0
    # The last row must be NaN because day t+1 is not in the dataset
    assert pd.isna(ml_df[TARGET_CLASSIFICATION].iloc[-1])
    assert pd.isna(ml_df[TARGET_REGRESSION].iloc[-1])


def test_chronological_split_strict_ordering():
    """Verifies that chronological splitting guarantees test period comes strictly after train period."""
    dates = pd.date_range("2000-01-01", periods=100, freq="D")
    sample_df = pd.DataFrame({
        "Date_Parsed": dates,
        "Temp Max": [20.0] * 100,
        "Temp Min": [10.0] * 100,
        "Temp_diff": [10.0] * 100,
        "Month": dates.month,
        "Day": dates.day,
        "Week": [1] * 100,
        "Has_rained_Binary": [0, 1] * 50,
    })

    ml_df = construct_ml_dataset(sample_df)
    clf_data = prepare_classification_data(ml_df, split_ratio=0.8)

    train_dates = clf_data["train_df"]["Date_Parsed"]
    test_dates = clf_data["test_df"]["Date_Parsed"]

    assert train_dates.max() < test_dates.min(), "Leakage detected: Train dates overlap or exceed test dates!"
    assert set(clf_data["features"]) == set(CLASSIFICATION_FEATURES)
    assert "Rain" not in clf_data["features"]
    assert "Has_rained" not in clf_data["features"]
