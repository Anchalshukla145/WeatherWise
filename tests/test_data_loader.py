"""
Unit tests for data loading, date parsing, cleaning, and quality audit.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import audit_data_quality, clean_weather_data, detect_and_load_csv, load_dataset
from src.utils import safe_to_float


def test_safe_to_float_conversions():
    """Validates numeric conversion logic including meteorological 'Tr' trace amounts."""
    assert safe_to_float(12.5) == 12.5
    assert safe_to_float("14.2") == 14.2
    assert safe_to_float("Tr") == 0.05
    assert safe_to_float("trace") == 0.05
    assert safe_to_float(None) is None
    assert safe_to_float(float("nan")) is None


def test_clean_weather_data_dates_and_sorting():
    """Verifies that clean_weather_data parses dates into datetimes and sorts chronologically."""
    sample_data = pd.DataFrame({
        "_id": [2, 1],
        "Date": ["02-01-1951", "01-01-1951"],
        "Rain": ["0", "Tr"],
        "Temp Max": [21.5, 20.0],
        "Temp Min": [4.0, 3.5],
        "Temp_diff": [17.5, 16.5],
        "Has_rained": ["NO", "YES"]
    })

    cleaned = clean_weather_data(sample_data)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date_Parsed"])
    assert cleaned["Date_Parsed"].iloc[0] < cleaned["Date_Parsed"].iloc[1]
    assert cleaned["Rain_Numeric"].iloc[0] == 0.05
    assert cleaned["Has_rained_Binary"].iloc[0] == 1
    assert cleaned["Has_rained_Binary"].iloc[1] == 0


def test_load_dataset_live():
    """Verifies that the live Delhi dataset loads, cleans, and passes quality audit."""
    df, audit = load_dataset()
    assert len(df) > 20000
    assert "Date_Parsed" in df.columns
    assert "Has_rained_Binary" in df.columns
    assert audit["total_records"] == len(df)
    assert audit["duplicate_dates"] == 0
    assert audit["rain_distribution"]["Rain_1"] > 0
    assert audit["rain_distribution"]["No_Rain_0"] > audit["rain_distribution"]["Rain_1"]
