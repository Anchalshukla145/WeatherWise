"""
Data Loading and Validation Module for WeatherWise Analytics.
Handles dataset loading, date parsing, data cleaning, schema validation, and quality auditing.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config import RAW_CSV_PATH
from src.utils import logger, safe_to_float


def detect_and_load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Loads weather records from the file, detecting if it is CSV-formatted
    regardless of whether the file extension is .xls, .csv, or .txt.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Weather dataset not found at {filepath}")

    logger.info(f"Loading weather dataset from {filepath}")
    # Read first line to inspect separator
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        delimiter = "," if "," in first_line else "\t"

    df = pd.read_csv(filepath, sep=delimiter, low_memory=False)
    return df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw weather dataframe:
    - Parses Date to datetime (DD-MM-YYYY)
    - Sorts records chronologically
    - Converts 'Rain' to numeric (treating 'Tr' as trace 0.05 mm)
    - Normalizes 'Has_rained' to integer binary (1 for YES, 0 for NO)
    - Validates numeric weather columns
    - Recomputes or validates Temp_diff where missing
    """
    cleaned = df.copy()

    # Standardize column names (strip whitespace)
    cleaned.columns = [c.strip() for c in cleaned.columns]

    # Parse Date
    cleaned["Date_Parsed"] = pd.to_datetime(cleaned["Date"], format="%d-%m-%Y", errors="coerce")
    if cleaned["Date_Parsed"].isna().any():
        bad_dates = cleaned[cleaned["Date_Parsed"].isna()]["Date"].tolist()
        logger.warning(f"Found {len(bad_dates)} unparseable dates; dropping invalid dates.")
        cleaned = cleaned.dropna(subset=["Date_Parsed"])

    # Sort chronologically and reset index
    cleaned = cleaned.sort_values("Date_Parsed").reset_index(drop=True)

    # Clean Rain column: convert 'Tr' to 0.05 and parse float
    if "Rain" in cleaned.columns:
        cleaned["Rain_Numeric"] = cleaned["Rain"].apply(safe_to_float)
    else:
        cleaned["Rain_Numeric"] = np.nan

    # Clean Has_rained: YES -> 1, NO -> 0
    if "Has_rained" in cleaned.columns:
        cleaned["Has_rained_Binary"] = cleaned["Has_rained"].astype(str).str.strip().str.upper().map(
            {"YES": 1, "NO": 0, "1": 1, "0": 0, "TRUE": 1, "FALSE": 0}
        )
    else:
        # Infer from Rain_Numeric if Has_rained is absent
        cleaned["Has_rained_Binary"] = (cleaned["Rain_Numeric"] > 0).astype(int)

    # Ensure numeric columns are float
    for col in ["Temp Max", "Temp Min", "Temp_diff"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # If Temp_diff is missing, compute as Temp Max - Temp Min
    mask_diff_na = cleaned["Temp_diff"].isna() & cleaned["Temp Max"].notna() & cleaned["Temp Min"].notna()
    cleaned.loc[mask_diff_na, "Temp_diff"] = cleaned.loc[mask_diff_na, "Temp Max"] - cleaned.loc[mask_diff_na, "Temp Min"]

    # Ensure calendar fields match parsed date
    cleaned["Month"] = cleaned["Date_Parsed"].dt.month
    cleaned["Year"] = cleaned["Date_Parsed"].dt.year
    cleaned["Day"] = cleaned["Date_Parsed"].dt.day
    cleaned["Week"] = cleaned["Date_Parsed"].dt.isocalendar().week.astype(int)
    cleaned["month_name"] = cleaned["Date_Parsed"].dt.strftime("%B")

    # Maintain a deterministic record identifier string
    cleaned["Record_Key"] = cleaned["Date_Parsed"].dt.strftime("%Y-%m-%d")

    return cleaned


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive data quality and validation audit report.
    """
    total_records = len(df)
    min_date = df["Date_Parsed"].min()
    max_date = df["Date_Parsed"].max()
    date_range_days = (max_date - min_date).days + 1 if pd.notna(min_date) and pd.notna(max_date) else 0

    unique_dates = df["Date_Parsed"].nunique()
    duplicate_dates = total_records - unique_dates

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    rows_with_any_null = int(df.isna().any(axis=1).sum())

    rain_distribution = {}
    if "Has_rained_Binary" in df.columns:
        counts = df["Has_rained_Binary"].value_counts(dropna=False).to_dict()
        rain_distribution = {
            "No_Rain_0": int(counts.get(0, 0)),
            "Rain_1": int(counts.get(1, 0)),
            "Missing": int(counts.get(np.nan, 0))
        }

    # Temperature statistics
    temp_stats = {}
    for col in ["Temp Max", "Temp Min", "Temp_diff"]:
        if col in df.columns and df[col].notna().any():
            s = df[col].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = int(((s < lower_bound) | (s > upper_bound)).sum())

            temp_stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q25": float(q1),
                "median": float(s.median()),
                "q75": float(q3),
                "max": float(s.max()),
                "iqr": float(iqr),
                "outliers_count": outliers
            }

    return {
        "total_records": total_records,
        "min_date": str(min_date.date()) if pd.notna(min_date) else None,
        "max_date": str(max_date.date()) if pd.notna(max_date) else None,
        "date_span_years": round(date_range_days / 365.25, 1),
        "expected_continuous_days": date_range_days,
        "missing_days": date_range_days - unique_dates,
        "duplicate_dates": duplicate_dates,
        "null_counts": null_counts,
        "rows_with_any_null": rows_with_any_null,
        "rain_distribution": rain_distribution,
        "temp_stats": temp_stats
    }


def load_dataset(filepath: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main loader function: reads, cleans, and audits the weather dataset.
    """
    path = Path(filepath) if filepath else RAW_CSV_PATH
    raw_df = detect_and_load_csv(path)
    cleaned_df = clean_weather_data(raw_df)
    audit = audit_data_quality(cleaned_df)
    return cleaned_df, audit
