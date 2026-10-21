"""
Utility functions and shared helpers for WeatherWise Analytics.
Provides safe conversions, formatting, logging, and JSON helpers.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WeatherWise")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely handle NumPy and Pandas data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def safe_to_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely converts an input value to float, handling 'Tr' (trace) and nulls."""
    if pd.isna(val) or val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if val_str.lower() in ("tr", "trace"):
        return 0.05
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return default


def format_metric(val: Optional[float], decimals: int = 4, suffix: str = "") -> str:
    """Format floating point numbers consistently for presentation."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def format_temp(val: Optional[float]) -> str:
    """Format temperature string with degree Celsius."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.1f} °C"


def save_json_file(data: Dict[str, Any], filepath: Any) -> None:
    """Safely save dictionary to JSON file handling NumPy types."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json_file(filepath: Any) -> Dict[str, Any]:
    """Load JSON file safely."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
