"""
Integration tests for MongoDBManager with graceful skip on offline database.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mongodb_manager import MongoDBManager


def test_mongodb_connection_status():
    """Checks MongoDB connection status without throwing unhandled exceptions."""
    mgr = MongoDBManager()
    connected, msg = mgr.ping()
    assert isinstance(connected, bool)
    assert isinstance(msg, str)


def test_mongodb_integration_workflow():
    """
    Executes integration tests if MongoDB is active.
    If MongoDB is offline, marks test as skipped with explicit reason.
    """
    mgr = MongoDBManager()
    connected, msg = mgr.ping()

    if not connected:
        pytest.skip(f"MongoDB is not currently running locally ({msg}). Skipping live database integration tests.")

    # Live integration checks
    stats = mgr.get_collection_statistics()
    assert stats["connected"] is True
    assert stats["count"] >= 0

    # Test single-record upsert with duplicate prevention
    test_df = pd.DataFrame([{
        "Record_Key": "9999-12-31",
        "Date": "31-12-9999",
        "Temp Max": 25.0,
        "Temp Min": 15.0,
        "Temp_diff": 10.0,
        "Rain": "0",
        "Has_rained": "NO",
        "Has_rained_Binary": 0,
        "Month": 12,
        "Year": 9999,
        "Day": 31,
        "Week": 52,
        "month_name": "December",
    }])

    # First import
    res1 = mgr.import_dataframe(test_df)
    assert res1["success"] is True

    # Repeated import must match existing and not create duplicate
    res2 = mgr.import_dataframe(test_df)
    assert res2["success"] is True
    assert res2["upserted_new"] == 0
    assert res2["matched_count"] == 1

    # Cleanup test record
    try:
        client = mgr.get_client()
        client[mgr.db_name][mgr.collection_name].delete_one({"Record_Key": "9999-12-31"})
    except Exception:
        pass
