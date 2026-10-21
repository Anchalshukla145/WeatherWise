"""
MongoDB Management Module for WeatherWise Analytics.
Provides robust PyMongo connection pooling, CSV-to-MongoDB migration,
duplicate-prevention upserting, querying, indexing, and offline-fallback handling.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

from config import (
    MONGO_COLLECTION_NAME,
    MONGO_DB_NAME,
    MONGO_TIMEOUT_MS,
    MONGO_URI,
)
from src.utils import logger


class MongoDBManager:
    """Manages connections, migrations, indexes, and queries for MongoDB."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ):
        self.uri = uri or MONGO_URI
        self.db_name = db_name or MONGO_DB_NAME
        self.collection_name = collection_name or MONGO_COLLECTION_NAME
        self.timeout_ms = timeout_ms or MONGO_TIMEOUT_MS
        self._client: Optional[MongoClient] = None

    def get_client(self) -> Optional[MongoClient]:
        """Returns or establishes a reusable MongoClient instance."""
        if self._client is None:
            try:
                self._client = MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=self.timeout_ms,
                    connectTimeoutMS=self.timeout_ms,
                    socketTimeoutMS=self.timeout_ms,
                )
            except Exception as e:
                logger.error(f"Failed to instantiate MongoClient: {e}")
                self._client = None
        return self._client

    def ping(self) -> Tuple[bool, str]:
        """
        Tests whether MongoDB is reachable via admin ping.
        Returns (is_connected: bool, status_message: str).
        """
        try:
            client = self.get_client()
            if client is None:
                return False, "MongoClient instance is null."
            client.admin.command("ping")
            return True, "MongoDB connection active and verified."
        except ServerSelectionTimeoutError as e:
            return False, f"Connection timed out ({self.timeout_ms}ms). Server unreachable: {e}"
        except ConnectionFailure as e:
            return False, f"Connection failed: {e}"
        except Exception as e:
            return False, f"Database error during ping: {e}"

    def ensure_indexes(self) -> List[str]:
        """Creates indexes on Date and Record_Key to enforce uniqueness and speed up queries."""
        connected, _ = self.ping()
        if not connected:
            return []
        try:
            db = self._client[self.db_name]
            coll = db[self.collection_name]
            idx1 = coll.create_index("Record_Key", unique=True, name="idx_record_key_unique")
            idx2 = coll.create_index("Date_Parsed", name="idx_date_parsed")
            idx3 = coll.create_index([("Year", 1), ("Month", 1)], name="idx_year_month")
            return [idx1, idx2, idx3]
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return []

    def import_dataframe(self, df: pd.DataFrame, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Imports cleaned DataFrame records into MongoDB using bulk upserts
        to prevent duplicate entries across repeated imports.
        """
        connected, msg = self.ping()
        if not connected:
            return {
                "success": False,
                "message": f"MongoDB is offline. Cannot import. Details: {msg}",
                "inserted_count": 0,
                "upserted_count": 0,
                "modified_count": 0,
                "total_records": 0,
            }

        self.ensure_indexes()
        db = self._client[self.db_name]
        coll = db[self.collection_name]

        # Prepare records
        records_to_import = []
        for _, row in df.iterrows():
            record = row.to_dict()
            # Convert Timestamp to ISO string or standard datetime for BSON
            if isinstance(record.get("Date_Parsed"), pd.Timestamp):
                record["Date_Parsed"] = record["Date_Parsed"].to_pydatetime()

            record_key = str(record.get("Record_Key", record.get("Date", "")))
            # Drop redundant dataframe internal index if present
            record.pop("_id", None)

            records_to_import.append(
                UpdateOne(
                    {"Record_Key": record_key},
                    {"$set": record},
                    upsert=True
                )
            )

        total_batches = (len(records_to_import) + batch_size - 1) // batch_size
        upserted_count = 0
        modified_count = 0
        matched_count = 0

        try:
            for i in range(total_batches):
                chunk = records_to_import[i * batch_size: (i + 1) * batch_size]
                result = coll.bulk_write(chunk, ordered=False)
                upserted_count += len(result.upserted_ids) if result.upserted_ids else 0
                modified_count += result.modified_count
                matched_count += result.matched_count

            current_total = coll.count_documents({})
            return {
                "success": True,
                "message": f"Import completed successfully. Processed {len(records_to_import)} records.",
                "upserted_new": upserted_count,
                "modified_existing": modified_count,
                "matched_count": matched_count,
                "total_collection_documents": current_total,
            }
        except Exception as e:
            logger.error(f"Bulk import error: {e}")
            return {
                "success": False,
                "message": f"Error during bulk write: {e}",
                "upserted_new": upserted_count,
                "modified_existing": modified_count,
                "matched_count": matched_count,
                "total_collection_documents": coll.count_documents({}),
            }

    def fetch_records(
        self,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort_by: str = "Date_Parsed",
        ascending: bool = True,
        limit: int = 0
    ) -> pd.DataFrame:
        """
        Fetches records matching query and returns them as a sorted DataFrame.
        """
        connected, _ = self.ping()
        if not connected:
            logger.warning("MongoDB offline during fetch_records; returning empty DataFrame.")
            return pd.DataFrame()

        try:
            db = self._client[self.db_name]
            coll = db[self.collection_name]
            filter_query = query or {}
            proj = projection or {"_id": 0}
            direction = 1 if ascending else -1

            cursor = coll.find(filter_query, proj).sort(sort_by, direction)
            if limit > 0:
                cursor = cursor.limit(limit)

            docs = list(cursor)
            if not docs:
                return pd.DataFrame()

            df = pd.DataFrame(docs)
            if "Date_Parsed" in df.columns:
                df["Date_Parsed"] = pd.to_datetime(df["Date_Parsed"])
            return df
        except Exception as e:
            logger.error(f"Error fetching records from MongoDB: {e}")
            return pd.DataFrame()

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Returns document count, index details, and date span."""
        connected, msg = self.ping()
        if not connected:
            return {
                "status": "Offline",
                "connected": False,
                "message": msg,
                "count": 0,
                "indexes": [],
                "date_min": None,
                "date_max": None,
            }

        try:
            db = self._client[self.db_name]
            coll = db[self.collection_name]
            count = coll.count_documents({})
            indexes = list(coll.list_indexes())
            index_names = [idx.get("name") for idx in indexes]

            earliest = coll.find({}, {"Date_Parsed": 1}).sort("Date_Parsed", 1).limit(1)
            latest = coll.find({}, {"Date_Parsed": 1}).sort("Date_Parsed", -1).limit(1)

            e_doc = list(earliest)
            l_doc = list(latest)

            date_min = str(e_doc[0]["Date_Parsed"].date()) if e_doc and "Date_Parsed" in e_doc[0] else None
            date_max = str(l_doc[0]["Date_Parsed"].date()) if l_doc and "Date_Parsed" in l_doc[0] else None

            return {
                "status": "Online",
                "connected": True,
                "message": "Connected",
                "database": self.db_name,
                "collection": self.collection_name,
                "count": count,
                "indexes": index_names,
                "date_min": date_min,
                "date_max": date_max,
            }
        except Exception as e:
            return {
                "status": "Error",
                "connected": False,
                "message": str(e),
                "count": 0,
                "indexes": [],
                "date_min": None,
                "date_max": None,
            }

    def drop_collection(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Destructive operation: drops collection only if explicit confirm=True is passed.
        """
        if not confirm:
            return {"success": False, "message": "Confirmation required to drop collection."}

        connected, msg = self.ping()
        if not connected:
            return {"success": False, "message": f"MongoDB is offline: {msg}"}

        try:
            db = self._client[self.db_name]
            db.drop_collection(self.collection_name)
            return {"success": True, "message": f"Collection '{self.collection_name}' dropped successfully."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def close(self) -> None:
        """Closes the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
