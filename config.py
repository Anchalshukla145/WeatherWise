"""
Configuration Module for WeatherWise Analytics
Centralized paths, database configurations, and model constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Project Directory Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"
SUBMISSION_DIR = BASE_DIR / "submission"

# Dataset File Paths
RAW_CSV_PATH = DATA_DIR / "delhi_weather.csv"

# MongoDB Database Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "weatherwise_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "daily_weather")
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "3000"))

# Reproducibility and Chronological Split Settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
CHRONOLOGICAL_SPLIT_RATIO = 0.80

# Machine Learning Feature and Target Specifications
# Exclude current-day target leakage (_id, current Rain, current Has_rained)
CLASSIFICATION_FEATURES = ["Temp Max", "Temp Min", "Temp_diff", "Month", "Day", "Week"]
REGRESSION_FEATURES = ["Temp Max", "Temp Min", "Temp_diff", "Month", "Day", "Week"]

TARGET_CLASSIFICATION = "Target_Rain_NextDay"
TARGET_REGRESSION = "Target_TempMax_NextDay"

# Serialized Model Artifacts
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
DECISION_TREE_MODEL_PATH = MODELS_DIR / "decision_tree.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
TEMP_REGRESSOR_MODEL_PATH = MODELS_DIR / "random_forest_regressor.joblib"

# Evaluation Outputs
CLASSIFICATION_METRICS_PATH = OUTPUTS_DIR / "classification_metrics.json"
REGRESSION_METRICS_PATH = OUTPUTS_DIR / "regression_metrics.json"
MODEL_COMPARISON_CSV_PATH = OUTPUTS_DIR / "model_comparison.csv"
CLASSIFICATION_TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions_classification.csv"
REGRESSION_TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions_regression.csv"

# Report Deliverable Paths
REPORT_PDF_PATH = REPORTS_DIR / "project_report.pdf"
REPORT_DOCX_PATH = REPORTS_DIR / "project_report.docx"

# Submission Archive Path
SUBMISSION_ZIP_PATH = SUBMISSION_DIR / "WeatherWise_Analytics_Submission.zip"

# UI Theme Tokens
APP_TITLE = "WeatherWise Analytics"
APP_SUBTITLE = "MongoDB-Powered Weather Intelligence and Machine Learning System"
PRIMARY_COLOR = "#0D3B66"
ACCENT_TEAL = "#00A896"
ACCENT_BLUE = "#028090"
BG_LIGHT = "#F8F9FA"
TEXT_NAVY = "#0F172A"
