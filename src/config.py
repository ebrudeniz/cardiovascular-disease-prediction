"""
Configuration file for Cardiovascular Disease Prediction project
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Data Files
RAW_DATA_FILE = RAW_DATA_DIR / "cardio_train.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cardio_processed.csv"
FINAL_FEATURES_FILE = PROCESSED_DATA_DIR / "final_features.json"
FEATURE_IMPORTANCE_FILE = PROCESSED_DATA_DIR / "final_feature_importance.csv"

# Model Files
MODEL_FILE = MODELS_DIR / "final_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
MODEL_PARAMS_FILE = MODELS_DIR / "model_params.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target Variable
TARGET_COL = "cardio"

# Feature Columns (original)
ORIGINAL_FEATURES = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

# Business Thresholds
PROBABILITY_THRESHOLD = 0.5
LOW_RISK_THRESHOLD = 0.4
HIGH_RISK_THRESHOLD = 0.7

# Risk Categories
RISK_CATEGORIES = {
    'low': 'Low Risk',
    'medium': 'Medium Risk',
    'high': 'High Risk'
}

# API Settings
API_TITLE = "Cardiovascular Disease Prediction API"
API_DESCRIPTION = """
Cardiovascular Disease Risk Prediction API using Machine Learning.

This API predicts the risk of cardiovascular disease based on patient data.

**Features:**
- Risk probability prediction
- Risk category classification
- Feature importance explanation

**Model Performance:**
- ROC-AUC: 0.7912
- Accuracy: 0.7286
- Model: XGBoost Classifier
"""
API_VERSION = "1.0.0"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"