import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data Files
RAW_DATA_FILE = RAW_DATA_DIR / "cardio_train.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cardio_processed.csv"

# Model Files
MODEL_FILE = MODELS_DIR / "final_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target Variable
TARGET_COL = "cardio"

# Business Thresholds
PROBABILITY_THRESHOLD = 0.5