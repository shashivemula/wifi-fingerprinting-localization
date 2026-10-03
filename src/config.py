"""Project paths and basic configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAINING_DATA_PATH = RAW_DATA_DIR / "trainingData.csv"
VALIDATION_DATA_PATH = RAW_DATA_DIR / "validationData.csv"

WAP_PREFIX = "WAP"
WAP_COUNT = 520
RSSI_FEATURE_COLUMNS = tuple(
    f"{WAP_PREFIX}{index:03d}" for index in range(1, WAP_COUNT + 1)
)
