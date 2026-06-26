"""
Global project configuration.

Loads environment variables and provides
centralized access to project paths.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------------------------------------
# Project Root
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------
# Data Directories
# ----------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MITDB_DIR = DATA_DIR / "mitdb"
EXPORT_DIR = DATA_DIR / "exports"

# ----------------------------------------------------
# Model Directory
# ----------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(MODEL_DIR / "best_model.keras"),
)

LABEL_ENCODER_PATH = os.getenv(
    "LABEL_ENCODER",
    str(MODEL_DIR / "label_encoder.pkl"),
)

HISTORY_PATH = MODEL_DIR / "history.pkl"

CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"

# ----------------------------------------------------
# Logging
# ----------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ----------------------------------------------------
# Backend
# ----------------------------------------------------

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")

BACKEND_PORT = int(
    os.getenv("BACKEND_PORT", "8000")
)

# ----------------------------------------------------
# Frontend
# ----------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
)