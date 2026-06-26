"""
Logging configuration.
"""

import logging
from pathlib import Path

from configs.config import LOG_DIR, LOG_LEVEL

LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = Path(LOG_DIR) / "app.log"

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("ECG_Project")