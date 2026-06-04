import os
import logging
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

# Ensure output directories exist
for d in [IMAGES_DIR, METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
IMAGES_CSV = METADATA_DIR / "images.csv"
DUPLICATES_CSV = METADATA_DIR / "duplicates.csv"
LOG_FILE = LOGS_DIR / "scraper.log"

# Scraper Constants
BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
IMAGE_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/"
API_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
}

# Configurable settings
TARGET_IMAGE_COUNT = 100
MIN_WIDTH = 150
MIN_HEIGHT = 150
MAX_RETRIES = 3
REQUEST_DELAY = 1.0  # seconds between requests in crawler
MAX_WORKERS = 5      # For ThreadPoolExecutor

def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
