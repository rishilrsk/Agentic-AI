"""
Logging utility for Agentic Data Analyst.
Configures structured, informative console and file logging.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "AgenticDataAnalyst", log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (with graceful fallback for restricted/read-only cloud environments)
    try:
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"File logging not available ({str(e)}). Logging to console only.")

    return logger


logger = setup_logger()
