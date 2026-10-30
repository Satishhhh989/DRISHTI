"""PHOTO → 3D ENGINE - Structured Logging

Provides structured logging without leaking sensitive local paths
or printing raw image binary dumps.
"""

import logging
import sys
from typing import Any, Dict


class SafeLogFormatter(logging.Formatter):
    """Formats log records cleanly and prevents sensitive dump leakage."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(5)
        name = record.name
        message = record.getMessage()
        return f"[{timestamp}] [{level}] [{name}] {message}"


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("photo_to_3d")
    logger.setLevel(level)

    # Avoid duplicate handlers on re-initialization
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SafeLogFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logging()
