"""
Centralized logging configuration for emb-explorer.

Usage:
    from shared.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import os
import sys
from typing import Optional


# Module-level flag to track if logging has been configured
_logging_configured = False

# Default log directory (relative to working directory)
_LOG_DIR = os.environ.get("EMB_EXPLORER_LOG_DIR", "logs")
_LOG_FILE = "emb_explorer.log"


def configure_logging(
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    log_to_file: bool = True,
):
    """
    Configure the root logger for the application.

    Args:
        level: Logging level (default: INFO)
        log_format: Custom log format string (optional)
        log_to_file: Whether to also write logs to a file (default: True)
    """
    global _logging_configured

    if _logging_configured:
        return

    if log_format is None:
        log_format = (
            "[%(asctime)s] %(levelname)s "
            "[%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (append mode, rotates implicitly by date via log dir)
    if log_to_file:
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(_LOG_DIR, _LOG_FILE), mode="a", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError:
            # Non-fatal: skip file logging if directory can't be created
            pass

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Automatically configures logging if not already done.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    configure_logging()
    return logging.getLogger(name)
