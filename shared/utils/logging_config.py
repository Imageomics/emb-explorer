"""
Centralized logging configuration for emb-explorer.

Usage:
    from shared.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import sys
from typing import Optional


# Module-level flag to track if logging has been configured
_logging_configured = False


def configure_logging(level: int = logging.INFO, log_format: Optional[str] = None):
    """
    Configure the root logger for the application.

    Args:
        level: Logging level (default: INFO)
        log_format: Custom log format string (optional)
    """
    global _logging_configured

    if _logging_configured:
        return

    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger.addHandler(console_handler)
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
