"""Structured logging utility for AdvancEye 2.0."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_logger_instance: Optional[logging.Logger] = None


def get_logger(name: str = "AdvancEye", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured logger with console and rotating file output.

    Args:
        name: Name of the logger instance.
        log_dir: Directory where log files are stored.
        level: Logging severity level.

    Returns:
        Configured logging.Logger instance.
    """
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Rotating file handler
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=log_path / "advanceye.log",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not initialize file log handler: {e}")

    _logger_instance = logger
    return _logger_instance
