"""Loguru-based logging configuration for VFI-gui."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    enable_console: bool = True,
) -> None:
    """
    Configure loguru logger for VFI-gui application.
    
    Args:
        log_dir: Directory for log files. If None, uses paths.logs_dir.
        log_level: Minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: Log file rotation size/time.
        retention: How long to keep old log files.
        enable_console: Whether to also log to console.
    """
    # Remove default handler
    logger.remove()
    
    # Determine log directory
    if log_dir is None:
        from core.paths import paths
        log_dir = str(paths.logs_dir)
    
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Log file path
    log_file = Path(log_dir) / "vfi_gui_{time:YYYY-MM-DD}.log"
    
    # Console format - colorful and concise
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # File format - detailed for debugging
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    # Add console handler if enabled
    if enable_console:
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    
    # Add file handler
    logger.add(
        str(log_file),
        format=file_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )
    
    logger.info(f"Logger initialized. Log directory: {log_dir}")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Optional name to bind to logger (e.g., module name).
    
    Returns:
        Logger instance with name bound if provided.
    """
    if name:
        return logger.bind(name=name)
    return logger


# Convenience exports
__all__ = ["logger", "setup_logger", "get_logger"]
