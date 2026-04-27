"""
VSGAN configuration module.
Centralized configuration for paths and settings.
"""

import os
from pathlib import Path
from typing import Optional


# Environment variable names
ENV_MODELS_DIR = "VSGAN_MODELS_DIR"
ENV_ENGINE_CACHE_DIR = "VSGAN_ENGINE_CACHE_DIR"
ENV_TEMP_DIR = "VSGAN_TEMP_DIR"

# Default paths (can be overridden)
_DEFAULT_MODELS_DIR: Optional[str] = None
_DEFAULT_ENGINE_CACHE_DIR: Optional[str] = None
_DEFAULT_TEMP_DIR: Optional[str] = None


def get_models_dir() -> str:
    """
    Get models directory path.
    Priority: environment variable > programmatically set default > centralized paths
    """
    if ENV_MODELS_DIR in os.environ:
        return os.environ[ENV_MODELS_DIR]
    if _DEFAULT_MODELS_DIR is not None:
        return _DEFAULT_MODELS_DIR
    # Use centralized path manager
    try:
        from core.paths import paths
        return str(paths.models_dir)
    except ImportError:
        # Fallback to current working directory (legacy behavior)
        return str(Path.cwd() / "models")


def get_engine_cache_dir() -> str:
    """Get TensorRT engine cache directory."""
    if ENV_ENGINE_CACHE_DIR in os.environ:
        return os.environ[ENV_ENGINE_CACHE_DIR]
    if _DEFAULT_ENGINE_CACHE_DIR is not None:
        return _DEFAULT_ENGINE_CACHE_DIR
    # Default to models directory
    return get_models_dir()


def get_temp_dir() -> str:
    """Get temporary directory for intermediate files."""
    if ENV_TEMP_DIR in os.environ:
        return os.environ[ENV_TEMP_DIR]
    if _DEFAULT_TEMP_DIR is not None:
        return _DEFAULT_TEMP_DIR
    # Use centralized path manager
    try:
        from core.paths import paths
        return str(paths.temp_dir)
    except ImportError:
        # Fallback to current working directory (legacy behavior)
        return str(Path.cwd() / "temp")


def configure(
    models_dir: Optional[str] = None,
    engine_cache_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
) -> None:
    """
    Configure VSGAN paths at runtime.
    
    Args:
        models_dir: Path to models directory (TensorRT engines, ONNX files)
        engine_cache_dir: Path to TensorRT engine cache directory
        temp_dir: Path to temporary files directory
    """
    global _DEFAULT_MODELS_DIR, _DEFAULT_ENGINE_CACHE_DIR, _DEFAULT_TEMP_DIR
    
    if models_dir is not None:
        _DEFAULT_MODELS_DIR = models_dir
        # Ensure directory exists
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    if engine_cache_dir is not None:
        _DEFAULT_ENGINE_CACHE_DIR = engine_cache_dir
        Path(engine_cache_dir).mkdir(parents=True, exist_ok=True)
    
    if temp_dir is not None:
        _DEFAULT_TEMP_DIR = temp_dir
        Path(temp_dir).mkdir(parents=True, exist_ok=True)


def reset() -> None:
    """Reset all configuration to defaults."""
    global _DEFAULT_MODELS_DIR, _DEFAULT_ENGINE_CACHE_DIR, _DEFAULT_TEMP_DIR
    _DEFAULT_MODELS_DIR = None
    _DEFAULT_ENGINE_CACHE_DIR = None
    _DEFAULT_TEMP_DIR = None
