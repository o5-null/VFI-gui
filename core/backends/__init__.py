"""Backend abstraction for video processing.

This module provides the backend interface and types for video processing,
supporting multiple backends (VapourSynth, PyTorch, etc.) with a unified API.

Types are imported from core.types (single source of truth).
BaseBackend and BackendFactory are defined in base_backend.py.

Architecture:
    core.types -> Single source of truth for all type definitions
    base_backend.py -> BaseBackend ABC + BackendFactory
    inprocess_backend.py -> InProcessBackend (zero IPC, direct Python calls)
    subprocess_backend.py -> SubProcessBackend (stdin/stdout JSON pipes)
    __init__.py -> Re-export hub + auto-registration

Re-exported from core.types:
    BackendType, BackendConfig, ProcessingConfig, ProcessingResult,
    InferenceRequest, InferenceResult, InferenceStrategy

Re-exported from base_backend.py:
    BaseBackend, BackendFactory

Re-exported from inprocess_backend.py:
    InProcessBackend

Re-exported from subprocess_backend.py:
    SubProcessBackend, EngineMessage
"""

from __future__ import annotations

# Re-export all types from core.types (single source of truth)
from core.types import (
    BackendType,
    BackendConfig,
    ProcessingConfig,
    ProcessingResult,
    InferenceRequest,
    InferenceResult,
    InferenceStrategy,
)

# Re-export BaseBackend and BackendFactory from base_backend.py
from .base_backend import BaseBackend, BackendFactory

# Lazy imports for execution mode backends (avoid circular imports)
# Import explicitly when needed:
#   from core.backends.inprocess_backend import InProcessBackend
#   from core.backends.subprocess_backend import SubProcessBackend, EngineMessage

# Import logger for warnings (must be before _register_builtin_backends)
from loguru import logger


# Auto-register built-in backends
def _register_builtin_backends():
    """Register built-in backend implementations."""
    # Torch backend (active)
    try:
        from .torch_backend import TorchBackend
        BackendFactory.register(BackendType.TORCH, TorchBackend)
    except ImportError as e:
        logger.warning(f"Failed to register TorchBackend: {e}")

    # NCNN backend (placeholder)
    try:
        from .ncnn_backend import NCNNBackend
        BackendFactory.register(BackendType.NCNN, NCNNBackend)
    except ImportError as e:
        logger.warning(f"Failed to register NCNNBackend: {e}")

    # DirectML backend (placeholder)
    try:
        from .directml_backend import DirectMLBackend
        BackendFactory.register(BackendType.DIRECTML, DirectMLBackend)
    except ImportError as e:
        logger.warning(f"Failed to register DirectMLBackend: {e}")


# Register backends on module import
_register_builtin_backends()
