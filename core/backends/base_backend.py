"""Backend abstract base class and factory.

This module defines the core backend interface for video processing.
All backends must inherit from BaseBackend and implement the required methods.

Architecture:
    BaseBackend (ABC) -> Abstract interface for all backends
    BackendFactory -> Creates backend instances by type

Core constraints for Backend implementations:
    - Backend 不接触文件路径，只接收 numpy/tensor 数据
    - Backend 不自主决定 IO 时机，由 TaskScheduler 调度
    - Backend 不直接写文件，推理结果返回给 TaskScheduler
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.types import (
    BackendType,
    BackendConfig,
    InferenceRequest,
    InferenceResult,
)


class BaseBackend(ABC):
    """Abstract base class for processing backends.

    All backends must implement this interface to be used by the Processor.
    Each backend handles the actual video processing using its specific
    technology (VapourSynth, PyTorch, etc.).

    Pure-inference interface:
        - infer(): Single frame-pair inference
        - infer_batch(): Batch inference for multiple frame pairs
    """

    # Backend metadata - override in subclasses
    BACKEND_TYPE: BackendType = BackendType.VAPOURSYNTH
    BACKEND_NAME: str = "base"
    BACKEND_DESCRIPTION: str = "Base backend"

    # Supported features - override in subclasses
    SUPPORTS_INTERPOLATION: bool = True
    SUPPORTS_UPSCALING: bool = True
    SUPPORTS_SCENE_DETECTION: bool = True
    SUPPORTED_MODELS: Dict[str, List[str]] = {}  # {model_type: [versions]}

    def __init__(
        self,
        config: BackendConfig,
        parent: Optional[object] = None,
    ):
        """Initialize the backend.

        Args:
            config: Backend configuration
            parent: Parent object for signal handling (optional)
        """
        self._config = config
        self._parent = parent
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized and ready."""
        return self._is_initialized

    # ====================
    # Pure-inference interface (zero IO)
    # ====================

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Single frame-pair inference.

        Args:
            request: Contains frame data and parameters, NO file paths.

        Returns:
            InferenceResult with interpolated frame or error.

        Raises:
            NotImplementedError: If backend hasn't implemented this method yet.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented infer()."
        )

    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch inference: merge multiple frame pairs into one batch.

        Multiple frame pairs are combined into a single batch for one inference
        pass. GPU excels at large batch data; batch inference is 2-3x faster
        than per-frame inference.

        Args:
            requests: List of InferenceRequest objects.

        Returns:
            List of InferenceResult, one per request.

        Raises:
            NotImplementedError: If model doesn't support batch inference.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented infer_batch(). "
            f"Fall back to calling infer() per request."
        )

    # ====================
    # Lifecycle management
    # ====================

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the current processing operation."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources after processing."""
        pass

    # ====================
    # Utility methods
    # ====================

    @classmethod
    def get_supported_models(cls) -> Dict[str, List[str]]:
        """Get supported models for this backend.

        Returns:
            Dict mapping model_type to list of supported versions
        """
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def is_model_supported(cls, model_type: str, model_version: str = "") -> bool:
        """Check if a model is supported by this backend.

        Args:
            model_type: Type of model (e.g., "rife", "film")
            model_version: Specific version (optional)

        Returns:
            True if model is supported
        """
        if model_type.lower() not in cls.SUPPORTED_MODELS:
            return False
        if not model_version:
            return True
        return model_version in cls.SUPPORTED_MODELS[model_type.lower()]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.BACKEND_TYPE.value})"


class BackendFactory:
    """Factory for creating backend instances.

    Usage:
        backend = BackendFactory.create(BackendType.TORCH, config)
    """

    _registry: Dict[BackendType, type] = {}

    @classmethod
    def register(cls, backend_type: BackendType, backend_class: type) -> None:
        """Register a backend class for a given type.

        Args:
            backend_type: Backend type enum value
            backend_class: Backend class (must inherit from BaseBackend)
        """
        if not issubclass(backend_class, BaseBackend):
            raise TypeError(f"{backend_class} must inherit from BaseBackend")
        cls._registry[backend_type] = backend_class

    @classmethod
    def create(
        cls,
        backend_type: BackendType,
        config: BackendConfig,
        parent: Optional[object] = None,
    ) -> BaseBackend:
        """Create a backend instance.

        Args:
            backend_type: Type of backend to create
            config: Backend configuration
            parent: Parent object

        Returns:
            Backend instance

        Raises:
            ValueError: If backend type is not registered
        """
        if backend_type not in cls._registry:
            raise ValueError(
                f"Backend type '{backend_type.value}' not registered. "
                f"Available: {[t.value for t in cls._registry.keys()]}"
            )

        backend_class = cls._registry[backend_type]
        return backend_class(config, parent)

    @classmethod
    def get_available_backends(cls) -> List[BackendType]:
        """Get list of registered backend types."""
        return list(cls._registry.keys())

    @classmethod
    def get_backend_info(cls, backend_type: BackendType) -> Dict[str, Any]:
        """Get information about a backend type.

        Args:
            backend_type: Backend type to query

        Returns:
            Dict with backend metadata
        """
        if backend_type not in cls._registry:
            return {}

        backend_class = cls._registry[backend_type]
        return {
            "type": backend_type.value,
            "name": backend_class.BACKEND_NAME,
            "description": backend_class.BACKEND_DESCRIPTION,
            "supports_interpolation": backend_class.SUPPORTS_INTERPOLATION,
            "supports_upscaling": backend_class.SUPPORTS_UPSCALING,
            "supports_scene_detection": backend_class.SUPPORTS_SCENE_DETECTION,
            "supported_models": backend_class.SUPPORTED_MODELS,
        }
