"""In-process inference backend: direct Python function calls, zero IPC.

This backend executes inference within the same Python process. It wraps
an existing BaseBackend implementation (e.g., TorchBackend) and provides
engine-level management (model loading, GPU device binding, status tracking).

Use cases:
    - PyTorch: Model is loaded in-process, IPC would add latency
    - TensorRT: TRT engine needs in-process GPU context
    - DirectML: DML session requires in-process GPU context

Architecture:
    InProcessBackend(BaseBackend) → wraps another BaseBackend
    - Delegates infer()/infer_batch() to the wrapped backend
    - Adds engine-level lifecycle management
    - Zero IPC overhead
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from core.types import (
    BackendType,
    BackendConfig,
    InferenceRequest,
    InferenceResult,
    EngineStatus,
)
from .base_backend import BaseBackend, BackendFactory


class InProcessBackend(BaseBackend):
    """In-process inference backend: direct Python function calls.

    Wraps an existing BaseBackend implementation for in-process execution.
    Provides engine-level lifecycle management on top of the wrapped backend.

    Features:
        - Zero IPC overhead (direct function calls)
        - Shared GPU memory space (no serialization/deserialization)
        - Engine status tracking (IDLE → LOADING → READY → RUNNING)
        - Model preloading support

    Limitations:
        - GPU crash affects the entire process
        - GPU memory is shared (no isolation)
    """

    # Backend metadata
    BACKEND_TYPE = BackendType.TORCH
    BACKEND_NAME = "InProcess"
    BACKEND_DESCRIPTION = "In-process inference backend (zero IPC overhead)"

    # Supported features - delegates to wrapped backend
    SUPPORTS_INTERPOLATION = True
    SUPPORTS_UPSCALING = False
    SUPPORTS_SCENE_DETECTION = False
    SUPPORTED_MODELS: Dict[str, List[str]] = {}

    def __init__(
        self,
        config: BackendConfig,
        parent: Optional[object] = None,
        wrapped_backend: Optional[BaseBackend] = None,
    ):
        """Initialize the in-process backend.

        Args:
            config: Backend configuration
            parent: Parent object for signal handling
            wrapped_backend: Pre-created backend to wrap (optional)
        """
        super().__init__(config, parent)
        self._wrapped: Optional[BaseBackend] = wrapped_backend
        self._engine_status: EngineStatus = EngineStatus.IDLE

    @property
    def engine_status(self) -> EngineStatus:
        """Get the current engine lifecycle status."""
        return self._engine_status

    @property
    def wrapped_backend(self) -> Optional[BaseBackend]:
        """Get the wrapped backend instance."""
        return self._wrapped

    def _ensure_wrapped(self) -> BaseBackend:
        """Ensure a wrapped backend exists, creating one if needed.

        Returns:
            The wrapped backend instance

        Raises:
            RuntimeError: If backend cannot be created
        """
        if self._wrapped is not None:
            return self._wrapped

        # Create wrapped backend via factory
        try:
            self._wrapped = BackendFactory.create(
                self._config.backend_type, self._config, self._parent
            )
            return self._wrapped
        except ValueError as e:
            raise RuntimeError(
                f"Cannot create wrapped backend for "
                f"type={self._config.backend_type.value}: {e}"
            ) from e

    def initialize(self) -> bool:
        """Initialize the backend and wrapped backend.

        Returns:
            True if initialization succeeded
        """
        self._engine_status = EngineStatus.LOADING
        try:
            backend = self._ensure_wrapped()
            result = backend.initialize()
            if result:
                self._is_initialized = True
                self._engine_status = EngineStatus.READY
                logger.info(
                    f"InProcessBackend initialized: "
                    f"type={self._config.backend_type.value}"
                )
            else:
                self._engine_status = EngineStatus.ERROR
                logger.error("InProcessBackend initialization failed")
            return result
        except Exception as e:
            self._engine_status = EngineStatus.ERROR
            logger.error(f"InProcessBackend initialization error: {e}")
            return False

    def load_model(self, model_config: Dict[str, Any]) -> bool:
        """Load model via the wrapped backend.

        Args:
            model_config: Model configuration dict

        Returns:
            True if model loaded successfully
        """
        if self._engine_status == EngineStatus.ERROR:
            logger.error("Cannot load model: engine in ERROR state")
            return False

        self._engine_status = EngineStatus.LOADING
        try:
            backend = self._ensure_wrapped()
            if hasattr(backend, "load_model"):
                result = backend.load_model(model_config)  # type: ignore[union-attr]
            else:
                result = True  # Some backends don't have explicit load_model

            if result:
                self._engine_status = EngineStatus.READY
                logger.info(f"Model loaded: {model_config.get('model_type', 'unknown')}")
            else:
                self._engine_status = EngineStatus.ERROR
            return result
        except Exception as e:
            self._engine_status = EngineStatus.ERROR
            logger.error(f"Model load error: {e}")
            return False

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Single frame-pair inference via direct function call.

        Zero IPC: calls the wrapped backend's infer() directly.

        Args:
            request: Inference request with frame data

        Returns:
            Inference result with interpolated frame
        """
        if self._engine_status not in (EngineStatus.READY, EngineStatus.RUNNING):
            return InferenceResult(
                output_frame=np.array([]),
                success=False,
                error=f"Engine not ready: status={self._engine_status.value}",
            )

        self._engine_status = EngineStatus.RUNNING
        try:
            backend = self._ensure_wrapped()
            result = backend.infer(request)

            # Return to READY after successful inference
            if result.success:
                self._engine_status = EngineStatus.READY

            return result
        except Exception as e:
            self._engine_status = EngineStatus.ERROR
            logger.error(f"Inference error: {e}")
            return InferenceResult(
                output_frame=np.array([]),
                success=False,
                error=str(e),
            )

    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch inference via direct function call.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results
        """
        if self._engine_status not in (EngineStatus.READY, EngineStatus.RUNNING):
            return [
                InferenceResult(
                    output_frame=np.array([]),
                    success=False,
                    error=f"Engine not ready: status={self._engine_status.value}",
                )
                for _ in requests
            ]

        self._engine_status = EngineStatus.RUNNING
        try:
            backend = self._ensure_wrapped()
            results = backend.infer_batch(requests)

            # Return to READY after inference
            self._engine_status = EngineStatus.READY
            return results
        except NotImplementedError:
            raise
        except Exception as e:
            self._engine_status = EngineStatus.ERROR
            return [
                InferenceResult(
                    output_frame=np.array([]),
                    success=False,
                    error=str(e),
                )
                for _ in requests
            ]

    def unload_model(self) -> None:
        """Unload model and release GPU memory."""
        if self._wrapped is not None and hasattr(self._wrapped, "unload_model"):
            try:
                self._wrapped.unload_model()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")

        self._engine_status = EngineStatus.IDLE

    def cancel(self) -> None:
        """Cancel the current processing operation."""
        if self._wrapped is not None:
            self._wrapped.cancel()
        self._engine_status = EngineStatus.IDLE

    def cleanup(self) -> None:
        """Clean up all resources."""
        if self._wrapped is not None:
            self._wrapped.cleanup()
            self._wrapped = None
        self._engine_status = EngineStatus.IDLE
        self._is_initialized = False
