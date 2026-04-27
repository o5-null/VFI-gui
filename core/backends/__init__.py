"""Backend abstraction for video processing.

This module defines the backend interface and types for video processing,
supporting multiple backends (VapourSynth, PyTorch, etc.) with a unified API.

Architecture:
    BackendType (enum) -> Defines available backends
    BackendConfig (dataclass) -> Configuration for a backend
    BaseBackend (ABC) -> Abstract interface for all backends
    BackendFactory -> Creates backend instances
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from PyQt6.QtCore import QThread, pyqtSignal, QObject
import numpy as np


class BackendType(Enum):
    """Supported processing backends."""
    VAPOURSYNTH = "vapoursynth"
    TORCH = "torch"
    TENSORRT = "tensorrt"  # Future: direct TensorRT backend
    ONNX = "onnx"          # Future: ONNX Runtime backend


@dataclass
class BackendConfig:
    """Configuration for a processing backend.
    
    Attributes:
        backend_type: Type of backend to use
        models_dir: Directory containing model files
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        num_threads: Number of threads for processing
        device: Device to use (e.g., "cuda:0", "cpu")
        fp16: Whether to use FP16 precision
        extra: Additional backend-specific options
    """
    backend_type: BackendType = BackendType.VAPOURSYNTH
    models_dir: str = "models"
    temp_dir: str = "temp"
    output_dir: str = "output"
    num_threads: int = 4
    device: str = "auto"  # "auto", "cuda:0", "cuda:1", "cpu", etc.
    fp16: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def get_device(self) -> str:
        """Resolve device string to actual device."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        return self.device


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline.
    
    Attributes:
        interpolation: Interpolation settings
        upscaling: Upscaling settings
        scene_detection: Scene detection settings
        output: Output encoding settings
    """
    interpolation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "model_type": "rife",
        "model_version": "",
        "multi": 2,
        "scale": 1.0,
        "scene_change": False,
    })
    upscaling: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "engine": "",
        "num_streams": 3,
        "tile_size": 0,
        "overlap": 0,
    })
    scene_detection: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "model": 12,
        "threshold": 0.5,
        "fp16": True,
    })
    output: Dict[str, Any] = field(default_factory=lambda: {
        "codec": "hevc_nvenc",
        "quality": 22,
        "preset": "p4",
        "audio_copy": True,
        "output_dir": "",  # Custom output directory (empty = use default)
        "output_subdir": "",  # Custom subdirectory name (empty = auto-generate)
        "output_filename": "",  # Custom filename pattern (empty = auto-generate)
    })


class ProcessingResult:
    """Result of a video processing operation.
    
    Attributes:
        success: Whether processing succeeded
        output_path: Path to output file (if successful)
        error_message: Error message (if failed)
        processing_time: Total processing time in seconds
        stats: Additional statistics
    """
    def __init__(
        self,
        success: bool,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
        processing_time: float = 0.0,
        stats: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.output_path = output_path
        self.error_message = error_message
        self.processing_time = processing_time
        self.stats = stats or {}


class BaseBackend(ABC):
    """Abstract base class for processing backends.
    
    All backends must implement this interface to be used by the Processor.
    Each backend handles the actual video processing using its specific
    technology (VapourSynth, PyTorch, etc.).
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
        parent: Optional[QObject] = None,
    ):
        """Initialize the backend.
        
        Args:
            config: Backend configuration
            parent: Parent QObject for Qt signal handling
        """
        self._config = config
        self._parent = parent
        self._is_initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized and ready."""
        return self._is_initialized
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend.
        
        Returns:
            True if initialization succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    def process(
        self,
        video_path: str,
        processing_config: ProcessingConfig,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> ProcessingResult:
        """Process a video file.

        Args:
            video_path: Path to input video
            processing_config: Processing pipeline configuration
            progress_callback: Callback for progress updates (current, total, fps)
            stage_callback: Callback for stage changes (stage_name)
            log_callback: Callback for log messages (message)

        Returns:
            ProcessingResult with output path or error
        """
        pass

    def process_frames(
        self,
        video_path: str,
        processing_config: ProcessingConfig,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Generator[Tuple[int, np.ndarray, Dict[str, Any]], None, None]:
        """Process video and yield frames one by one.

        This method separates processing from IO, allowing the caller
        to handle output writing. Yields (frame_index, frame_data, metadata)
        tuples for each processed frame.

        Args:
            video_path: Path to input video
            processing_config: Processing pipeline configuration
            progress_callback: Callback for progress updates (current, total, fps)
            stage_callback: Callback for stage changes (stage_name)
            log_callback: Callback for log messages (message)

        Yields:
            Tuple of (frame_index, frame_data, metadata) where:
                - frame_index: int, sequential frame number
                - frame_data: np.ndarray, frame data in RGB uint8 format [H, W, C]
                - metadata: dict with keys like 'fps', 'width', 'height', 'total_frames'
        """
        # Default implementation calls process() - subclasses should override
        # for true streaming behavior
        raise NotImplementedError(
            "Backend must implement process_frames() for streaming support"
        )

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the current processing operation."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources after processing."""
        pass
    
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
        parent: Optional[QObject] = None,
    ) -> BaseBackend:
        """Create a backend instance.
        
        Args:
            backend_type: Type of backend to create
            config: Backend configuration
            parent: Parent QObject
            
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


# Import logger for warnings (must be before _register_builtin_backends)
from loguru import logger


# Auto-register built-in backends
def _register_builtin_backends():
    """Register built-in backend implementations."""
    try:
        from .torch_backend import TorchBackend
        BackendFactory.register(BackendType.TORCH, TorchBackend)
    except ImportError as e:
        logger.warning(f"Failed to register TorchBackend: {e}")


# Register backends on module import
_register_builtin_backends()
