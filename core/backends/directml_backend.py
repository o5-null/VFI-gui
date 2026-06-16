"""DirectML backend for video processing.

DirectML is Microsoft's machine learning inference API for Windows, providing
GPU acceleration on DirectX 12 compatible hardware (AMD, NVIDIA, Intel).

Status: PLACEHOLDER - Not yet implemented
"""

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from loguru import logger

from . import (
    BackendType,
    BaseBackend,
    BackendConfig,
    ProcessingConfig,
    ProcessingResult,
)


class DirectMLBackend(BaseBackend):
    """DirectML-based video processing backend.
    
    PLACEHOLDER: This backend is not yet implemented.
    Will use DirectML for Windows GPU acceleration.
    
    Features (planned):
    - Windows GPU acceleration via DirectX 12
    - AMD, NVIDIA, Intel GPU support (no vendor-specific drivers needed)
    - FP16/INT8 quantization
    - Works with ONNX models
    - No CUDA dependency (good for AMD GPUs)
    
    Limitations:
    - Windows only
    - Requires DirectX 12 capable GPU
    - Requires ONNX model conversion
    - Performance may vary between GPU vendors
    """
    
    # Backend metadata
    BACKEND_TYPE = BackendType.DIRECTML
    BACKEND_NAME = "DirectML"
    BACKEND_DESCRIPTION = "DirectML Windows GPU backend (placeholder)"
    
    # Supported features (planned)
    SUPPORTS_INTERPOLATION = True  # Planned
    SUPPORTS_UPSCALING = False     # Not planned
    SUPPORTS_SCENE_DETECTION = False  # Not planned
    
    # Supported models (placeholder - will be updated when implemented)
    SUPPORTED_MODELS: Dict[str, List[str]] = {}
    
    def __init__(
        self,
        config: BackendConfig,
        parent: Optional[Any] = None,
    ):
        """Initialize the DirectML backend.
        
        Args:
            config: Backend configuration
            parent: Parent QObject for Qt signal handling
        """
        super().__init__(config, parent)
        self._model = None
        self._session = None  # ONNX DirectML session (placeholder)
        self._device_id = 0
        self._cancelled = False
        logger.warning("DirectMLBackend is a placeholder - not yet implemented")
    
    def initialize(self) -> bool:
        """Initialize the backend.
        
        PLACEHOLDER: Always returns False as backend is not implemented.
        
        Returns:
            False - backend not yet implemented
        """
        logger.error("DirectMLBackend.initialize() not implemented")
        return False
    
    def process(
        self,
        video_path: str,
        processing_config: ProcessingConfig,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> ProcessingResult:
        """Process a video file.
        
        PLACEHOLDER: Raises NotImplementedError.
        
        Args:
            video_path: Path to input video
            processing_config: Processing pipeline configuration
            progress_callback: Callback for progress updates
            stage_callback: Callback for stage changes
            log_callback: Callback for log messages
            
        Returns:
            ProcessingResult with error message
        """
        raise NotImplementedError(
            "DirectMLBackend is not yet implemented. "
            "Use BackendType.TORCH for now."
        )
    
    def process_frames(
        self,
        video_path: str,
        processing_config: ProcessingConfig,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Generator[Tuple[int, Any, Dict[str, Any]], None, None]:
        """Process video and yield frames one by one.
        
        PLACEHOLDER: Raises NotImplementedError.
        
        Args:
            video_path: Path to input video
            processing_config: Processing pipeline configuration
            progress_callback: Callback for progress updates
            stage_callback: Callback for stage changes
            log_callback: Callback for log messages
            
        Yields:
            Tuple of (frame_index, frame_data, metadata)
        """
        raise NotImplementedError(
            "DirectMLBackend is not yet implemented. "
            "Use BackendType.TORCH for now."
        )
    
    def cancel(self) -> None:
        """Cancel the current processing operation.
        
        PLACEHOLDER: Sets cancel flag but backend is not functional.
        """
        self._cancelled = True
        logger.warning("DirectMLBackend.cancel() called on placeholder backend")
    
    def cleanup(self) -> None:
        """Clean up resources after processing.
        
        PLACEHOLDER: Clears model and session references.
        """
        self._model = None
        self._session = None
        self._cancelled = False
        logger.warning("DirectMLBackend.cleanup() called on placeholder backend")