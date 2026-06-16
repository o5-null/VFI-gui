"""NCNN backend for video processing.

NCNN is a high-performance neural network inference framework optimized
for mobile devices and embedded systems. This backend provides CPU-based
inference with excellent cross-platform compatibility.

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


class NCNNBackend(BaseBackend):
    """NCNN-based video processing backend.
    
    PLACEHOLDER: This backend is not yet implemented.
    Will use NCNN framework for CPU-optimized inference.
    
    Features (planned):
    - Mobile/embedded optimized inference
    - Cross-platform CPU support (ARM, x86)
    - Vulkan GPU acceleration
    - INT8/FP16 quantization
    - Small memory footprint
    
    Limitations:
    - Requires model conversion to NCNN format (.param/.bin)
    - Limited model support compared to PyTorch
    """
    
    # Backend metadata
    BACKEND_TYPE = BackendType.NCNN
    BACKEND_NAME = "NCNN"
    BACKEND_DESCRIPTION = "NCNN mobile/embedded inference backend (placeholder)"
    
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
        """Initialize the NCNN backend.
        
        Args:
            config: Backend configuration
            parent: Parent QObject for Qt signal handling
        """
        super().__init__(config, parent)
        self._model = None
        self._net = None  # NCNN Net object (placeholder)
        self._cancelled = False
        logger.warning("NCNNBackend is a placeholder - not yet implemented")
    
    def initialize(self) -> bool:
        """Initialize the backend.
        
        PLACEHOLDER: Always returns False as backend is not implemented.
        
        Returns:
            False - backend not yet implemented
        """
        logger.error("NCNNBackend.initialize() not implemented")
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
            "NCNNBackend is not yet implemented. "
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
            "NCNNBackend is not yet implemented. "
            "Use BackendType.TORCH for now."
        )
    
    def cancel(self) -> None:
        """Cancel the current processing operation.
        
        PLACEHOLDER: Sets cancel flag but backend is not functional.
        """
        self._cancelled = True
        logger.warning("NCNNBackend.cancel() called on placeholder backend")
    
    def cleanup(self) -> None:
        """Clean up resources after processing.
        
        PLACEHOLDER: Clears model reference.
        """
        self._model = None
        self._net = None
        self._cancelled = False
        logger.warning("NCNNBackend.cleanup() called on placeholder backend")