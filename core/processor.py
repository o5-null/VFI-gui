"""Video processing engine with multi-backend support.

.. deprecated::
    Use :class:`TaskOrchestrator <core.task_orchestrator.TaskOrchestrator>` instead.
    The Processor class is kept for backward compatibility only and will be
    removed in a future version.

This module provides a unified video processing interface that supports
multiple backends (VapourSynth, PyTorch, etc.) through a common API.

Usage:
    from core.processor import Processor, BackendType, BackendConfig
    
    config = BackendConfig(backend_type=BackendType.TORCH)
    processor = Processor(config)
    result = processor.process("input.mp4", processing_config)
"""

import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from loguru import logger

from .backends import (
    BackendType,
    BackendConfig,
    ProcessingConfig,
    ProcessingResult,
    BaseBackend,
    BackendFactory,
)
from .i18n import tr


# Supported interpolation models (unified across backends)
SUPPORTED_INTERPOLATION_MODELS = {
    "rife": {
        "name": "RIFE",
        "description": "Real-Time Intermediate Flow Estimation",
        "versions": ["4.0", "4.6", "4.7", "4.17", "4.22", "4.26", "4.26.heavy"],
        "default_version": "4.22",
    },
    "film": {
        "name": "FILM",
        "description": "Frame Interpolation for Large Motion",
        "versions": ["fp32"],
        "default_version": "fp32",
    },
    "ifrnet": {
        "name": "IFRNet",
        "description": "Intermediate Feature Refine Network",
        "versions": ["S_Vimeo90K", "L_Vimeo90K", "S_GoPro", "L_GoPro"],
        "default_version": "L_Vimeo90K",
    },
    "amt": {
        "name": "AMT",
        "description": "All-Pairs Multi-Field Transforms",
        "versions": ["s", "l", "g"],
        "default_version": "s",
    },
}


class Processor(QThread):
    """Unified video processor with multi-backend support.

    .. deprecated::
        Use :class:`TaskOrchestrator <core.task_orchestrator.TaskOrchestrator>` instead.
        The Processor class is kept for backward compatibility only.

    This class provides a Qt-compatible interface for video processing,
    supporting multiple backends through a unified API.
    
    Signals:
        progress_updated: (current_frame, total_frames, fps)
        stage_changed: (stage_name)
        log_message: (message)
        finished: (output_path)
        error_occurred: (error_message)
    
    Usage:
        config = BackendConfig(backend_type=BackendType.TORCH)
        processor = Processor(config)
        processor.set_video("input.mp4")
        processor.set_processing_config(processing_config)
        processor.start()
    """
    
    # Signals
    progress_updated = pyqtSignal(int, int, float)  # current_frame, total_frames, fps
    stage_changed = pyqtSignal(str)  # stage name
    log_message = pyqtSignal(str)  # log message
    finished = pyqtSignal(str)  # output path
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(
        self,
        backend_config: Optional[BackendConfig] = None,
        parent: Optional[QObject] = None,
    ):
        """Initialize the processor.
        
        .. deprecated:: Use TaskOrchestrator instead.

        Args:
            backend_config: Backend configuration. Uses VapourSynth if None.
            parent: Parent QObject
        """
        warnings.warn(
            "Processor is deprecated. Use TaskOrchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(parent)
        
        self._backend_config = backend_config or BackendConfig()
        self._processing_config = ProcessingConfig()
        self._video_path: Optional[str] = None
        self._output_path: Optional[str] = None
        
        self._backend: Optional[BaseBackend] = None
        self._paused = False
        self._cancelled = False
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self) -> None:
        """Initialize the processing backend."""
        try:
            self._backend = BackendFactory.create(
                self._backend_config.backend_type,
                self._backend_config,
                self,
            )
            logger.info(f"Initialized backend: {self._backend}")
        except ValueError as e:
            logger.error(f"Failed to initialize backend: {e}")
            # Fall back to VapourSynth if available
            if self._backend_config.backend_type != BackendType.VAPOURSYNTH:
                logger.info("Falling back to VapourSynth backend")
                self._backend_config.backend_type = BackendType.VAPOURSYNTH
                self._backend = BackendFactory.create(
                    BackendType.VAPOURSYNTH,
                    self._backend_config,
                    self,
                )
    
    def set_video(self, video_path: str) -> None:
        """Set the input video path.
        
        Args:
            video_path: Path to input video file
        """
        self._video_path = video_path
        logger.debug(f"Set input video: {video_path}")
    
    def set_processing_config(self, config: ProcessingConfig) -> None:
        """Set the processing configuration.
        
        Args:
            config: Processing pipeline configuration
        """
        self._processing_config = config
        logger.debug(f"Set processing config: {config}")
    
    def set_backend_config(self, config: BackendConfig) -> None:
        """Set the backend configuration.
        
        Note: This will reinitialize the backend.
        
        Args:
            config: Backend configuration
        """
        self._backend_config = config
        self._init_backend()
        logger.debug(f"Set backend config: {config}")
    
    def run(self) -> None:
        """Run the video processing pipeline."""
        if not self._video_path:
            self.error_occurred.emit("No video path set")
            return

        if not self._backend:
            self.error_occurred.emit("No backend initialized")
            return

        try:
            logger.info(f"Starting video processing: {self._video_path}")
            self.stage_changed.emit(tr("Initializing..."))
            self.log_message.emit(f"{tr('Input')}: {self._video_path}")
            self.log_message.emit(f"{tr('Backend')}: {self._backend.BACKEND_NAME}")

            # Initialize backend
            if not self._backend.initialize():
                raise RuntimeError(f"{tr('Failed to initialize')} {self._backend.BACKEND_NAME} {tr('backend')}")

            # Setup output path
            input_path = Path(self._video_path)
            output_config = self._processing_config.output

            # Determine output directory
            custom_output_dir = output_config.get("output_dir", "")
            custom_subdir = output_config.get("output_subdir", "")

            if custom_output_dir:
                # Use custom output directory
                output_dir = Path(custom_output_dir)
            else:
                # Use default output directory from backend config
                output_dir = Path(self._backend_config.output_dir)

            # Add custom subdirectory if specified
            if custom_subdir:
                output_dir = output_dir / custom_subdir

            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine output mode and extension
            output_mode = output_config.get("output_mode", "video")

            if output_mode == "images":
                image_format = output_config.get("image_format", "png")
                output_ext = f".{image_format}"
            else:
                output_ext = ".mp4"

            # Generate output filename
            custom_filename = output_config.get("output_filename", "")

            if custom_filename:
                # Use custom filename pattern
                # Replace placeholders: {input} = input filename, {backend} = backend type
                base_name = custom_filename.replace("{input}", input_path.stem)
                base_name = base_name.replace("{backend}", self._backend.BACKEND_TYPE.value)
                output_name = f"{base_name}{output_ext}"
            else:
                # Auto-generate filename
                if "%" in self._video_path:
                    import re
                    match = re.search(r'/([^/]+?)%0\d+d', self._video_path.replace("\\", "/"))
                    if match:
                        base_name = match.group(1).rstrip('_.-')
                        if not base_name:
                            base_name = input_path.parent.name
                    else:
                        base_name = input_path.parent.name
                    output_name = f"{base_name}_{self._backend.BACKEND_TYPE.value}_interpolated{output_ext}"
                else:
                    output_name = f"{input_path.stem}_{self._backend.BACKEND_TYPE.value}_interpolated{output_ext}"

            self._output_path = str(output_dir / output_name)

            # Process video using frame generator
            start_time = time.time()

            # Collect frames from generator
            frames = []
            metadata = {}

            for frame_idx, frame_data, meta in self._backend.process_frames(
                video_path=self._video_path,
                processing_config=self._processing_config,
                progress_callback=self._on_progress,
                stage_callback=self._on_stage,
                log_callback=self._on_log,
            ):
                if self._cancelled:
                    logger.info("Processing cancelled by user")
                    return

                frames.append(frame_data)
                metadata = meta

            if self._cancelled:
                return

            # Save frames using frame_writer
            self.stage_changed.emit(tr("Saving..."))
            self._save_frames(frames, metadata)

            processing_time = time.time() - start_time

            self.stage_changed.emit(tr("Complete"))
            self.log_message.emit(f"{tr('Output')}: {self._output_path}")
            self.log_message.emit(f"{tr('Processing time')}: {processing_time:.2f}s")
            logger.info(f"Processing complete: {self._output_path}")
            self.finished.emit(self._output_path)

        except Exception as e:
            logger.exception(f"Processing error: {e}")
            self.error_occurred.emit(str(e))

        finally:
            if self._backend:
                self._backend.cleanup()

    def _save_frames(
        self,
        frames: List[np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        """Save processed frames using frame_writer.

        Args:
            frames: List of frame data (RGB uint8 numpy arrays)
            metadata: Video metadata including fps, width, height, total_frames
        """
        from core.io.frame_writer import FrameWriterFactory
        from core.io.frame_data import ProcessedFrameData, VideoMetadata
        from core.codec_manager import CodecConfig

        # Create VideoMetadata
        video_meta = VideoMetadata(
            width=metadata.get("width", 1920),
            height=metadata.get("height", 1080),
            fps=metadata.get("fps", 30.0),
            total_frames=len(frames),
            duration=len(frames) / metadata.get("fps", 30.0),
        )

        # Create codec config from processing config
        output_config = self._processing_config.output
        codec_config = CodecConfig.from_dict(output_config)

        # Check output mode for appropriate message
        output_mode = output_config.get("output_mode", "video")
        is_image_output = output_mode == "images"

        # Create appropriate writer
        writer = FrameWriterFactory.create_writer(
            output_path=self._output_path,
            codec_config=output_config,
        )

        # Convert frames to ProcessedFrameData iterator
        def frame_iterator():
            for i, frame in enumerate(frames):
                yield ProcessedFrameData(
                    data=frame,
                    source_frame_idx=i,
                    interpolated=False,
                )

        # Write frames with appropriate progress message
        if is_image_output:
            image_format = output_config.get("image_format", "png").upper()
            self.log_message.emit(tr("Saving {} frames as {} images...").format(len(frames), image_format))
        else:
            codec = output_config.get("codec", "unknown")
            self.log_message.emit(tr("Encoding {} frames to video ({})...").format(len(frames), codec))

        writer.write_frames(
            frames=frame_iterator(),
            output_path=self._output_path,
            metadata=video_meta,
            progress_callback=lambda current, total: self._on_progress(current, total, 0.0),
        )
        writer.close()

        if is_image_output:
            self.log_message.emit(tr("Saved {} images to: {}").format(len(frames), Path(self._output_path).parent))
        else:
            self.log_message.emit(tr("Saved video to: {}").format(self._output_path))
    
    def _on_progress(self, current: int, total: int, fps: float) -> None:
        """Handle progress updates from backend."""
        if not self._cancelled:
            self.progress_updated.emit(current, total, fps)
    
    def _on_stage(self, stage: str) -> None:
        """Handle stage changes from backend."""
        self.stage_changed.emit(stage)
    
    def _on_log(self, message: str) -> None:
        """Handle log messages from backend."""
        self.log_message.emit(message)
    
    def pause(self) -> None:
        """Pause processing."""
        self._paused = True
        logger.info("Processing paused")
    
    def resume(self) -> None:
        """Resume processing."""
        self._paused = False
        logger.info("Processing resumed")
    
    def cancel(self, force: bool = False, timeout: float = 5.0) -> None:
        """Cancel processing.

        Args:
            force: Whether to force terminate worker threads immediately
            timeout: Timeout for graceful shutdown before forcing termination
        """
        self._cancelled = True
        if self._backend:
            # Check if backend supports force cancel
            import inspect
            sig = inspect.signature(self._backend.cancel)
            if 'force' in sig.parameters:
                self._backend.cancel(force=force, timeout=timeout)
            else:
                self._backend.cancel()
        logger.info(f"Processing cancelled (force={force}, timeout={timeout})")
    
    def is_paused(self) -> bool:
        """Check if processing is paused."""
        return self._paused
    
    def is_running(self) -> bool:
        """Check if processing is running."""
        return self.isRunning() and not self._cancelled
    
    @property
    def backend_type(self) -> BackendType:
        """Get the current backend type."""
        return self._backend_config.backend_type
    
    @property
    def backend_name(self) -> str:
        """Get the current backend name."""
        if self._backend:
            return self._backend.BACKEND_NAME
        return "none"
    
    @staticmethod
    def get_supported_models() -> Dict[str, Dict[str, Any]]:
        """Get list of supported interpolation models."""
        return SUPPORTED_INTERPOLATION_MODELS.copy()
    
    @staticmethod
    def get_model_versions(model_type: str) -> List[str]:
        """Get available versions for a model type."""
        model_type = model_type.lower()
        if model_type in SUPPORTED_INTERPOLATION_MODELS:
            return SUPPORTED_INTERPOLATION_MODELS[model_type]["versions"].copy()
        return []
    
    @staticmethod
    def get_available_backends() -> List[BackendType]:
        """Get list of available backend types."""
        return BackendFactory.get_available_backends()
    
    @staticmethod
    def get_backend_info(backend_type: BackendType) -> Dict[str, Any]:
        """Get information about a backend type."""
        return BackendFactory.get_backend_info(backend_type)


# Backward compatibility alias
VideoProcessor = Processor


def register_backend(backend_class: type) -> None:
    """Register a backend class.
    
    Args:
        backend_class: Backend class to register (must inherit from BaseBackend)
    """
    if not issubclass(backend_class, BaseBackend):
        raise TypeError(f"{backend_class} must inherit from BaseBackend")
    
    BackendFactory.register(backend_class.BACKEND_TYPE, backend_class)
    logger.info(f"Registered backend: {backend_class.BACKEND_TYPE.value}")
