"""PyTorch backend for video processing (refactored with IO components).

This backend uses the centralized IO components for all input/output operations:
- VideoFrameReader for reading input
- VideoFrameWriter for writing output
- FrameData for carrying data between components

The backend only handles:
- Model loading and inference
- Frame interpolation processing
- Data transformation
"""

import gc
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
from loguru import logger

from . import (
    BackendType,
    BaseBackend,
    BackendConfig,
    ProcessingConfig,
    ProcessingResult,
)
from core.io import (
    FrameReaderFactory,
    FrameWriterFactory,
    VideoFrameSequence,
    ProcessedFrameData,
    VideoMetadata,
    FrameFormat,
)


class TorchBackend(BaseBackend):
    """PyTorch-based video processing backend.
    
    Uses centralized IO components for all input/output operations.
    Only handles model inference and frame processing.
    """
    
    BACKEND_TYPE = BackendType.TORCH
    BACKEND_NAME = "PyTorch"
    BACKEND_DESCRIPTION = "Pure PyTorch inference backend"
    
    SUPPORTS_INTERPOLATION = True
    SUPPORTS_UPSCALING = False
    SUPPORTS_SCENE_DETECTION = False
    
    SUPPORTED_MODELS = {
        "rife": ["4.0", "4.6", "4.7", "4.17", "4.22", "4.26"],
        "film": ["fp32"],
        "ifrnet": ["S_Vimeo90K", "L_Vimeo90K"],
        "amt": ["s", "l", "g"],
    }
    
    def __init__(
        self,
        config: BackendConfig,
        parent=None,
    ):
        super().__init__(config, parent)
        self._model = None
        self._cancelled = False
        self._output_path: Optional[str] = None
    
    def initialize(self) -> bool:
        """Initialize the PyTorch backend."""
        try:
            device = self._config.get_device()
            logger.info(f"Initializing PyTorch backend on device: {device}")
            
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self._config.device = "cpu"
            
            self._is_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"PyTorch not available: {e}")
            return False
    
    def process(
        self,
        video_path: str,
        processing_config: ProcessingConfig,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> ProcessingResult:
        """Process a video file using PyTorch."""
        start_time = time.time()
        self._cancelled = False
        
        def emit_stage(stage: str):
            if stage_callback:
                stage_callback(stage)
        
        def emit_log(msg: str):
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        
        def emit_progress(current: int, total: int, fps: float = 0.0):
            if progress_callback:
                progress_callback(current, total, fps)
        
        try:
            # Setup output path
            self._output_path = self._generate_output_path(video_path, processing_config)
            
            emit_stage("Loading model...")
            emit_log(f"Input: {video_path}")
            emit_log(f"Output: {self._output_path}")
            
            # Load model
            self._load_model(processing_config, emit_log)
            
            if self._cancelled:
                return ProcessingResult(success=False, error_message="Cancelled by user")
            
            # Read input frames using IO component
            emit_stage("Reading input...")
            frame_sequence = self._read_input(video_path, emit_log, emit_progress)
            
            if self._cancelled:
                return ProcessingResult(success=False, error_message="Cancelled by user")
            
            # Process frames
            emit_stage("Processing...")
            processed_frames = self._process_frames(
                frame_sequence,
                processing_config,
                emit_progress,
            )
            
            if self._cancelled:
                return ProcessingResult(success=False, error_message="Cancelled by user")
            
            # Write output frames using IO component
            emit_stage("Writing output...")
            self._write_output(
                processed_frames,
                frame_sequence.metadata,
                emit_log,
                emit_progress,
            )
            
            processing_time = time.time() - start_time
            emit_log(f"Processing complete in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                output_path=self._output_path,
                processing_time=processing_time,
            )
            
        except Exception as e:
            logger.exception(f"Processing error: {e}")
            return ProcessingResult(success=False, error_message=str(e))
        
        finally:
            self._cleanup()
    
    def _generate_output_path(
        self,
        video_path: str,
        processing_config: ProcessingConfig,
    ) -> str:
        """Generate output file path."""
        import re
        
        input_path = Path(video_path)
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_config = processing_config.output
        output_mode = output_config.get("output_mode", "video")
        
        if output_mode == "images":
            image_format = output_config.get("image_format", "png")
            output_ext = f".{image_format}"
        else:
            output_ext = ".mp4"
        
        # Handle image sequence input
        if "%" in video_path:
            match = re.search(r'/([^/]+?)%0\d+d', video_path.replace("\\", "/"))
            if match:
                base_name = match.group(1).rstrip('_.-')
                if not base_name:
                    base_name = input_path.parent.name
            else:
                base_name = input_path.parent.name
            output_name = f"{base_name}_torch_interpolated{output_ext}"
        else:
            output_name = f"{input_path.stem}_torch_interpolated{output_ext}"
        
        return str(output_dir / output_name)
    
    def _read_input(
        self,
        video_path: str,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int, int, float], None],
    ) -> VideoFrameSequence:
        """Read input video using IO component."""
        device = self._config.get_device()
        
        # Create reader using factory
        reader = FrameReaderFactory.create_reader(
            source=video_path,
            target_format=FrameFormat.TENSOR_NHWC,
            device=device,
        )
        
        # Read all frames
        def on_progress(current: int, total: int):
            progress_callback(current, total, 0.0)
        
        frame_sequence = reader.read_frames(video_path, on_progress)
        
        log_callback(
            f"Video: {frame_sequence.metadata.width}x{frame_sequence.metadata.height} "
            f"@ {frame_sequence.metadata.fps:.2f} fps, "
            f"{len(frame_sequence)} frames"
        )
        
        return frame_sequence
    
    def _process_frames(
        self,
        frame_sequence: VideoFrameSequence,
        processing_config: ProcessingConfig,
        progress_callback: Callable[[int, int, float], None],
    ) -> Iterator[ProcessedFrameData]:
        """Process frames using model inference."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        interp_config = processing_config.interpolation
        multiplier = interp_config.get("multi", 2)
        
        device = self._config.get_device()
        
        # Get frames as tensor [N, H, W, C]
        frames_tensor = frame_sequence.to_tensor(device)
        
        # Preprocess: NHWC -> NCHW
        frames_chw = frames_tensor.permute(0, 3, 1, 2)  # [N, C, H, W]
        
        n = len(frames_chw)
        output_idx = 0
        
        # Process frame pairs
        for i in range(n - 1):
            if self._cancelled:
                break
            
            frame0 = frames_chw[i:i+1]  # [1, C, H, W]
            frame1 = frames_chw[i+1:i+2]
            
            # Add original frame
            frame_nhwc = frame0.permute(0, 2, 3, 1).squeeze(0)  # [H, W, C]
            yield ProcessedFrameData(
                data=frame_nhwc,
                source_frame_idx=i,
                interpolated=False,
                interpolation_ratio=0.0,
            )
            output_idx += 1
            
            # Generate interpolated frames
            for j in range(1, multiplier):
                ratio = j / multiplier
                
                with torch.no_grad():
                    result = self._model.interpolate(frame0, frame1, ratio)
                
                result_nhwc = result.permute(0, 2, 3, 1).squeeze(0)
                yield ProcessedFrameData(
                    data=result_nhwc,
                    source_frame_idx=i,
                    interpolated=True,
                    interpolation_ratio=ratio,
                )
                output_idx += 1
            
            if i % 10 == 0:
                progress_callback(i, n, 0.0)
        
        # Add last frame
        if not self._cancelled and n > 0:
            last_frame = frames_chw[-1:].permute(0, 2, 3, 1).squeeze(0)
            yield ProcessedFrameData(
                data=last_frame,
                source_frame_idx=n - 1,
                interpolated=False,
                interpolation_ratio=1.0,
            )
        
        logger.info(f"Generated {output_idx + 1} output frames")
    
    def _write_output(
        self,
        frames: Iterator[ProcessedFrameData],
        input_metadata: VideoMetadata,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int, int, float], None],
    ) -> None:
        """Write output using IO component."""
        if self._output_path is None:
            raise RuntimeError("Output path not set")
        
        # Calculate output metadata
        interp_config = {"multi": 2}  # Default
        output_fps = input_metadata.fps * 2  # Simplified
        
        output_metadata = VideoMetadata(
            width=input_metadata.width,
            height=input_metadata.height,
            fps=output_fps,
            total_frames=input_metadata.total_frames * 2,  # Estimate
        )
        
        # Create writer using factory
        codec_config = getattr(self, '_codec_config', None)
        writer = FrameWriterFactory.create_writer(self._output_path, codec_config)
        
        # Write frames
        def on_progress(current: int, total: int):
            progress_callback(current, total, 0.0)
        
        writer.write_frames(
            frames,
            self._output_path,
            output_metadata,
            on_progress,
        )
        
        log_callback(f"Output saved to: {self._output_path}")
    
    def _load_model(
        self,
        config: ProcessingConfig,
        log_callback: Callable[[str], None],
    ) -> None:
        """Load the interpolation model."""
        from ..torch_backend import VFIConfig, ModelType, get_model
        
        interp_config = config.interpolation
        model_type_str = interp_config.get("model_type", "rife").lower()
        model_version = interp_config.get("model_version", "")
        
        if not model_version and model_type_str in self.SUPPORTED_MODELS:
            model_version = self.SUPPORTED_MODELS[model_type_str][0]
        
        log_callback(f"Loading {model_type_str} model: {model_version}")
        
        model_type_map = {
            "rife": ModelType.RIFE,
            "film": ModelType.FILM,
            "ifrnet": ModelType.IFRNET,
            "amt": ModelType.AMT,
        }
        model_type = model_type_map.get(model_type_str, ModelType.RIFE)
        
        vfi_config = VFIConfig(
            model_type=model_type,
            model_version=model_version,
            multiplier=interp_config.get("multi", 2),
            scale=interp_config.get("scale", 1.0),
            fp16=self._config.fp16,
        )
        
        checkpoint_path = interp_config.get("checkpoint_path")
        if not checkpoint_path:
            models_dir = self._config.models_dir
            checkpoint_name = self._get_checkpoint_name(model_type_str, model_version)
            checkpoint_path = str(Path(models_dir) / model_type_str / checkpoint_name)
        
        log_callback(f"Checkpoint: {checkpoint_path}")
        
        self._model = get_model(model_type, vfi_config)
        self._model.load_model(checkpoint_path)
        
        log_callback(f"Model loaded: {model_type_str}")
    
    def _get_checkpoint_name(self, model_type: str, version: str) -> str:
        """Get checkpoint filename for a model version."""
        checkpoint_map = {
            "rife": {
                "4.0": "sudo_rife4_269.662_testV1_scale1.pth",
                "4.6": "flownet.pkl",
                "4.7": "rife47.pth",
                "4.17": "rife417.pth",
                "4.22": "rife49.pth",
                "4.26": "rife426.pth",
            },
            "film": {"fp32": "film_net_fp32.pt"},
            "ifrnet": {
                "S_Vimeo90K": "IFRNet_S_Vimeo90K.pth",
                "L_Vimeo90K": "IFRNet_L_Vimeo90K.pth",
            },
            "amt": {"s": "amt-s.pth", "l": "amt-l.pth", "g": "amt-g.pth"},
        }
        
        if model_type in checkpoint_map and version in checkpoint_map[model_type]:
            return checkpoint_map[model_type][version]
        return f"{model_type}_{version}.pth"
    
    def cancel(self) -> None:
        """Cancel processing."""
        self._cancelled = True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Internal cleanup."""
        if self._model is not None:
            try:
                self._model.unload()
            except Exception:
                pass
            self._model = None
        
        try:
            from ..torch_backend import clear_cache
            clear_cache()
        except Exception:
            pass
        
        gc.collect()
