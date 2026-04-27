"""PyTorch backend for video processing.

This backend uses the torch_backend module for video frame interpolation,
providing pure PyTorch inference without VapourSynth dependency.

Supports multi-threaded inference for improved performance.
"""

import gc
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from . import (
    BackendType,
    BaseBackend,
    BackendConfig,
    ProcessingConfig,
    ProcessingResult,
)
from .inference_thread_pool import InferenceThreadPool, InferenceTask


class TorchBackend(BaseBackend):
    """PyTorch-based video processing backend.
    
    Uses the torch_backend module for frame interpolation with
    support for RIFE, FILM, IFRNet, and AMT models.
    """
    
    # Backend metadata
    BACKEND_TYPE = BackendType.TORCH
    BACKEND_NAME = "PyTorch"
    BACKEND_DESCRIPTION = "Pure PyTorch inference backend"
    
    # Supported features
    SUPPORTS_INTERPOLATION = True
    SUPPORTS_UPSCALING = False  # Not implemented yet
    SUPPORTS_SCENE_DETECTION = False  # Not implemented yet
    
    # Supported models
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
        
        # Multi-threading support
        self._thread_pool: Optional[InferenceThreadPool] = None
        self._num_inference_threads = config.extra.get("inference_threads", 1)
        self._use_threading = self._num_inference_threads > 1
    
    def initialize(self) -> bool:
        """Initialize the PyTorch backend."""
        try:
            import torch
            
            # Check device
            device = self._config.get_device()
            logger.info(f"Initializing PyTorch backend on device: {device}")
            
            # Verify CUDA availability if using CUDA
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
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
        """Process a video file using PyTorch.
        
        Args:
            video_path: Path to input video
            processing_config: Processing pipeline configuration
            progress_callback: Progress callback
            stage_callback: Stage change callback
            log_callback: Log message callback
            
        Returns:
            ProcessingResult with output path or error
        """
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
            # Setup paths
            input_path = Path(video_path)
            output_dir = Path(self._config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine output extension based on output mode
            output_config = processing_config.output
            output_mode = output_config.get("output_mode", "video")
            
            if output_mode == "images":
                # Image sequence output - use the specified format
                image_format = output_config.get("image_format", "png")
                output_ext = f".{image_format}"
            else:
                # Video output
                output_ext = ".mp4"
            
            # Generate output name
            # For image sequences (containing %0Xd pattern), use parent directory name
            if "%" in video_path:
                # Image sequence: extract meaningful name from path
                # e.g., "E:/path/to/images/%04d.png" -> use "images" as base name
                # or "E:/path/to/prefix_%04d.png" -> use "prefix" as base name
                import re
                match = re.search(r'/([^/]+?)%0\d+d', video_path.replace("\\", "/"))
                if match:
                    base_name = match.group(1).rstrip('_.-')
                    if not base_name:
                        # Use parent directory name
                        base_name = input_path.parent.name
                else:
                    base_name = input_path.parent.name
                output_name = f"{base_name}_torch_interpolated{output_ext}"
            else:
                output_name = f"{input_path.stem}_torch_interpolated{output_ext}"
            self._output_path = str(output_dir / output_name)
            
            emit_stage("Loading model...")
            emit_log(f"Input: {video_path}")
            
            # Load model
            self._load_model(processing_config, emit_log)
            
            if self._cancelled:
                return ProcessingResult(
                    success=False,
                    error_message="Cancelled by user",
                )
            
            # Process video
            emit_stage("Processing...")
            self._process_video(
                video_path,
                processing_config,
                emit_progress,
                emit_log,
            )
            
            if self._cancelled:
                return ProcessingResult(
                    success=False,
                    error_message="Cancelled by user",
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
            return ProcessingResult(
                success=False,
                error_message=str(e),
            )
        
        finally:
            self._cleanup()

    def process_frames(
        self,
        video_path: str,
        processing_config: ProcessingConfig,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Generator[Tuple[int, np.ndarray, Dict[str, Any]], None, None]:
        """Process video and yield frames one by one.

        This method separates processing from IO, yielding processed frames
        for the caller to handle output writing.

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
                - metadata: dict with 'fps', 'width', 'height', 'total_frames', 'multiplier'
        """
        import cv2
        import torch

        from ..torch_backend import clear_cache

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
            emit_stage("Loading model...")
            emit_log(f"Input: {video_path}")

            # Load model
            self._load_model(processing_config, emit_log)

            if self._cancelled:
                return

            emit_stage("Processing...")

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            emit_log(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

            # Read all frames
            frames = []
            frame_idx = 0

            while True:
                if self._cancelled:
                    cap.release()
                    return

                ret, frame = cap.read()
                if not ret:
                    break

                # BGR to RGB, normalize to [0, 1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frames.append(frame_tensor)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    emit_log(f"Read {frame_idx}/{total_frames} frames")

            cap.release()

            if self._cancelled:
                return

            # Stack frames [N, H, W, C]
            frames_tensor = torch.stack(frames, dim=0)
            emit_log(f"Read complete: {len(frames)} frames, shape: {frames_tensor.shape}")

            # Interpolate
            interp_config = processing_config.interpolation
            multiplier = interp_config.get("multi", 2)

            output_frames = self._interpolate_frames(
                frames_tensor,
                multiplier,
                emit_progress,
            )

            emit_log(f"Interpolation complete: {len(output_frames)} frames")

            if self._cancelled:
                return

            # Prepare metadata
            metadata = {
                "fps": fps * multiplier,
                "width": width,
                "height": height,
                "total_frames": len(output_frames),
                "multiplier": multiplier,
                "input_fps": fps,
            }

            # Yield frames one by one
            for i, frame in enumerate(output_frames):
                if self._cancelled:
                    break

                # Convert to uint8 numpy array [H, W, C] RGB
                frame_uint8 = (frame.clamp(0, 1) * 255).byte().cpu().numpy()

                yield (i, frame_uint8, metadata)

        except Exception as e:
            logger.exception(f"Processing error: {e}")
            raise

        finally:
            self._cleanup()

    def _load_model(
        self,
        config: ProcessingConfig,
        log_callback: Callable[[str], None],
    ) -> None:
        """Load the interpolation model."""
        from ..torch_backend import (
            VFIConfig,
            ModelType,
            get_model,
        )
        
        interp_config = config.interpolation
        model_type_str = interp_config.get("model_type", "rife").lower()
        model_version = interp_config.get("model_version", "")
        
        # Get default version if not specified
        if not model_version and model_type_str in self.SUPPORTED_MODELS:
            model_version = self.SUPPORTED_MODELS[model_type_str][0]
        
        log_callback(f"Loading {model_type_str} model: {model_version}")
        
        # Map model type string to enum
        model_type_map = {
            "rife": ModelType.RIFE,
            "film": ModelType.FILM,
            "ifrnet": ModelType.IFRNET,
            "amt": ModelType.AMT,
        }
        
        model_type = model_type_map.get(model_type_str, ModelType.RIFE)
        
        # Create config
        vfi_config = VFIConfig(
            model_type=model_type,
            model_version=model_version,
            multiplier=interp_config.get("multi", 2),
            scale=interp_config.get("scale", 1.0),
            fp16=self._config.fp16,
        )
        
        # Get model path - prefer checkpoint_path from config (absolute path)
        checkpoint_path = interp_config.get("checkpoint_path")
        
        if not checkpoint_path:
            # Fallback to building path from models_dir
            models_dir = self._config.models_dir
            checkpoint_name = self._get_checkpoint_name(model_type_str, model_version)
            checkpoint_path = str(Path(models_dir) / model_type_str / checkpoint_name)
        
        log_callback(f"Checkpoint: {checkpoint_path}")
        
        # Create and load model
        self._model = get_model(model_type, vfi_config)
        self._model.load_model(checkpoint_path)
        
        # Save config for multi-threading
        self._current_model_type = model_type
        self._current_model_version = model_version
        self._current_checkpoint_path = checkpoint_path
        
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
            "film": {
                "fp32": "film_net_fp32.pt",
            },
            "ifrnet": {
                "S_Vimeo90K": "IFRNet_S_Vimeo90K.pth",
                "L_Vimeo90K": "IFRNet_L_Vimeo90K.pth",
            },
            "amt": {
                "s": "amt-s.pth",
                "l": "amt-l.pth",
                "g": "amt-g.pth",
            },
        }
        
        if model_type in checkpoint_map and version in checkpoint_map[model_type]:
            return checkpoint_map[model_type][version]
        return f"{model_type}_{version}.pth"
    
    def _process_video(
        self,
        video_path: str,
        config: ProcessingConfig,
        progress_callback: Callable[[int, int, float], None],
        log_callback: Callable[[str], None],
    ) -> None:
        """Process video frames."""
        import cv2
        import torch
        
        from ..torch_backend import (
            VFIResult,
            clear_cache,
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        log_callback(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
        
        # Read all frames
        frames = []
        frame_idx = 0
        
        while True:
            if self._cancelled:
                cap.release()
                return
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB, normalize to [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frames.append(frame_tensor)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                log_callback(f"Read {frame_idx}/{total_frames} frames")
        
        cap.release()
        
        if self._cancelled:
            return
        
        # Stack frames [N, H, W, C]
        frames_tensor = torch.stack(frames, dim=0)
        log_callback(f"Read complete: {len(frames)} frames, shape: {frames_tensor.shape}")
        
        # Interpolate
        interp_config = config.interpolation
        multiplier = interp_config.get("multi", 2)
        
        output_frames = self._interpolate_frames(
            frames_tensor,
            multiplier,
            progress_callback,
        )
        
        log_callback(f"Interpolation complete: {len(output_frames)} frames")
        
        if self._cancelled:
            return
        
        # Encode output
        self._encode_video(output_frames, fps, width, height, config, log_callback, progress_callback)
    
    def _interpolate_frames(
        self,
        frames,  # torch.Tensor
        multiplier: int,
        progress_callback: Callable[[int, int, float], None],
    ):  # -> torch.Tensor
        """Interpolate frame sequence.
        
        Supports both single-threaded and multi-threaded inference.
        
        Args:
            frames: Input frames tensor [N, H, W, C]
            multiplier: Interpolation multiplier
            progress_callback: Progress callback
            
        Returns:
            Output frames tensor
        """
        from ..torch_backend import get_device
        
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        # Get target device from model
        device = get_device()
        
        # Preprocess: NHWC -> NCHW, move to device
        frames_chw = frames.permute(0, 3, 1, 2).to(device)  # [N, C, H, W] on device
        
        # Use multi-threading if enabled
        if self._use_threading:
            return self._interpolate_frames_threaded(
                frames_chw, multiplier, progress_callback, device
            )
        else:
            return self._interpolate_frames_single(
                frames_chw, multiplier, progress_callback
            )
    
    def _interpolate_frames_single(
        self,
        frames_chw: torch.Tensor,
        multiplier: int,
        progress_callback: Callable[[int, int, float], None],
    ) -> torch.Tensor:
        """Single-threaded frame interpolation.
        
        Args:
            frames_chw: Input frames tensor [N, C, H, W]
            multiplier: Interpolation multiplier
            progress_callback: Progress callback
            
        Returns:
            Output frames tensor [M, H, W, C]
        """
        n = len(frames_chw)
        output_frames = []
        clear_interval = self._config.extra.get("clear_cache_every", 10)
        
        for i in range(n - 1):
            if self._cancelled:
                break
            
            # Add original frame (from device tensor)
            output_frames.append(frames_chw[i].permute(1, 2, 0))  # [H, W, C]
            
            # Generate interpolated frames
            for j in range(1, multiplier):
                timestep = j / multiplier
                
                # Interpolate
                result = self._model.interpolate(
                    frames_chw[i],
                    frames_chw[i + 1],
                    timestep=timestep,
                )
                
                # Convert back to NHWC
                interp_frame = result.frame.permute(1, 2, 0)  # [H, W, C]
                output_frames.append(interp_frame)
            
            # Progress callback
            progress_callback(i + 1, n - 1, 0.0)
            
            # Clear cache periodically
            if (i + 1) % clear_interval == 0:
                from ..torch_backend import clear_cache
                clear_cache()
        
        # Add last frame (from device tensor)
        if not self._cancelled:
            output_frames.append(frames_chw[-1].permute(1, 2, 0))  # [H, W, C]
        
        return torch.stack(output_frames)
    
    def _interpolate_frames_threaded(
        self,
        frames_chw: torch.Tensor,
        multiplier: int,
        progress_callback: Callable[[int, int, float], None],
        device: torch.device,
    ) -> torch.Tensor:
        """Multi-threaded frame interpolation.
        
        Uses InferenceThreadPool for parallel processing.
        
        Args:
            frames_chw: Input frames tensor [N, C, H, W]
            multiplier: Interpolation multiplier
            progress_callback: Progress callback
            device: Device to run inference on
            
        Returns:
            Output frames tensor [M, H, W, C]
        """
        logger.info(f"Using multi-threaded inference with {self._num_inference_threads} workers")
        
        n = len(frames_chw)
        clear_interval = self._config.extra.get("clear_cache_every", 10)
        
        # Create model factory function
        def model_factory():
            # Create a new model instance for each worker
            model = self._create_model_for_threading()
            return model
        
        # Create and start thread pool
        self._thread_pool = InferenceThreadPool(
            num_workers=self._num_inference_threads,
            model_factory=model_factory,
            device=device,
            queue_size=self._config.extra.get("task_queue_size", 100),
        )
        self._thread_pool.start()
        
        try:
            # Submit all tasks
            task_ids = []
            total_tasks = (n - 1) * (multiplier - 1)
            
            for i in range(n - 1):
                for j in range(1, multiplier):
                    timestep = j / multiplier
                    
                    task = InferenceTask(
                        task_id=len(task_ids),
                        frame0=frames_chw[i],
                        frame1=frames_chw[i + 1],
                        timestep=timestep,
                    )
                    
                    if self._thread_pool.submit(task):
                        task_ids.append(len(task_ids))
                    
                    # Update progress
                    progress_callback(len(task_ids), total_tasks, 0.0)
                    
                    # Check cancellation
                    if self._cancelled:
                        break
                
                if self._cancelled:
                    break
            
            if self._cancelled:
                return torch.empty(0)
            
            # Collect results
            logger.info(f"Submitted {len(task_ids)} tasks, collecting results...")
            results = self._thread_pool.get_results(
                expected_count=len(task_ids),
                timeout=None,  # Wait for all results
            )
            
            logger.info(f"Collected {len(results)} results")
            
            # Build output frames list
            # Map: result.task_id -> interpolated frame
            # Need to reconstruct in correct order
            result_map = {r.task_id: r.frame for r in results if r.success}
            
            output_frames = []
            task_id = 0
            
            for i in range(n - 1):
                # Add original frame
                output_frames.append(frames_chw[i].permute(1, 2, 0))  # [H, W, C]
                
                # Add interpolated frames in order
                for j in range(1, multiplier):
                    if task_id in result_map:
                        interp_frame = result_map[task_id].permute(1, 2, 0)  # [H, W, C]
                        output_frames.append(interp_frame)
                    task_id += 1
                
                # Progress callback
                progress_callback(i + 1, n - 1, 0.0)
                
                # Clear cache periodically
                if (i + 1) % clear_interval == 0:
                    from ..torch_backend import clear_cache
                    clear_cache()
            
            # Add last frame
            output_frames.append(frames_chw[-1].permute(1, 2, 0))  # [H, W, C]
            
            return torch.stack(output_frames)
            
        finally:
            # Shutdown thread pool
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
    
    def _create_model_for_threading(self):
        """Create a model instance for use in worker threads.
        
        Returns:
            New model instance
        """
        # Import here to avoid circular imports
        from ..torch_backend import ModelType, get_model, VFIConfig
        
        # Get current model config
        if not hasattr(self, '_current_model_type') or not hasattr(self, '_current_model_version'):
            # Use defaults if not set
            model_type = ModelType.RIFE
            model_version = "4.22"
        else:
            model_type = self._current_model_type
            model_version = self._current_model_version
        
        # Create VFI config
        device = self._config.get_device()
        vfi_config = VFIConfig(
            model_type=model_type,
            model_version=model_version,
            device=device,
        )
        
        # Create and return model
        model = get_model(vfi_config)
        
        # Load weights
        if hasattr(model, 'load_model') and hasattr(self, '_current_checkpoint_path'):
            model.load_model(self._current_checkpoint_path)
        
        return model
    
    def _encode_video(
        self,
        frames,  # torch.Tensor
        fps: float,
        width: int,
        height: int,
        config: ProcessingConfig,
        log_callback: Callable[[str], None],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        """Encode frames to video or image sequence using FFmpeg.
        
        Args:
            frames: Output frames tensor [N, H, W, C]
            fps: Input frame rate
            width: Frame width
            height: Frame height
            config: Processing configuration
            log_callback: Log message callback
            progress_callback: Optional progress callback
        """
        # Calculate output fps
        interp_config = config.interpolation
        multiplier = interp_config.get("multi", 2)
        output_fps = fps * multiplier
        
        # Use CodecManager to build FFmpeg command
        from core.codec_manager import get_codec_manager, CodecConfig
        
        codec_manager = get_codec_manager()
        
        # Build CodecConfig from output config
        output_config = config.output
        codec_config = CodecConfig.from_dict(output_config)
        codec_manager.set_config(codec_config)
        
        assert self._output_path is not None
        
        # Check output mode
        if codec_manager.is_image_output():
            # Image sequence output
            self._save_image_sequence(
                frames=frames,
                output_path=self._output_path,
                image_format=codec_config.image_format,
                quality=codec_config.image_quality,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )
            return
        
        # Video output
        cmd = codec_manager.build_ffmpeg_encode_args(
            width=width,
            height=height,
            fps=output_fps,
            input_source="pipe:",
            output_path=self._output_path,
            include_audio=False,
        )
        
        log_callback(f"Running: {' '.join(cmd)}")
        
        # Start FFmpeg
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        assert proc.stdin is not None
        
        # Write frames
        for i, frame in enumerate(frames):
            if self._cancelled:
                proc.terminate()
                break
            
            # Convert to uint8 (move to CPU first if on CUDA)
            frame_uint8 = (frame.clamp(0, 1) * 255).byte().cpu().numpy()
            proc.stdin.write(frame_uint8.tobytes())
            
            if i % 100 == 0:
                if progress_callback:
                    progress_callback(i, len(frames), 0.0)
        
        proc.stdin.close()
        proc.wait()
        
        if proc.returncode != 0 and not self._cancelled:
            stderr = proc.stderr.read().decode() if proc.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg error: {stderr}")
    
    def _save_image_sequence(
        self,
        frames,  # torch.Tensor
        output_path: str,
        image_format: str,
        quality: int,
        log_callback: Callable[[str], None],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        """Save frames as image sequence.
        
        Args:
            frames: Output frames tensor [N, H, W, C]
            output_path: Output path (will be used as base for images)
            image_format: Image format (png, jpg, tiff, exr)
            quality: JPEG quality (1-100)
            log_callback: Log message callback
            progress_callback: Optional progress callback
        """
        from pathlib import Path
        from PIL import Image
        import numpy as np
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base filename without extension
        base_name = Path(output_path).stem
        
        # Determine extension
        ext_map = {"jpg": "jpg", "jpeg": "jpg", "png": "png", "tiff": "tiff", "exr": "exr"}
        ext = ext_map.get(image_format.lower(), "png")
        
        log_callback(f"Saving {len(frames)} frames as {ext.upper()} sequence...")
        
        total_frames = len(frames)
        for i, frame in enumerate(frames):
            if self._cancelled:
                break
            
            # Convert to uint8 numpy array
            frame_uint8 = (frame.clamp(0, 1) * 255).byte().cpu().numpy()
            
            # Create PIL Image
            img = Image.fromarray(frame_uint8)
            
            # Save frame
            frame_path = output_dir / f"{base_name}_{i:06d}.{ext}"
            
            if ext == "jpg":
                img.save(frame_path, quality=quality)
            else:
                img.save(frame_path)
            
            if i % 100 == 0:
                if progress_callback:
                    progress_callback(i, total_frames, 0.0)
        
        log_callback(f"Saved {len(frames)} frames to {output_dir}")
    
    def cancel(self, force: bool = False, timeout: float = 5.0) -> bool:
        """Cancel processing.
        
        Args:
            force: Whether to force terminate worker threads immediately
            timeout: Timeout for graceful shutdown before forcing termination
            
        Returns:
            True if cancellation succeeded
        """
        self._cancelled = True
        
        # Cancel thread pool if active
        if self._thread_pool:
            logger.info(f"Cancelling thread pool (force={force}, timeout={timeout})")
            return self._thread_pool.cancel(force=force, timeout=timeout)
        
        return True
    
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
        
        # Clear CUDA cache
        try:
            from ..torch_backend import clear_cache
            clear_cache()
        except Exception:
            pass
        
        gc.collect()
