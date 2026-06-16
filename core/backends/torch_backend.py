"""PyTorch backend for video frame interpolation.

This backend uses PyTorch for frame interpolation, supporting RIFE, FILM,
IFRNet, and AMT models.

Core constraint:
    Backend 不接触文件路径，只接收 numpy/tensor 数据。
    Backend 不自主决定 IO 时机，由 TaskScheduler 调度。
    Backend 不直接写文件，推理结果返回给 TaskScheduler。
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from core.types import (
    BackendType,
    BackendConfig,
    InferenceRequest,
    InferenceResult,
)
from .base_backend import BaseBackend


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
        
        # Multi-threading support (legacy, not actively used)
        self._thread_pool = None
        self._num_inference_threads = config.extra.get("inference_threads", 1)
        self._use_threading = self._num_inference_threads > 1
    
    def initialize(self) -> bool:
        """Initialize the PyTorch backend."""
        import time
        _debug_start = time.time()
        logger.debug(f"[DEBUG] TorchBackend.initialize() called")
        
        try:
            import torch
            logger.debug(f"[DEBUG] torch imported, elapsed={time.time()-_debug_start:.3f}s")
            
            # Check device
            device = self._config.get_device()
            logger.info(f"Initializing PyTorch backend on device: {device}")
            logger.debug(f"[DEBUG] device={device}, elapsed={time.time()-_debug_start:.3f}s")
            
            # Verify CUDA availability if using CUDA
            if device.startswith("cuda"):
                logger.debug(f"[DEBUG] checking CUDA availability, elapsed={time.time()-_debug_start:.3f}s")
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    self._config.device = "cpu"
                logger.debug(f"[DEBUG] CUDA check done, elapsed={time.time()-_debug_start:.3f}s")
            
            self._is_initialized = True
            logger.debug(f"[DEBUG] TorchBackend.initialize() complete, elapsed={time.time()-_debug_start:.3f}s")
            return True
            
        except ImportError as e:
            logger.error(f"PyTorch not available: {e}")
            return False

    def load_model(self, model_config: Dict[str, Any]) -> bool:
        """Load model weights (public interface for BaseBackend).

        Args:
            model_config: {
                "model_type": "rife",
                "model_version": "4.22",
                "scale": 1.0,
                "checkpoint_path": "/path/to/model.pth",  # optional
            }

        Returns:
            True if model loaded successfully
        """
        try:
            model_type_str = model_config.get("model_type", "rife").lower()
            model_version = model_config.get("model_version", "")

            if not model_version and model_type_str in self.SUPPORTED_MODELS:
                model_version = self.SUPPORTED_MODELS[model_type_str][0]

            logger.info(f"Loading {model_type_str} model: {model_version}")

            from ..pytorch_models import (
                VFIConfig,
                ModelType,
                get_model,
            )

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
                multiplier=model_config.get("multi", 2),
                scale=model_config.get("scale", 1.0),
                precision=self._config.precision,
            )

            self._model = get_model(model_type, vfi_config)

            # Load checkpoint
            checkpoint_path = model_config.get("checkpoint_path")
            if not checkpoint_path:
                models_dir = self._config.models_dir
                checkpoint_name = self._get_checkpoint_name(model_type_str, model_version)
                checkpoint_path = str(Path(models_dir) / model_type_str / checkpoint_name)

            self._model.load_model(checkpoint_path)

            # Apply torch.compile if requested
            if self._config.torch_compile:
                self._model.compile()

            # Save config for multi-threading
            self._current_model_type = model_type
            self._current_model_version = model_version
            self._current_checkpoint_path = checkpoint_path

            logger.info(f"Model loaded: {model_type_str}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

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

    # ====================
    # Pure inference interface (zero IO)
    # ====================

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Single frame-pair inference.

        Receives frame data only — no file paths, no cv2, no IO.

        Args:
            request: InferenceRequest with frame0, frame1, timestep, model_config

        Returns:
            InferenceResult with interpolated frame on CPU
        """
        if self._model is None:
            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error="Model not loaded. Call _load_model() first.",
            )

        start_time = time.perf_counter()
        try:
            device = self._config.get_torch_device()

            with torch.inference_mode():
                frame0 = request.frame0.to(device)
                frame1 = request.frame1.to(device)

                if self._config.precision in ("fp16", "bf16") and device.type == "cuda":
                    dtype = torch.float16 if self._config.precision == "fp16" else torch.bfloat16
                    frame0 = frame0.to(dtype)
                    frame1 = frame1.to(dtype)

                output = self._model.interpolate(
                    frame0,
                    frame1,
                    timestep=request.timestep,
                    **request.model_config,
                )

                # Handle VFIResult wrapper
                from ..pytorch_models import VFIResult
                if isinstance(output, VFIResult):
                    output = output.frame

                # Ensure output is float32 on CPU
                output = output.float().cpu()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return InferenceResult(
                output_frame=output,
                success=True,
                inference_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=str(e),
            )

    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch inference: merge multiple frame pairs into one batch.

        GPU excels at large batch data; batch inference is 2-3x faster
        than per-frame inference.

        Args:
            requests: List of InferenceRequest objects

        Returns:
            List of InferenceResult, one per request

        Raises:
            NotImplementedError: If model doesn't support batch inference
        """
        if self._model is None:
            return [InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error="Model not loaded",
            ) for _ in requests]

        # Check if model supports true batch inference
        if not hasattr(self._model, "interpolate_batch"):
            raise NotImplementedError(
                f"{type(self._model).__name__} does not support batch inference"
            )

        start_time = time.perf_counter()
        try:
            device = self._config.get_torch_device()

            # Stack all frames into batch tensors
            batch_frame0 = torch.stack([r.frame0.to(device) for r in requests])
            batch_frame1 = torch.stack([r.frame1.to(device) for r in requests])
            batch_timestep = torch.tensor(
                [r.timestep for r in requests], device=device
            )

            if self._config.precision in ("fp16", "bf16") and device.type == "cuda":
                dtype = torch.float16 if self._config.precision == "fp16" else torch.bfloat16
                batch_frame0 = batch_frame0.to(dtype)
                batch_frame1 = batch_frame1.to(dtype)

            with torch.inference_mode():
                batch_output = self._model.interpolate_batch(
                    batch_frame0, batch_frame1, batch_timestep
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            per_frame_ms = elapsed_ms / len(requests) if requests else 0.0

            results = []
            for i in range(len(requests)):
                output = batch_output[i].float().cpu()
                results.append(InferenceResult(
                    output_frame=output,
                    success=True,
                    inference_time_ms=per_frame_ms,
                ))
            return results

        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            return [InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=str(e),
            ) for _ in requests]

    def cancel(self, force: bool = False, timeout: float = 5.0) -> None:
        """Cancel processing.
        
        Args:
            force: Whether to force terminate worker threads immediately
            timeout: Timeout for graceful shutdown before forcing termination
        """
        self._cancelled = True
        
        # Cancel thread pool if active (legacy)
        if self._thread_pool:
            logger.info(f"Cancelling thread pool (force={force}, timeout={timeout})")
            self._thread_pool.cancel(force=force, timeout=timeout)

    def unload_model(self) -> None:
        """Unload model, release GPU memory.

        Safe to call multiple times. After calling this, infer() will return
        an error until _load_model() is called again.
        """
        if self._model is not None:
            try:
                self._model.unload()
            except Exception:
                pass
            self._model = None

            # Clear CUDA cache if on CUDA
            device_str = self._config.get_device()
            if device_str.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

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
            from ..pytorch_models import clear_cache
            clear_cache()
        except Exception:
            pass
        
        gc.collect()
