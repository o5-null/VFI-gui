"""Abstract base class and utilities for VFI models.

This module defines the core abstractions for video frame interpolation models,
providing a consistent interface for different model implementations.
"""

import gc
import torch
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class DType(Enum):
    """Supported data types for inference."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


DTYPE_MAP = {
    DType.FLOAT32: torch.float32,
    DType.FLOAT16: torch.float16,
    DType.BFLOAT16: torch.bfloat16,
}


@dataclass
class VFIModelInfo:
    """Information about a VFI model."""
    name: str
    model_type: str  # e.g., "rife", "amt", "film"
    checkpoint_name: str
    checkpoint_path: Optional[str] = None
    supported_multipliers: Tuple[int, ...] = (2,)
    supports_timestep: bool = True
    min_frames: int = 2
    default_dtype: DType = DType.FLOAT32
    
    @property
    def display_name(self) -> str:
        """Get a display-friendly name."""
        return f"{self.name} ({self.model_type.upper()})"


@dataclass
class InferencePerfEntry:
    """Per-inference timing entry collected when perf_stats is enabled."""
    pair_index: int = 0          # Frame pair index (0-based)
    timestep: float = 0.5        # Interpolation timestep
    inference_ms: float = 0.0    # Pure model.interpolate() time in ms
    total_ms: float = 0.0        # Total including pre/post overhead in ms


@dataclass
class PerfStats:
    """Performance statistics collected during frame processing.
    
    Only populated when FrameProcessor is created with enable_perf_stats=True.
    """
    # Per-inference timing entries
    entries: List[InferencePerfEntry] = field(default_factory=list)
    
    # Summary (computed after processing)
    total_inference_ms: float = 0.0     # Sum of all inference_ms
    avg_inference_ms: float = 0.0       # Average inference_ms
    min_inference_ms: float = 0.0
    max_inference_ms: float = 0.0
    std_inference_ms: float = 0.0
    fps: float = 0.0                    # Based on avg_inference_ms
    
    total_wall_ms: float = 0.0          # Total wall-clock time including overhead
    
    @property
    def num_inferences(self) -> int:
        return len(self.entries)
    
    def compute_summary(self) -> None:
        """Compute summary statistics from entries."""
        if not self.entries:
            return
        
        times = [e.inference_ms for e in self.entries]
        self.total_inference_ms = sum(times)
        self.avg_inference_ms = self.total_inference_ms / len(times)
        self.min_inference_ms = min(times)
        self.max_inference_ms = max(times)
        
        if len(times) > 1:
            variance = sum((t - self.avg_inference_ms) ** 2 for t in times) / (len(times) - 1)
            self.std_inference_ms = variance ** 0.5
        
        self.fps = 1000.0 / self.avg_inference_ms if self.avg_inference_ms > 0 else 0.0


@dataclass
class InterpolationResult:
    """Result of a frame interpolation operation."""
    frames: torch.Tensor  # Output frames [N, C, H, W]
    original_count: int
    output_count: int
    processing_time: float = 0.0
    perf_stats: Optional[PerfStats] = None
    
    @property
    def multiplier_achieved(self) -> float:
        """Calculate the actual multiplier achieved."""
        return self.output_count / self.original_count if self.original_count > 0 else 0.0


@dataclass
class InterpolationConfig:
    """Configuration for interpolation process."""
    multiplier: int = 2
    clear_cache_after_n_frames: int = 10
    dtype: DType = DType.FLOAT32
    batch_size: int = 1
    scale_factor: float = 1.0
    fast_mode: bool = False
    ensemble: bool = False
    torch_compile: bool = False
    skip_frames: Optional[List[int]] = None


class InterpolationStateList:
    """Manages which frames should be skipped during interpolation.
    
    Inspired by ComfyUI-Frame-Interpolation's InterpolationStateList.
    """
    
    def __init__(self, frame_indices: List[int], is_skip_list: bool = True):
        """
        Args:
            frame_indices: List of frame indices
            is_skip_list: If True, listed frames are skipped.
                         If False, only listed frames are processed.
        """
        self.frame_indices = set(frame_indices)
        self.is_skip_list = is_skip_list
    
    def is_frame_skipped(self, frame_index: int) -> bool:
        """Check if a frame should be skipped."""
        is_in_list = frame_index in self.frame_indices
        return self.is_skip_list and is_in_list or not self.is_skip_list and not is_in_list
    
    @classmethod
    def from_string(cls, frame_str: str, is_skip_list: bool = True) -> "InterpolationStateList":
        """Create from comma-separated string like '1,2,3'."""
        indices = [int(x.strip()) for x in frame_str.split(",") if x.strip()]
        return cls(indices, is_skip_list)


def get_torch_device() -> torch.device:
    """Get the best available torch device.
    
    Delegates to vfi_torch.utils.get_device() for unified implementation.
    """
    from .vfi_torch.utils import get_device
    return get_device()


# Note: preprocess_frames and postprocess_frames are now defined in vfi_torch/utils.py
# These are imported and re-exported in __init__.py for backward compatibility.


def clear_cuda_cache():
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    gc.collect()


def assert_batch_size(frames: torch.Tensor, min_frames: int = 2, model_name: Optional[str] = None):
    """Assert that we have enough frames for interpolation."""
    subject_verb = "Most VFI models require" if model_name is None else f"VFI model {model_name} requires"
    assert len(frames) >= min_frames, (
        f"{subject_verb} at least {min_frames} frames to work with, "
        f"only found {frames.shape[0]}."
    )


class VFIModelBase(ABC):
    """Abstract base class for video frame interpolation models.
    
    This class defines the interface that all VFI models must implement.
    It provides common functionality for model loading, inference, and memory management.
    
    Inspired by ComfyUI-Frame-Interpolation's model architecture.
    """
    
    # Model metadata - override in subclasses
    MODEL_TYPE: str = "base"
    MODEL_INFO: Optional[VFIModelInfo] = None
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: DType = DType.FLOAT32,
    ):
        """
        Args:
            device: Target device for inference. Auto-detected if None.
            dtype: Data type for inference.
        """
        self.device = device or get_torch_device()
        self.dtype = dtype
        self.torch_dtype = DTYPE_MAP[dtype]
        self._model: Optional[torch.nn.Module] = None
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded and self._model is not None
    
    @abstractmethod
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """Load the model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint.
            **kwargs: Additional model-specific arguments.
        """
        pass
    
    @abstractmethod
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float,
        **kwargs,
    ) -> torch.Tensor:
        """Interpolate a single middle frame between two frames.
        
        Args:
            frame0: First frame [1, C, H, W]
            frame1: Second frame [1, C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)
            **kwargs: Additional model-specific arguments.
            
        Returns:
            Interpolated frame [1, C, H, W]
        """
        pass
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint weights into the model.
        
        Override this method for custom checkpoint loading logic.
        """
        model = self._check_model_loaded()
        
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        
        model.load_state_dict(state_dict, strict=True)
    
    def to_device(self) -> None:
        """Move model to target device and convert to target dtype.
        
        Handles device placement and precision conversion (FP16/BF16)
        in a single call. Subclasses should call this instead of
        manually calling .to(device) and .half().
        """
        if self._model is not None:
            self._model = self._model.to(self.device)
            if self.dtype != DType.FLOAT32:
                self._model = self._model.to(self.torch_dtype)
    
    def prepare_frames(self, *frames: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Move frames to target device and convert to target dtype.
        
        Handles device placement and precision conversion (FP16/BF16)
        in a single call. Subclasses should call this instead of
        manually calling .to(device) and .half() on input frames.
        
        Args:
            *frames: Input frame tensors to prepare
            
        Returns:
            Tuple of prepared frame tensors on the target device and dtype
        """
        result = []
        for frame in frames:
            frame = frame.to(self.device)
            if self.dtype != DType.FLOAT32:
                frame = frame.to(self.torch_dtype)
            result.append(frame)
        return tuple(result)
    
    def eval(self) -> None:
        """Set model to evaluation mode."""
        if self._model is not None:
            self._model.eval()
    
    def compile(self) -> None:
        """Compile model with torch.compile for faster inference."""
        if self._model is not None and hasattr(torch, "compile"):
            self._model = torch.compile(self._model)  # type: ignore[assignment]
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            clear_cuda_cache()
    
    @classmethod
    def get_available_checkpoints(cls) -> List[str]:
        """Get list of available checkpoint names for this model type.
        
        Override in subclasses to return model-specific checkpoints.
        """
        return []
    
    @classmethod
    def get_default_checkpoint(cls) -> Optional[str]:
        """Get the default checkpoint for this model type."""
        checkpoints = cls.get_available_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def _check_model_loaded(self) -> torch.nn.Module:
        """Check if model is loaded and return it."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")
        return self._model
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(device={self.device}, dtype={self.dtype.value}, status={status})"
