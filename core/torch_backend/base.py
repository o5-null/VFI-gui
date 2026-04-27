"""Abstract base class and utilities for VFI models.

This module defines the core abstractions for video frame interpolation models,
providing a consistent interface for different model implementations.
"""

import gc
import torch
import typing
import einops
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
class InterpolationResult:
    """Result of a frame interpolation operation."""
    frames: torch.Tensor  # Output frames [N, C, H, W]
    original_count: int
    output_count: int
    processing_time: float = 0.0
    
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
    """Get the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Preprocess frames from NHWC to NCHW format.
    
    Args:
        frames: Input frames in NHWC format [N, H, W, C]
        
    Returns:
        Frames in NCHW format [N, C, H, W]
    """
    # Handle RGBA by taking only RGB channels
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")


def postprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Postprocess frames from NCHW to NHWC format.
    
    Args:
        frames: Input frames in NCHW format [N, C, H, W]
        
    Returns:
        Frames in NHWC format [N, H, W, C]
    """
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def assert_batch_size(frames: torch.Tensor, min_frames: int = 2, model_name: Optional[str] = None):
    """Assert that we have enough frames for interpolation."""
    subject_verb = "Most VFI models require" if model_name is None else f"VFI model {model_name} requires"
    assert len(frames) >= min_frames, (
        f"{subject_verb} at least {min_frames} frames to work with, "
        f"only found {frames.shape[0]}."
    )


class VFIBaseModel(ABC):
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
        """Move model to target device."""
        if self._model is not None:
            self._model = self._model.to(self.device)
            if self.dtype != DType.FLOAT32:
                self._model = self._model.to(self.torch_dtype)
    
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
