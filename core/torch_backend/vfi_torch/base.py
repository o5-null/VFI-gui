"""
Base classes and interfaces for VFI models.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable


class ModelType(Enum):
    """Supported interpolation model types."""
    RIFE = "rife"
    FILM = "film"
    IFRNET = "ifrnet"  # Changed from IFRNet to IFRNET for consistency
    AMT = "amt"
    GMFSS = "gmfss"
    STMFNET = "stmfnet"
    FLAVR = "flavr"
    CAIN = "cain"


class BackendType(Enum):
    """Inference backend types."""
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    NCNN = "ncnn"


@dataclass
class VFIConfig:
    """Configuration for VFI model inference."""
    # Model settings
    model_type: ModelType = ModelType.RIFE
    model_version: str = "4.22"  # Model specific version
    checkpoint_path: Optional[str] = None
    
    # Inference settings
    multiplier: int = 2  # Frame multiplication factor
    scale: float = 1.0  # Resolution scale factor
    use_scene_detection: bool = True
    
    # Backend settings
    backend: BackendType = BackendType.PYTORCH
    device: str = "auto"  # Device string: "auto", "cuda:0", "rocm:0", "xpu:0", "cpu"
    device_id: int = 0  # DEPRECATED: Use device string instead
    fp16: bool = True
    num_streams: int = 1
    
    # Performance settings
    batch_size: int = 1
    clear_cache_every: int = 10  # Clear cache every N frames
    tile_size: int = 0  # 0 = no tiling
    tile_overlap: int = 8
    
    # Model-specific settings
    fast_mode: bool = False  # RIFE fast mode
    ensemble: bool = False  # RIFE ensemble


@dataclass
class VFIResult:
    """Result from VFI inference."""
    frame: torch.Tensor  # Interpolated frame [C, H, W] or [B, C, H, W]
    model_type: ModelType  # Model type used
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class BaseVFIModel(ABC):
    """
    Abstract base class for Video Frame Interpolation models.
    
    All VFI models should inherit from this class and implement
    the required abstract methods.
    """
    
    # Model metadata (override in subclass)
    MODEL_NAME: str = "base"
    SUPPORTED_VERSIONS: List[str] = []
    DEFAULT_VERSION: str = ""
    MIN_INPUT_FRAMES: int = 2
    
    def __init__(self, config: Optional[VFIConfig] = None):
        """
        Initialize the VFI model.
        
        Args:
            config: Model configuration
        """
        if config is None:
            config = VFIConfig()
        self._config = config
        self._model: Optional[torch.nn.Module] = None
    
    @property
    def device(self) -> torch.device:
        """Get the torch device based on configuration.
        
        Uses DeviceManager for device resolution.
        """
        from core.device_manager import get_torch_device
        return get_torch_device(self._config.device)
    
    @abstractmethod
    def load_model(self, checkpoint_path: str) -> None:
        """
        Load the model weights.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        pass
    
    @abstractmethod
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs
    ) -> VFIResult:
        """
        Interpolate a single frame between two input frames.
        
        Args:
            frame0: First input frame [C, H, W] or [B, C, H, W]
            frame1: Second input frame [C, H, W] or [B, C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)
            **kwargs: Additional model-specific arguments
            
        Returns:
            VFIResult with interpolated frame
        """
        pass
    
    def interpolate_batch(
        self,
        frames: torch.Tensor,
        multiplier: int = 2,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """
        Interpolate multiple frames in a video sequence.
        
        Args:
            frames: Input frames [N, C, H, W]
            multiplier: Frame multiplication factor
            callback: Progress callback function (current, total)
            
        Returns:
            Interpolated frames [N * multiplier, C, H, W]
        """
        N = frames.shape[0]
        output_frames = []
        
        for i in range(N - 1):
            # Add original frame
            output_frames.append(frames[i])
            
            # Generate interpolated frames
            for j in range(1, multiplier):
                timestep = j / multiplier
                result = self.interpolate(
                    frames[i],
                    frames[i + 1],
                    timestep=timestep
                )
                output_frames.append(result.frame)
                
                if callback:
                    callback(i * (multiplier - 1) + j, (N - 1) * (multiplier - 1))
        
        # Add last original frame
        output_frames.append(frames[-1])
        
        return torch.stack(output_frames)
    
    def __call__(
        self,
        frames: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Callable interface for model inference.
        
        Args:
            frames: Input frames
            **kwargs: Additional arguments
            
        Returns:
            Interpolated frames
        """
        multiplier = kwargs.get("multiplier", self._config.multiplier)
        callback = kwargs.get("callback", None)
        return self.interpolate_batch(frames, multiplier, callback)
    
    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        return self._model is not None
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()


def get_model(config: VFIConfig) -> BaseVFIModel:
    """
    Factory function to create VFI model instance.
    
    Args:
        config: Model configuration
        
    Returns:
        VFI model instance
    """
    from .rife import RIFEModel
    from .film import FILMModel
    from .ifrnet import IFRNetModel
    from .amt import AMTModel
    
    model_classes = {
        ModelType.RIFE: RIFEModel,
        ModelType.FILM: FILMModel,
        ModelType.IFRNET: IFRNetModel,
        ModelType.AMT: AMTModel,
    }
    
    if config.model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    return model_classes[config.model_type](config)
