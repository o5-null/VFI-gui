"""
Base classes and interfaces for VFI models.
"""

import torch
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable

# Import from parent module for unified inheritance
from ..base import VFIModelBase, DType


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
    XVFI = "xvfi"
    EISAI = "eisai"


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
    precision: str = "fp16"  # Precision: "fp32", "fp16", "bf16"
    fp16: bool = True  # DEPRECATED: use precision instead. Kept for backward compat.
    num_streams: int = 1
    
    # Performance settings
    batch_size: int = 1
    clear_cache_every: int = 10  # Clear cache every N frames
    tile_size: int = 0  # 0 = no tiling
    tile_overlap: int = 8
    
    # Model-specific settings
    fast_mode: bool = False  # RIFE fast mode
    ensemble: bool = False  # RIFE ensemble

    def __post_init__(self):
        """Synchronize precision and fp16 for backward compatibility."""
        if self.fp16 and self.precision == "fp32":
            # fp16=True explicitly set but precision was fp32 - fp16 wins for compat
            self.precision = "fp16"
        elif not self.fp16 and self.precision in ("fp16", "bf16"):
            # fp16=False explicitly set, override precision
            self.precision = "fp32"
        # Sync fp16 from precision as the canonical source
        self.fp16 = self.precision in ("fp16", "bf16")

    def get_dtype(self) -> "DType":
        """Convert precision string to DType enum.

        Returns:
            DType enum value corresponding to the precision setting
        """
        from ..base import DType
        mapping = {
            "fp32": DType.FLOAT32,
            "fp16": DType.FLOAT16,
            "bf16": DType.BFLOAT16,
        }
        return mapping.get(self.precision, DType.FLOAT16)
    
    def to_interpolation_config(self) -> "InterpolationConfig":
        """Convert to InterpolationConfig for FrameProcessor.
        
        Returns:
            InterpolationConfig instance with compatible settings.
        """
        from ..base import InterpolationConfig
        return InterpolationConfig(
            multiplier=self.multiplier,
            dtype=self.get_dtype(),
            batch_size=self.batch_size,
            scale_factor=self.scale,
            fast_mode=self.fast_mode,
            ensemble=self.ensemble,
            clear_cache_after_n_frames=self.clear_cache_every,
        )


@dataclass
class VFIResult:
    """Result from VFI inference."""
    frame: torch.Tensor  # Interpolated frame [C, H, W] or [B, C, H, W]
    model_type: ModelType  # Model type used
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class PyTorchVFIModel(VFIModelBase):
    """
    Extended base class for Video Frame Interpolation models.
    
    Inherits from VFIBaseModel and adds batch processing capabilities.
    All VFI models should inherit from this class.
    
    Supports two initialization patterns:
    1. config-based: BaseVFIModel(config=VFIConfig(...))
    2. device/dtype-based: BaseVFIModel(device=torch.device("cuda"), dtype=DType.FLOAT16)
    """
    
    # Model metadata (override in subclass)
    MODEL_NAME: str = "base"
    SUPPORTED_VERSIONS: List[str] = []
    DEFAULT_VERSION: str = ""
    MIN_INPUT_FRAMES: int = 2
    
    def __init__(
        self,
        config: Optional["VFIConfig"] = None,
        device: Optional[torch.device] = None,
        dtype: DType = DType.FLOAT32,
    ):
        """
        Initialize the VFI model.
        
        Supports two initialization patterns:
        1. config-based: RIFEModel(config=VFIConfig(model_type=ModelType.RIFE))
        2. device/dtype-based: RIFEModel(device=torch.device("cuda"), dtype=DType.FLOAT16)
        
        Args:
            config: Model configuration (if provided, device/dtype are derived from it)
            device: Target device (used if config is None)
            dtype: Data type (used if config is None)
        """
        # Derive device and dtype from config if provided
        if config is not None:
            from .utils import get_device
            if config.device == "auto":
                device = get_device()
            else:
                device = torch.device(config.device)
            dtype = config.get_dtype()
        
        # Initialize parent class
        super().__init__(device=device, dtype=dtype)
        
        self._config = config or VFIConfig()
    
    @abstractmethod
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """
        Load the model weights.
        
        Args:
            checkpoint_path: Path to model checkpoint
            **kwargs: Additional model-specific arguments
        """
        pass
    
    @abstractmethod
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs
    ) -> torch.Tensor:
        """
        Interpolate a single frame between two input frames.
        
        Args:
            frame0: First input frame [C, H, W] or [B, C, H, W]
            frame1: Second input frame [C, H, W] or [B, C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Interpolated frame tensor [C, H, W] or [B, C, H, W]
        """
        pass
    
    def interpolate_with_result(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs
    ) -> "VFIResult":
        """
        Interpolate a frame and return VFIResult with metadata.
        
        This is a convenience method that wraps interpolate() 
        and returns a VFIResult object.
        
        Args:
            frame0: First input frame
            frame1: Second input frame
            timestep: Interpolation timestep
            **kwargs: Additional arguments
            
        Returns:
            VFIResult with interpolated frame and metadata
        """
        frame = self.interpolate(frame0, frame1, timestep, **kwargs)
        return VFIResult(frame=frame, model_type=self._config.model_type)
    
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
                frame = self.interpolate(
                    frames[i],
                    frames[i + 1],
                    timestep=timestep
                )
                output_frames.append(frame)
                
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


# Note: is_loaded property and unload() method are inherited from VFIBaseModel


def get_model(config: VFIConfig) -> PyTorchVFIModel:
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
    from .xvfi import XVFIModel
    from .eisai import EISAIModel
    from .gmfss import GMFSSModel
    
    model_classes = {
        ModelType.RIFE: RIFEModel,
        ModelType.FILM: FILMModel,
        ModelType.IFRNET: IFRNetModel,
        ModelType.AMT: AMTModel,
        ModelType.XVFI: XVFIModel,
        ModelType.EISAI: EISAIModel,
        ModelType.GMFSS: GMFSSModel,
    }
    
    if config.model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    return model_classes[config.model_type](config)
