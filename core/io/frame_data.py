"""Frame data structures for video processing.

This module defines data classes for carrying video frame data
between components without direct IO operations.
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, Callable
from pathlib import Path
from enum import Enum

import numpy as np
import torch


class FrameFormat(Enum):
    """Supported frame formats."""
    RGB = "rgb"           # [H, W, 3] uint8
    RGB_FLOAT = "rgb_f"   # [H, W, 3] float32 [0, 1]
    BGR = "bgr"           # [H, W, 3] uint8 (OpenCV default)
    RGBA = "rgba"         # [H, W, 4] uint8
    GRAY = "gray"         # [H, W] uint8
    GRAY_FLOAT = "gray_f" # [H, W] float32 [0, 1]
    TENSOR_NHWC = "nhwc"  # [N, H, W, C] float32
    TENSOR_NCHW = "nchw"  # [N, C, H, W] float32


@dataclass
class VideoMetadata:
    """Video file metadata."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float = 0.0
    codec: str = ""
    pixel_format: str = ""
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)."""
        return (self.width, self.height)


@dataclass
class FrameData:
    """Container for frame data.
    
    Carries frame data between components without direct IO.
    Supports both numpy arrays and PyTorch tensors.
    
    Attributes:
        data: Frame data (numpy array or torch tensor)
        frame_idx: Frame index in sequence
        format: Frame format
        metadata: Optional metadata dictionary
    """
    data: np.ndarray | torch.Tensor
    frame_idx: int
    format: FrameFormat = FrameFormat.RGB_FLOAT
    metadata: dict = field(default_factory=dict)
    
    def to_numpy(self) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data
    
    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert data to torch tensor."""
        if isinstance(self.data, torch.Tensor):
            return self.data.to(device)
        return torch.from_numpy(self.data).to(device)
    
    def to_format(self, target_format: FrameFormat) -> "FrameData":
        """Convert frame to target format."""
        # This would contain format conversion logic
        # For now, return as-is
        return FrameData(
            data=self.data,
            frame_idx=self.frame_idx,
            format=target_format,
            metadata=self.metadata.copy(),
        )


@dataclass
class FrameBatch:
    """Batch of frames for processing.
    
    Attributes:
        frames: List of FrameData
        batch_idx: Batch index
        metadata: Batch metadata
    """
    frames: List[FrameData]
    batch_idx: int = 0
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert batch to tensor [N, H, W, C]."""
        tensors = [f.to_tensor(device) for f in self.frames]
        return torch.stack(tensors, dim=0)
    
    def to_numpy(self) -> np.ndarray:
        """Convert batch to numpy array [N, H, W, C]."""
        arrays = [f.to_numpy() for f in self.frames]
        return np.stack(arrays, axis=0)


@dataclass
class ProcessedFrameData:
    """Frame data after processing.
    
    Carries processed frame data and processing metadata.
    
    Attributes:
        data: Processed frame data
        source_frame_idx: Original frame index
        interpolated: Whether this is an interpolated frame
        interpolation_ratio: Interpolation position (0-1)
        metadata: Processing metadata
    """
    data: np.ndarray | torch.Tensor
    source_frame_idx: int
    interpolated: bool = False
    interpolation_ratio: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data
    
    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert to torch tensor."""
        if isinstance(self.data, torch.Tensor):
            return self.data.to(device)
        return torch.from_numpy(self.data).to(device)


@dataclass
class VideoFrameSequence:
    """Complete frame sequence from video.
    
    Attributes:
        frames: List of FrameData
        metadata: Video metadata
        source_path: Original video path
    """
    frames: List[FrameData]
    metadata: VideoMetadata
    source_path: Optional[Path] = None
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __iter__(self) -> Iterator[FrameData]:
        return iter(self.frames)
    
    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert all frames to tensor [N, H, W, C]."""
        tensors = [f.to_tensor(device) for f in self.frames]
        return torch.stack(tensors, dim=0)
    
    def to_numpy(self) -> np.ndarray:
        """Convert all frames to numpy array [N, H, W, C]."""
        arrays = [f.to_numpy() for f in self.frames]
        return np.stack(arrays, axis=0)
    
    def get_batch(self, start_idx: int, batch_size: int) -> FrameBatch:
        """Get a batch of frames."""
        end_idx = min(start_idx + batch_size, len(self.frames))
        batch_frames = self.frames[start_idx:end_idx]
        return FrameBatch(
            frames=batch_frames,
            batch_idx=start_idx // batch_size,
            metadata={"start_idx": start_idx, "end_idx": end_idx},
        )
