"""Core type definitions for VFI-gui.

This module provides a single source of truth for all shared data structures
used across the architecture. All components import from this module to obtain
unified type definitions, avoiding scattered and duplicate type definitions.

Design rationale:
    1. Cross-component imports only need one location
    2. Avoids circular dependencies (SubTask references FrameRef,
       FrameRef references enums)
    3. Type changes propagate automatically to all consumers

Dependencies:
    Only standard library + numpy + torch are imported.
    No business modules from core/ are imported to avoid circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


# ====================
# 1. Enumerations
# ====================


class BackendType(Enum):
    """Inference backend type."""
    VAPOURSYNTH = "vapoursynth"
    TORCH = "torch"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    NCNN = "ncnn"
    DIRECTML = "directml"


class TaskState(Enum):
    """Task lifecycle state."""
    PENDING = "pending"
    LOADING = "loading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubTaskState(Enum):
    """Sub-task scheduling state."""
    PENDING = "pending"
    WAITING_IO = "waiting_io"
    WAITING_GPU = "waiting_gpu"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FramePairAction(Enum):
    """Pre-processing decision for a frame pair."""
    INTERPOLATE = "interpolate"      # Needs interpolation -> generate SubTask
    SCENE_CUT = "scene_cut"          # Scene change -> write frame0 only, no interp
    DUPLICATE = "duplicate"          # Duplicate frame -> write frame0 only, no interp
    LAST_FRAME = "last_frame"        # Last frame -> write directly


class IORequestType(Enum):
    """IO request type."""
    LOAD_FRAMES = "load_frames"
    WRITE_FRAMES = "write_frames"
    RELEASE_FRAMES = "release_frames"
    LOAD_MODEL = "load_model"
    RELEASE_MODEL = "release_model"


class FrameFormat(Enum):
    """Frame data format."""
    RGB = "rgb"              # [H, W, 3] uint8
    RGB_FLOAT = "rgb_f"      # [H, W, 3] float32 [0, 1]
    TENSOR_NHWC = "nhwc"     # [N, H, W, C] float32
    TENSOR_NCHW = "nchw"     # [N, C, H, W] float32


class DecoderType(Enum):
    """Video decoder type."""
    PYAV = "pyav"               # PyAV software decode, full format compat
    PYAV_HW = "pyav_hw"         # PyAV hardware accel (NVDEC/QSV)
    VAPOURSYNTH = "vapoursynth" # VapourSynth source filter


class InferenceStrategy(Enum):
    """Inference parallelism strategy."""
    SERIAL = "serial"              # Sequential
    MULTI_MODEL = "multi_model"    # Multiple model replicas
    CUDA_STREAMS = "cuda_streams"  # Single model, multiple CUDA streams
    BATCH = "batch"                # Batch inference


class SceneDetectionMethod(Enum):
    """Scene detection method."""
    PLANESTATS = "planestats"
    NEURAL = "neural"
    VAPOURSYNTH = "vapoursynth"


class EngineStatus(Enum):
    """Engine lifecycle state."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"


# ====================
# 2. Configuration Data Structures
# ====================


@dataclass
class BackendConfig:
    """Backend configuration.

    Single source of truth for backend config. Delegates device resolution
    to DeviceManager (authoritative source for GPU detection).

    Attributes:
        backend_type: Type of inference backend
        models_dir: Directory containing model files
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        num_threads: Number of processing threads
        device: Device string - "auto" | "cuda:0" | "xpu:0" | "cpu"
        precision: Inference precision - "fp32" | "fp16" | "bf16"
        fp16: Deprecated - use precision instead. Kept for backward compatibility.
        torch_compile: Whether to enable torch.compile (PyTorch only)
        extra: Additional backend-specific options
    """
    backend_type: BackendType = BackendType.VAPOURSYNTH
    models_dir: str = "models"
    temp_dir: str = "temp"
    output_dir: str = "output"
    num_threads: int = 4
    device: str = "auto"         # "auto" | "cuda:0" | "xpu:0" | "cpu"
    precision: str = "fp16"      # "fp32" | "fp16" | "bf16"
    fp16: bool = True            # DEPRECATED: use precision instead
    torch_compile: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Synchronize precision and fp16 for backward compatibility."""
        if self.precision == "fp16" and not self.fp16:
            self.precision = "fp32"
        elif self.precision == "fp32" and self.fp16:
            self.fp16 = False
        elif self.fp16 and self.precision not in ("fp16", "bf16"):
            self.precision = "fp16"
        self.fp16 = self.precision in ("fp16", "bf16")

    def get_device(self) -> str:
        """Resolve device string to actual device.

        Delegates to DeviceManager for device resolution.

        Returns:
            Device string (e.g., "cuda:0", "xpu:0", "cpu")
        """
        from core.device_manager import resolve_device
        return resolve_device(self.device)

    def get_torch_device(self) -> torch.device:
        """Get torch.device object for the configured device.

        Delegates to DeviceManager for torch device resolution.

        Returns:
            torch.device instance
        """
        from core.device_manager import get_torch_device
        return get_torch_device(self.device)


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration.

    Attributes:
        interpolation: Interpolation settings dict
        upscaling: Upscaling settings dict
        scene_detection: Scene detection settings dict
        output: Output encoding settings dict
    """
    interpolation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "model_type": "rife",
        "model_version": "",
        "multi": 2,
        "scale": 1.0,
        "scene_change": False,
    })
    upscaling: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "engine": "",
        "num_streams": 3,
        "tile_size": 0,
        "overlap": 0,
    })
    scene_detection: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "model": 12,
        "threshold": 0.5,
        "fp16": True,
    })
    output: Dict[str, Any] = field(default_factory=lambda: {
        "codec": "hevc_nvenc",
        "quality": 22,
        "preset": "p4",
        "audio_copy": True,
        "output_dir": "",
        "output_subdir": "",
        "output_filename": "",
    })


@dataclass
class ColorSpaceInfo:
    """Color space metadata.

    Attributes:
        matrix: Color matrix - "bt601" / "bt709" / "bt2020"
        transfer: Transfer characteristic - "sdr" / "pq" / "hlg" / "bt470bg"
        primaries: Color primaries - "bt601" / "bt709" / "bt2020"
        range: Pixel range - "limited" / "full"
        bit_depth: Bit depth - 8 / 10 / 12
        chroma_location: Chroma sample location - "left" / "center" / "topleft"
    """
    matrix: str = "bt709"
    transfer: str = "sdr"
    primaries: str = "bt709"
    range: str = "limited"
    bit_depth: int = 8
    chroma_location: str = "left"


@dataclass
class AudioConfig:
    """Audio processing configuration.

    Attributes:
        mode: Audio mode - "copy" / "stretch" / "reencode" / "none"
        codec: Audio codec for reencode
        bitrate: Audio bitrate
        stretch_factor: Time-stretch factor for slow-motion
    """
    mode: str = "copy"
    codec: str = "aac"
    bitrate: str = "192k"
    stretch_factor: float = 1.0


@dataclass
class ParallelConfig:
    """Parallel processing configuration.

    Attributes:
        num_inference_workers: Number of concurrent inference workers
        prefetch_subtasks: Number of subtasks to prefetch
        result_buffer_size: Buffer size for result queue
    """
    num_inference_workers: int = 2
    prefetch_subtasks: int = 4
    result_buffer_size: int = 64


# ====================
# 3. IO Data Structures
# ====================


@dataclass
class VideoMetadata:
    """Video file metadata.

    Attributes:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second (average)
        total_frames: Total number of frames
        duration: Video duration in seconds
        codec: Codec name string
        pixel_format: Pixel format string
        is_vfr: Whether video uses variable framerate
        color_space: Color space information
        has_audio: Whether video contains audio tracks
        audio_codec: Audio codec name
        audio_sample_rate: Audio sample rate in Hz
    """
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float = 0.0
    codec: str = ""
    pixel_format: str = ""
    is_vfr: bool = False
    color_space: Optional[ColorSpaceInfo] = None
    has_audio: bool = False
    audio_codec: str = ""
    audio_sample_rate: int = 0


@dataclass
class FrameTimestamps:
    """Frame timestamp information for VFR-aware interpolation.

    Attributes:
        pts: Per-frame presentation timestamps
        is_vfr: Whether timestamps are variable framerate
        avg_fps: Average framerate
        timebase: Timebase denominator for PTS values
    """
    pts: List[float]
    is_vfr: bool
    avg_fps: float
    timebase: float

    def get_timestep(self, frame_i: int, frame_j: int, interp_pos: float) -> float:
        """Compute VFR-aware interpolation timestep.

        For constant framerate, timestep equals interp_pos directly.
        For variable framerate, timestep is scaled by the actual duration
        between frames relative to the average frame duration.

        Args:
            frame_i: Index of the earlier frame
            frame_j: Index of the later frame
            interp_pos: Interpolation position in [0, 1]

        Returns:
            Adjusted timestep value for the model
        """
        duration = self.pts[frame_j] - self.pts[frame_i]
        avg_duration = 1.0 / self.avg_fps
        return interp_pos * (duration / avg_duration)


@dataclass
class FrameData:
    """Frame data container.

    Carries frame pixel data between components. Supports both numpy arrays
    and PyTorch tensors with format-aware conversion.

    Attributes:
        data: Frame pixel data (numpy array or torch tensor)
        frame_idx: Frame index in the source sequence
        format: Frame data format
        metadata: Optional metadata dictionary
    """
    data: Union[np.ndarray, torch.Tensor]
    frame_idx: int
    format: FrameFormat = FrameFormat.RGB_FLOAT
    metadata: dict = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        """Convert data to numpy array.

        Returns:
            numpy ndarray (CPU, contiguous)
        """
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return np.asarray(self.data)

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert data to torch tensor on specified device.

        Args:
            device: Target device string (e.g., "cpu", "cuda:0")

        Returns:
            torch.Tensor on the specified device
        """
        if isinstance(self.data, torch.Tensor):
            return self.data.to(device)
        return torch.from_numpy(np.asarray(self.data)).to(device)


@dataclass
class ProcessedFrameData:
    """Frame data after processing.

    Carries processed frame data with interpolation metadata.

    Attributes:
        data: Processed frame pixel data
        source_frame_idx: Original frame index that produced this output
        interpolated: Whether this frame was interpolated (vs. original)
        interpolation_ratio: Interpolation position (0.0 = original, >0 = interpolated)
        metadata: Processing metadata dictionary
    """
    data: Union[np.ndarray, torch.Tensor]
    source_frame_idx: int
    interpolated: bool = False
    interpolation_ratio: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        """Convert data to numpy array.

        Returns:
            numpy ndarray (CPU, contiguous)
        """
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return np.asarray(self.data)

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert data to torch tensor on specified device.

        Args:
            device: Target device string

        Returns:
            torch.Tensor on the specified device
        """
        if isinstance(self.data, torch.Tensor):
            return self.data.to(device)
        return torch.from_numpy(np.asarray(self.data)).to(device)


@dataclass
class FrameRef:
    """Lightweight frame reference (does not hold pixel data).

    Used for scheduling and cache indexing without memory overhead.

    Attributes:
        source_path: Path to the source video file
        frame_index: Frame index within the source
        cache_key: Unique cache lookup key (f"{source_path}:{frame_index}")
    """
    source_path: str
    frame_index: int
    cache_key: str  # f"{source_path}:{frame_index}"


@dataclass
class FrameBundle:
    """Frame data bundle returned by IO to Task.

    Attributes:
        frames: Tensor of loaded frames [N, H, W, C] or [N, C, H, W]
        metadata: Video metadata for the source
        format: Frame data format
    """
    frames: torch.Tensor           # [N, H, W, C] or [N, C, H, W]
    metadata: VideoMetadata
    format: FrameFormat


@dataclass
class FramePair:
    """Frame pair: the minimal input unit for interpolation.

    Attributes:
        frame0: First frame tensor [C, H, W] float32
        frame1: Second frame tensor, None for the last frame
        index: Frame0's source index
        pts: Frame0's precise PTS (for VFR support)
        pts_next: Frame1's precise PTS (for VFR support)
    """
    frame0: torch.Tensor          # [C, H, W] float32
    frame1: Optional[torch.Tensor]  # None = last frame
    index: int                     # frame0's index
    pts: Optional[float] = None    # frame0's precise PTS
    pts_next: Optional[float] = None  # frame1's precise PTS


# ====================
# 4. IO Request / Response
# ====================


@dataclass
class IORequest:
    """IO request submitted to the IO scheduler.

    Attributes:
        request_id: Unique request identifier
        request_type: Type of IO operation
        file_path: Path to the source file
        consumer_id: ID of the consuming component
        frame_indices: Specific frame indices to load (None = all)
        priority: Request priority (higher = processed first)
    """
    request_id: str
    request_type: IORequestType
    file_path: str
    consumer_id: str
    frame_indices: Optional[List[int]] = None
    priority: int = 0


@dataclass
class IOResponse:
    """IO response returned to the requesting component.

    Attributes:
        request_id: Matches the original IORequest.request_id
        success: Whether the IO operation succeeded
        data: Loaded frame bundle (on success)
        error: Error message (on failure)
        from_cache: Whether data was served from cache
        released: Whether resources were successfully released
    """
    request_id: str
    success: bool
    data: Optional[FrameBundle] = None
    error: Optional[str] = None
    from_cache: bool = False
    released: bool = False


@dataclass
class CachedFrameBundle:
    """Cached frame data bundle with reference tracking.

    Attributes:
        file_path: Source video file path
        frames: Cached frame tensor
        metadata: Video metadata
        consumers: Set of consumer IDs currently using this data
        loaded_at: Timestamp when data was loaded
        memory_size: Estimated memory size in bytes
    """
    file_path: str
    frames: torch.Tensor
    metadata: VideoMetadata
    consumers: set[str] = field(default_factory=set)
    loaded_at: datetime = field(default_factory=datetime.now)
    memory_size: int = 0


# ====================
# 5. Inference Request / Result
# ====================


@dataclass
class InferenceRequest:
    """Inference request: Task -> Backend.

    Attributes:
        frame0: First input frame tensor [C, H, W] float32
        frame1: Second input frame tensor [C, H, W] float32
        timestep: Interpolation position in [0, 1]
        model_config: Model-specific configuration dict
    """
    frame0: torch.Tensor          # [C, H, W] float32
    frame1: torch.Tensor          # [C, H, W] float32
    timestep: float               # [0, 1]
    model_config: Dict[str, Any]


@dataclass
class InferenceResult:
    """Inference result: Backend -> Task.

    Attributes:
        output_frame: Interpolated frame tensor [C, H, W] float32
        success: Whether inference succeeded
        error: Error message (on failure)
        inference_time_ms: Wall-clock inference time in milliseconds
    """
    output_frame: torch.Tensor    # [C, H, W] float32
    success: bool
    error: Optional[str] = None
    inference_time_ms: float = 0.0


@dataclass
class ProcessingResult:
    """Result of a video processing operation.

    Attributes:
        success: Whether processing succeeded
        output_path: Path to output file (if successful)
        error_message: Error message (if failed)
        processing_time: Total processing time in seconds
        stats: Additional statistics
    """
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    stats: Optional[Dict[str, Any]] = None


# ====================
# 6. Task Definitions
# ====================


@dataclass
class TaskDescriptor:
    """Task descriptor submitted by UI.

    Attributes:
        video_path: Path to input video file
        pipeline_config: Pipeline configuration dict from UI
        image_sequence_frames: Optional list of image file paths for sequences
    """
    video_path: str
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    image_sequence_frames: Optional[List[str]] = None


@dataclass
class SubTaskPlan:
    """Sub-task plan computed during task parsing.

    Attributes:
        total_subtasks: Total number of subtasks to generate
        input_frame_count: Number of input frames
        output_frame_count: Expected output frame count
        multiplier: Frame multiplier (e.g., 2 for 2x interpolation)
        batch_size: Batch size for grouped processing
        requires_scene_detect: Whether scene detection is needed
    """
    total_subtasks: int
    input_frame_count: int
    output_frame_count: int
    multiplier: int
    batch_size: int
    requires_scene_detect: bool


@dataclass
class TaskDefinition:
    """Parsed and resolved task definition.

    Attributes:
        task_id: Unique task identifier
        video_path: Path to input video
        backend_type: Selected backend type
        backend_config: Resolved backend configuration
        processing_config: Resolved processing configuration
        subtask_plan: Computed sub-task plan
        output_path: Resolved output file path
        image_sequence_frames: Optional image sequence paths
        created_at: Task creation timestamp
        priority: Task priority (higher = processed first)
    """
    task_id: str
    video_path: str
    backend_type: BackendType
    backend_config: BackendConfig
    processing_config: ProcessingConfig
    subtask_plan: SubTaskPlan
    output_path: Path
    image_sequence_frames: Optional[List[str]] = None
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0


@dataclass
class SubTask:
    """Sub-task: the minimal scheduling unit.

    Attributes:
        subtask_id: Unique sub-task identifier
        parent_task_id: ID of the parent TaskDefinition
        input_frames: List of frame references to process
        model_config: Model-specific configuration for this subtask
        depends_on: IDs of subtasks that must complete first
        required_files: Files that must be loaded before execution
        output_frame_refs: Frame references for output frames
        state: Current sub-task state
        retry_count: Number of retry attempts on failure
    """
    subtask_id: str
    parent_task_id: str
    input_frames: List[FrameRef]
    model_config: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    required_files: List[str] = field(default_factory=list)
    output_frame_refs: List[FrameRef] = field(default_factory=list)
    state: SubTaskState = SubTaskState.PENDING
    retry_count: int = 0


# ====================
# 7. Pre-processing Decision Types
# ====================


@dataclass
class FramePairDecision:
    """Pre-processing decision for a frame pair.

    Attributes:
        frame0_index: Index of the first frame
        frame1_index: Index of the second frame
        action: Decision action (interpolate, scene_cut, duplicate, last_frame)
        reason: Human-readable reason for the decision
    """
    frame0_index: int
    frame1_index: int
    action: FramePairAction
    reason: str = ""


@dataclass
class ValidationResult:
    """Result validation outcome.

    Attributes:
        valid: Whether the result passed validation
        error: Error description (if invalid)
    """
    valid: bool
    error: Optional[str] = None


@dataclass
class TaskCheckpoint:
    """Checkpoint for resumable video processing.

    Stores progress information (NOT frame data) to allow resuming
    interrupted processing from the last completed frame.

    Attributes:
        task_id: Unique task identifier (matches TaskDefinition.task_id)
        video_path: Path to input video file
        output_path: Path to output file
        last_completed_frame: Index of the last successfully written frame
        total_frames: Total number of input frames
        multiplier: Frame multiplier (e.g., 2 for 2x interpolation)
        codec: Output codec name
        created_at: When the checkpoint was first created
        updated_at: When the checkpoint was last updated
    """
    task_id: str
    video_path: str
    output_path: str
    last_completed_frame: int
    total_frames: int
    multiplier: int = 2
    codec: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ====================
# __all__ exports
# ====================

__all__ = [
    # Enums
    "BackendType",
    "TaskState",
    "SubTaskState",
    "FramePairAction",
    "IORequestType",
    "FrameFormat",
    "DecoderType",
    "InferenceStrategy",
    "SceneDetectionMethod",
    "EngineStatus",
    # Config dataclasses
    "BackendConfig",
    "ProcessingConfig",
    "ColorSpaceInfo",
    "AudioConfig",
    "ParallelConfig",
    # IO data structures
    "VideoMetadata",
    "FrameTimestamps",
    "FrameData",
    "ProcessedFrameData",
    "FrameRef",
    "FrameBundle",
    "FramePair",
    # IO request / response
    "IORequest",
    "IOResponse",
    "CachedFrameBundle",
    # Inference types
    "InferenceRequest",
    "InferenceResult",
    "ProcessingResult",
    # Task definitions
    "TaskDescriptor",
    "SubTaskPlan",
    "TaskDefinition",
    "SubTask",
    # Pre-processing types
    "FramePairDecision",
    "ValidationResult",
    # Checkpoint types
    "TaskCheckpoint",
    # Engine types (re-exported from core.engine_manager)
    # EngineInstance is in core.engine_manager to avoid circular imports
]