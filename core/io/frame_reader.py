"""Video frame reader component.

Handles all video input operations, reading frames into memory
for processing by backend components.

Uses PyAV-based readers for optimal performance with:
- Correct color space conversion (FFmpeg uses proper matrix coefficients)
- Precise PTS for VFR video support
- Native 10bit/12bit/HDR support
- Multi-threaded decoding
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Callable, List, Union
import warnings

import numpy as np
import torch
from loguru import logger

# Legacy imports for fallback classes (keep unchanged)
from core.io.frame_data import FrameData as LegacyFrameData
from core.io.frame_data import FrameBatch, VideoMetadata as LegacyVideoMetadata
from core.io.frame_data import VideoFrameSequence, FrameFormat

# Canonical type imports for new PyAV classes
from core.types import VideoMetadata, ColorSpaceInfo, FrameData, FrameFormat as TypesFrameFormat


class FrameReader(ABC):
    """Abstract base class for frame readers."""
    
    @abstractmethod
    def read_frames(
        self,
        source: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VideoFrameSequence:
        """Read all frames from source.
        
        Args:
            source: Video file path or image sequence pattern
            progress_callback: Callback(current, total)
            
        Returns:
            VideoFrameSequence containing all frames
        """
        pass
    
    @abstractmethod
    def read_frames_iter(
        self,
        source: Union[str, Path],
    ) -> Iterator[LegacyFrameData]:
        """Read frames as iterator (memory efficient).
        
        Args:
            source: Video file path or image sequence pattern
            
        Yields:
            FrameData objects
        """
        pass
    
    @abstractmethod
    def get_metadata(self, source: Union[str, Path]) -> LegacyVideoMetadata:
        """Get video metadata without reading frames.
        
        Args:
            source: Video file path
            
        Returns:
            VideoMetadata
        """
        pass


# =============================================================================
# PyAV-based Readers (Primary Implementation)
# =============================================================================


class PyAVVideoReader:
    """Video file frame reader using PyAV.
    
    Replaces OpenCV-based VideoFrameReader with:
    - Correct color space conversion (FFmpeg uses proper matrix coefficients)
    - Precise PTS for VFR video support
    - Native 10bit/12bit/HDR support
    - Multi-threaded decoding
    """
    
    def __init__(
        self,
        video_path: str,
        target_format: str = "rgb_f",
        device: str = "cpu",
        hw_accel: str = "",
    ):
        """Initialize PyAV video reader.
        
        Args:
            video_path: Path to video file
            target_format: Target format - "rgb_f" (float32), "rgb" (uint8)
            device: Target device for tensors (e.g., "cpu", "cuda:0")
            hw_accel: Hardware acceleration type (e.g., "cuda", "qsv")
        """
        import av
        
        self._video_path = video_path
        self._target_format = target_format
        self._device = device
        self._hw_accel = hw_accel
        
        # Open container and stream
        self._container = av.open(video_path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"  # Multi-threaded decoding
        
        # Extract metadata
        self._metadata = self._extract_metadata()
        
        # Map target_format string to FrameFormat enum
        self._frame_format = TypesFrameFormat.RGB_FLOAT if target_format == "rgb_f" else TypesFrameFormat.RGB
    
    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        return self._metadata
    
    def read_frames(
        self,
        start: int = 0,
        count: Optional[int] = None,
    ) -> List[FrameData]:
        """Read frames from video.
        
        Args:
            start: Starting frame index (default: 0)
            count: Number of frames to read (None = read all)
            
        Returns:
            List of FrameData objects
        """
        frames: List[FrameData] = []
        frame_idx = 0
        frames_read = 0
        
        for av_frame in self._container.decode(self._stream):
            # Skip frames before start
            if frame_idx < start:
                frame_idx += 1
                continue
            
            # Stop if we've read enough frames
            if count is not None and frames_read >= count:
                break
            
            # Convert frame
            frame_data = self._convert_frame(av_frame, frame_idx)
            frames.append(frame_data)
            
            frame_idx += 1
            frames_read += 1
        
        logger.info(f"Read {len(frames)} frames from {self._video_path}")
        return frames
    
    def read_frames_iter(
        self,
        source: Optional[str] = None,
    ) -> Iterator[FrameData]:
        """Read frames as iterator.
        
        Args:
            source: Ignored (uses video_path from constructor)
            
        Yields:
            FrameData objects
        """
        frame_idx = 0
        
        for av_frame in self._container.decode(self._stream):
            frame_data = self._convert_frame(av_frame, frame_idx)
            yield frame_data
            frame_idx += 1
    
    def get_metadata(self, source: Optional[str] = None) -> VideoMetadata:
        """Get video metadata.
        
        Args:
            source: Ignored (uses cached metadata)
            
        Returns:
            VideoMetadata object
        """
        return self._metadata
    
    def close(self) -> None:
        """Close the video container."""
        if self._container:
            self._container.close()
    
    def __enter__(self) -> "PyAVVideoReader":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def _convert_frame(self, av_frame, frame_idx: int) -> FrameData:
        """Convert PyAV frame to FrameData.
        
        PyAV handles color space conversion automatically via
        to_ndarray(format="rgb24") using FFmpeg's internal matrix coefficients.
        """
        # PyAV handles YUV->RGB color space conversion automatically
        # No manual cv2.cvtColor needed
        arr = av_frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8
        
        # Extract PTS (presentation timestamp)
        pts = float(av_frame.pts * av_frame.time_base) if av_frame.pts is not None else None
        
        # Convert to target format
        if self._target_format == "rgb_f":
            data = torch.from_numpy(arr).float() / 255.0
            if self._device != "cpu":
                data = data.to(self._device)
        else:
            data = arr
        
        return FrameData(
            data=data,
            frame_idx=frame_idx,
            format=self._frame_format,
            metadata={"pts": pts},
        )
    
    def _extract_metadata(self) -> VideoMetadata:
        """Extract video metadata from PyAV stream info."""
        stream = self._stream
        codec_context = stream.codec_context
        
        # Basic metadata
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        total_frames = stream.frames if stream.frames > 0 else 0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        
        # Color space info
        color_space = ColorSpaceInfo(
            matrix=self._map_colorspace(codec_context.colorspace),
            transfer=self._map_transfer(codec_context.color_trc),
            primaries=self._map_primaries(codec_context.color_primaries),
            range="limited" if codec_context.color_range == 1 else "full",
        )
        
        # Audio stream info
        has_audio = len(self._container.streams.audio) > 0
        audio_codec = ""
        audio_sample_rate = 0
        if has_audio:
            audio_stream = self._container.streams.audio[0]
            audio_codec = audio_stream.codec_context.name
            audio_sample_rate = audio_stream.codec_context.sample_rate
        
        # VFR detection
        avg_rate = stream.average_rate
        r_frame_rate = stream.codec_context.framerate
        is_vfr = avg_rate != r_frame_rate if avg_rate and r_frame_rate else False
        
        return VideoMetadata(
            width=codec_context.width,
            height=codec_context.height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec_context.name,
            pixel_format=codec_context.pix_fmt,
            is_vfr=is_vfr,
            color_space=color_space,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
        )
    
    @staticmethod
    def _map_colorspace(val: int) -> str:
        """Map PyAV color matrix to standard name."""
        mapping: dict = {1: "bt709", 5: "bt601", 9: "bt2020"}
        return mapping.get(val, "bt709")
    
    @staticmethod
    def _map_transfer(val: int) -> str:
        """Map PyAV transfer characteristic to standard name."""
        mapping: dict = {1: "sdr", 16: "pq", 18: "hlg"}
        return mapping.get(val, "sdr")
    
    @staticmethod
    def _map_primaries(val: int) -> str:
        """Map PyAV color primaries to standard name."""
        mapping: dict = {1: "bt709", 5: "bt601", 9: "bt2020"}
        return mapping.get(val, "bt709")


class PyAVImageReader:
    """Image sequence reader using PyAV with PIL fallback.
    
    Reads image sequences using PyAV for efficient decoding,
    falling back to PIL for unsupported formats.
    """
    
    def __init__(
        self,
        image_dir: str,
        pattern: str = "*.png",
        target_format: str = "rgb_f",
        device: str = "cpu",
    ):
        """Initialize PyAV image reader.
        
        Args:
            image_dir: Directory containing images or path to single image
            pattern: Glob pattern for filtering images (default: "*.png")
            target_format: Target format - "rgb_f" (float32), "rgb" (uint8)
            device: Target device for tensors
        """
        self._image_dir = Path(image_dir)
        self._pattern = pattern
        self._target_format = target_format
        self._device = device
        
        # Collect image files
        self._image_files = self._collect_image_files()
        
        if not self._image_files:
            raise RuntimeError(f"No images found at: {image_dir}")
        
        # Map target_format string to FrameFormat enum
        self._frame_format = TypesFrameFormat.RGB_FLOAT if target_format == "rgb_f" else TypesFrameFormat.RGB
        
        # Cache metadata from first image
        self._metadata = self._extract_metadata()
    
    @property
    def metadata(self) -> VideoMetadata:
        """Get image sequence metadata."""
        return self._metadata
    
    def read_frames(
        self,
        start: int = 0,
        count: Optional[int] = None,
    ) -> List[FrameData]:
        """Read images from sequence.
        
        Args:
            start: Starting frame index (default: 0)
            count: Number of frames to read (None = read all)
            
        Returns:
            List of FrameData objects
        """
        frames: List[FrameData] = []
        
        end_idx = len(self._image_files) if count is None else min(start + count, len(self._image_files))
        
        for frame_idx in range(start, end_idx):
            img_path = self._image_files[frame_idx]
            frame_data = self._load_image(img_path, frame_idx)
            frames.append(frame_data)
        
        logger.info(f"Read {len(frames)} images from {self._image_dir}")
        return frames
    
    def read_frames_iter(
        self,
        source: Optional[str] = None,
    ) -> Iterator[FrameData]:
        """Read images as iterator.
        
        Args:
            source: Ignored (uses images from constructor)
            
        Yields:
            FrameData objects
        """
        for frame_idx, img_path in enumerate(self._image_files):
            yield self._load_image(img_path, frame_idx)
    
    def get_metadata(self, source: Optional[str] = None) -> VideoMetadata:
        """Get image sequence metadata.
        
        Args:
            source: Ignored (uses cached metadata)
            
        Returns:
            VideoMetadata object
        """
        return self._metadata
    
    def _collect_image_files(self) -> List[Path]:
        """Collect and sort image files from directory."""
        from core.utils.file_utils import IMAGE_EXTENSIONS, sort_files_naturally
        
        if self._image_dir.is_file():
            # Single file
            return [self._image_dir]
        
        # Directory - collect images
        image_files = [
            f for f in self._image_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        
        return sort_files_naturally(image_files)
    
    def _extract_metadata(self) -> VideoMetadata:
        """Extract metadata from first image."""
        import av
        
        first_img = self._image_files[0]
        
        # Try PyAV first
        try:
            with av.open(str(first_img)) as container:
                stream = container.streams.video[0]
                return VideoMetadata(
                    width=stream.codec_context.width,
                    height=stream.codec_context.height,
                    fps=30.0,  # Default for image sequences
                    total_frames=len(self._image_files),
                )
        except Exception:
            # Fall back to PIL
            from PIL import Image
            with Image.open(first_img) as img:
                width, height = img.size
            return VideoMetadata(
                width=width,
                height=height,
                fps=30.0,
                total_frames=len(self._image_files),
            )
    
    def _load_image(self, img_path: Path, frame_idx: int) -> FrameData:
        """Load single image using PyAV or PIL fallback."""
        import av
        
        # Try PyAV first
        try:
            with av.open(str(img_path)) as container:
                stream = container.streams.video[0]
                for av_frame in container.decode(stream):
                    # PyAV handles color space conversion
                    arr = av_frame.to_ndarray(format="rgb24")
                    
                    if self._target_format == "rgb_f":
                        data = torch.from_numpy(arr).float() / 255.0
                        if self._device != "cpu":
                            data = data.to(self._device)
                    else:
                        data = arr
                    
                    return FrameData(
                        data=data,
                        frame_idx=frame_idx,
                        format=self._frame_format,
                        metadata={"source_file": str(img_path)},
                    )
        except Exception as e:
            logger.debug(f"PyAV failed for {img_path}, falling back to PIL: {e}")
        
        # Fallback to PIL
        from PIL import Image
        
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        
        if self._target_format == "rgb_f":
            data = torch.from_numpy(arr).float() / 255.0
            if self._device != "cpu":
                data = data.to(self._device)
        else:
            data = arr
        
        return FrameData(
            data=data,
            frame_idx=frame_idx,
            format=self._frame_format,
            metadata={"source_file": str(img_path)},
        )


# =============================================================================
# Factory
# =============================================================================


class FrameReaderFactory:
    """Factory for creating appropriate frame reader."""
    
    @staticmethod
    def create_reader(
        source: Union[str, Path],
        target_format: FrameFormat = FrameFormat.RGB_FLOAT,
        device: str = "cpu",
        **kwargs,
    ) -> Union[PyAVVideoReader, PyAVImageReader]:
        """Create appropriate reader for source.
        
        Args:
            source: Video file or image sequence path
            target_format: Target frame format (legacy FrameFormat enum)
            device: Target device
            **kwargs: Additional arguments passed to reader
            
        Returns:
            FrameReader instance (PyAVVideoReader for videos, PyAVImageReader for images)
        """
        source_path = Path(source)
        
        from core.utils.file_utils import VIDEO_EXTENSIONS
        
        # Determine target_format string for PyAV readers
        target_format_str = "rgb_f"
        if target_format == FrameFormat.RGB:
            target_format_str = "rgb"
        
        # Video file
        if source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTENSIONS:
            return PyAVVideoReader(
                str(source_path),
                target_format=target_format_str,
                device=device,
                **kwargs,
            )
        
        # Image sequence (directory or single image)
        pattern = kwargs.get("pattern", "*.png")
        return PyAVImageReader(
            str(source_path),
            pattern=pattern,
            target_format=target_format_str,
            device=device,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse
    import json
    import time
    
    parser = argparse.ArgumentParser(description="Frame Reader CLI")
    parser.add_argument("--input", required=True, help="Video/image path")
    parser.add_argument("--count", type=int, default=10, help="Frames to read")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--metadata", action="store_true", help="Show metadata only")
    args = parser.parse_args()
    
    source_path = Path(args.input)
    
    from core.utils.file_utils import VIDEO_EXTENSIONS
    
    # Determine if video or image sequence
    if source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTENSIONS:
        # Video file - use PyAVVideoReader
        reader = PyAVVideoReader(str(source_path))
        
        if args.metadata:
            meta = reader.get_metadata()
            result = {
                "path": str(source_path),
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "total_frames": meta.total_frames,
                "duration": meta.duration,
                "codec": meta.codec,
                "pixel_format": meta.pixel_format,
                "is_vfr": meta.is_vfr,
                "has_audio": meta.has_audio,
                "audio_codec": meta.audio_codec,
                "audio_sample_rate": meta.audio_sample_rate,
                "color_space": {
                    "matrix": meta.color_space.matrix if meta.color_space else "unknown",
                    "transfer": meta.color_space.transfer if meta.color_space else "unknown",
                    "primaries": meta.color_space.primaries if meta.color_space else "unknown",
                    "range": meta.color_space.range if meta.color_space else "unknown",
                } if meta.color_space else None,
            }
            print(json.dumps(result, indent=2))
            reader.close()
        else:
            start_time = time.time()
            frames = reader.read_frames(count=args.count)
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = {
                "success": True,
                "path": str(source_path),
                "frames_read": len(frames),
                "time_ms": elapsed_ms,
                "avg_frame_ms": elapsed_ms / max(len(frames), 1),
                "first_frame_shape": list(frames[0].data.shape) if frames else None,
                "first_frame_pts": frames[0].metadata.get("pts") if frames else None,
            }
            
            reader.close()
            
            print(json.dumps(result, indent=2) if args.json else 
                  f"Read {len(frames)} frames in {elapsed_ms:.1f}ms ({elapsed_ms/len(frames):.1f}ms/frame)")
    else:
        # Image sequence - use PyAVImageReader
        reader = PyAVImageReader(str(source_path))
        
        if args.metadata:
            meta = reader.get_metadata()
            result = {
                "path": str(source_path),
                "width": meta.width,
                "height": meta.height,
                "total_frames": meta.total_frames,
            }
            print(json.dumps(result, indent=2))
        else:
            start_time = time.time()
            frames = reader.read_frames(count=args.count)
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = {
                "success": True,
                "path": str(source_path),
                "frames_read": len(frames),
                "time_ms": elapsed_ms,
                "avg_frame_ms": elapsed_ms / max(len(frames), 1),
                "first_frame_shape": list(frames[0].data.shape) if frames else None,
            }
            
            print(json.dumps(result, indent=2) if args.json else
                  f"Read {len(frames)} images in {elapsed_ms:.1f}ms ({elapsed_ms/len(frames):.1f}ms/frame)")
