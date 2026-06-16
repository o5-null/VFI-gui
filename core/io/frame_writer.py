"""Video frame writer component.

Handles all video output operations, writing processed frames to files.
Separates IO concerns from backend processing logic.

Uses PyAV-based writers for direct encoding and audio muxing:
- Direct audio stream copying from input container (no temp files)
- Proper color space metadata preservation
- Native encoder configuration
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Callable, List, Union
import warnings

import numpy as np
import torch
from loguru import logger

# Old imports (kept for backward compatibility with existing classes)
from core.io.frame_data import (
    FrameData,
    ProcessedFrameData,
    VideoMetadata,
    FrameFormat,
    VideoFrameSequence,
)

# New imports for PyAV-based writers
from core.types import (
    VideoMetadata as VideoMetadataNew,
    ProcessedFrameData as ProcessedFrameDataNew,
    AudioConfig,
    ColorSpaceInfo,
    FrameFormat as FrameFormatNew,
)


class FrameWriter(ABC):
    """Abstract base class for frame writers."""

    @abstractmethod
    def open(
        self,
        output_path: Union[str, Path],
        metadata: VideoMetadata,
    ) -> None:
        """Open output for streaming writes.

        Args:
            output_path: Output file path
            metadata: Video metadata
        """
        pass

    @abstractmethod
    def write_frame(self, frame: ProcessedFrameData) -> None:
        """Write single frame (streaming).

        Args:
            frame: Processed frame data
        """
        pass

    @abstractmethod
    def write_frames(
        self,
        frames: Iterator[ProcessedFrameData],
        output_path: Union[str, Path],
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Write frames to output (batch mode).

        Args:
            frames: Iterator of processed frames
            output_path: Output file path
            metadata: Video metadata
            progress_callback: Callback(current, total)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close writer and cleanup resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# PyAV-based Writers (New Implementation)
# =============================================================================


class PyAVVideoWriter(FrameWriter):
    """Video file writer using PyAV for direct encoding and audio muxing.

    Replaces FFmpeg subprocess with native PyAV encoding. Supports direct
    audio stream copying from input container without intermediate files.
    """

    # Color space name to FFmpeg int mapping
    _COLORSPACE_MAP = {
        "bt709": 1,
        "bt601": 5,
        "bt2020": 9,
        "smpte170m": 6,
        "smpte240m": 7,
    }

    _TRANSFER_MAP = {
        "sdr": 1,
        "bt709": 1,
        "gamma22": 4,
        "gamma28": 5,
        "smpte170m": 6,
        "smpte240m": 7,
        "linear": 8,
        "pq": 16,
        "hlg": 18,
        "bt470bg": 5,
    }

    _PRIMARIES_MAP = {
        "bt709": 1,
        "bt601": 5,
        "bt2020": 9,
        "smpte170m": 6,
        "smpte240m": 7,
    }

    def __init__(self, codec_config: Optional[dict] = None):
        """Initialize PyAV video writer.

        Args:
            codec_config: Codec configuration with keys:
                - codec: Codec name (default: "hevc_nvenc")
                - quality: CRF/CQ value (default: 22)
                - preset: Encoder preset (default: "p4")
                - audio_copy: Whether to copy audio (default: True)
        """
        config = codec_config or {}
        self._codec_name = config.get("codec", "hevc_nvenc")
        self._quality = config.get("quality", 22)
        self._preset = config.get("preset", "p4")
        self._audio_config = AudioConfig(
            mode=config.get("audio_copy", True) and "copy" or "none",
        )
        self._output = None
        self._video_stream = None
        self._input_container = None
        self._audio_streams = []
        self._audio_output_streams = []
        self._frame_count = 0
        self._cancelled = False
        self._output_path = None
        self._metadata = None

    def open(
        self,
        output_path: Union[str, Path],
        metadata: VideoMetadata,
        input_container=None,
    ) -> None:
        """Open output for writing.

        Args:
            output_path: Output file path
            metadata: Video metadata
            input_container: Optional PyAV input container for audio muxing
        """
        import av

        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = metadata

        # Create output container
        self._output = av.open(str(self._output_path), mode="w")

        # Add video stream
        fps = metadata.fps
        self._video_stream = self._output.add_stream(self._codec_name, fps=fps)
        self._video_stream.width = metadata.width
        self._video_stream.height = metadata.height
        self._video_stream.pix_fmt = "yuv420p"

        # Restore color space from metadata
        if hasattr(metadata, "color_space") and metadata.color_space:
            self._apply_color_space(metadata.color_space)

        # Set encoder-specific options
        self._apply_encoder_options()

        # Setup audio stream direct mapping (zero temp files)
        self._input_container = input_container
        if input_container and self._audio_config.mode == "copy":
            self._setup_audio_copy(input_container)

        logger.info(
            f"Opened PyAV writer: {self._codec_name} @ {metadata.width}x{metadata.height}, "
            f"{fps} fps, quality={self._quality}"
        )

    def _apply_color_space(self, color_space: ColorSpaceInfo) -> None:
        """Apply color space metadata to video stream."""
        self._video_stream.codec_context.colorspace = self._map_colorspace(
            color_space.matrix
        )
        self._video_stream.codec_context.color_trc = self._map_transfer(
            color_space.transfer
        )
        self._video_stream.codec_context.color_primaries = self._map_primaries(
            color_space.primaries
        )
        self._video_stream.codec_context.color_range = (
            1 if color_space.range == "limited" else 2
        )

    def _apply_encoder_options(self) -> None:
        """Apply encoder-specific options."""
        if self._codec_name.endswith("_nvenc"):
            # NVIDIA NVENC options
            self._video_stream.codec_context.options = {
                "preset": self._preset,
                "cq": str(self._quality),
            }
        elif self._codec_name.startswith("libx264") or self._codec_name.startswith(
            "libx265"
        ):
            # x264/x265 options
            preset = (
                self._preset
                if self._preset in ["ultrafast", "superfast", "veryfast", "faster",
                                   "fast", "medium", "slow", "slower", "veryslow"]
                else "medium"
            )
            self._video_stream.codec_context.options = {
                "crf": str(self._quality),
                "preset": preset,
            }
        else:
            # Generic encoder - try preset and quality
            self._video_stream.codec_context.options = {
                "crf": str(self._quality),
                "preset": self._preset if self._preset.startswith("p") else "medium",
            }

    def _setup_audio_copy(self, input_container) -> None:
        """Setup audio stream copying from input container."""
        for audio_stream in input_container.streams.audio:
            out_stream = self._output.add_stream(template=audio_stream)
            self._audio_streams.append(audio_stream)
            self._audio_output_streams.append(out_stream)
            logger.debug(f"Audio stream {audio_stream.index} will be copied")

    @staticmethod
    def _map_colorspace(name: str) -> int:
        """Map color space name to FFmpeg int value."""
        return PyAVVideoWriter._COLORSPACE_MAP.get(name.lower(), 1)

    @staticmethod
    def _map_transfer(name: str) -> int:
        """Map transfer characteristic name to FFmpeg int value."""
        return PyAVVideoWriter._TRANSFER_MAP.get(name.lower(), 1)

    @staticmethod
    def _map_primaries(name: str) -> int:
        """Map color primaries name to FFmpeg int value."""
        return PyAVVideoWriter._PRIMARIES_MAP.get(name.lower(), 1)

    def write_frame(self, frame: ProcessedFrameData) -> None:
        """Write single frame.

        Args:
            frame: Processed frame data
        """
        import av

        if self._output is None or self._video_stream is None:
            raise RuntimeError("Writer not opened")

        if self._cancelled:
            return

        # Convert to uint8 numpy array
        arr = frame.to_numpy()
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

        # Create AVFrame from numpy array
        av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")

        # Encode and mux
        for packet in self._video_stream.encode(av_frame):
            self._output.mux(packet)

        self._frame_count += 1

    def write_frames(
        self,
        frames: Iterator[ProcessedFrameData],
        output_path: Union[str, Path],
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Write all frames."""
        self.open(output_path, metadata)

        try:
            frame_count = 0
            total_frames = metadata.total_frames

            for frame in frames:
                if self._cancelled:
                    break

                self.write_frame(frame)
                frame_count += 1

                if progress_callback and frame_count % 10 == 0:
                    progress_callback(frame_count, total_frames)

            logger.info(f"Encoded {frame_count} frames to video")

        finally:
            self.close()

    def _mux_audio_packets(self) -> None:
        """Mux audio packets from input container to output."""
        if not self._input_container or not self._audio_streams:
            return

        logger.debug(f"Muxing {len(self._audio_streams)} audio streams...")

        for i, audio_stream in enumerate(self._audio_streams):
            # Find corresponding output stream by enumeration index
            out_stream = self._audio_output_streams[i]

            for packet in self._input_container.demux(audio_stream):
                if packet.dts is None:
                    continue

                # Remap packet to output stream
                packet.stream = out_stream
                self._output.mux(packet)

    def close(self) -> None:
        """Close writer and finalize output."""
        if self._output is None:
            return

        try:
            # Flush video encoder
            if self._video_stream:
                for packet in self._video_stream.encode():
                    self._output.mux(packet)

            # Mux audio packets (direct copy, no temp files)
            if self._input_container and self._audio_config.mode == "copy":
                self._mux_audio_packets()

            logger.info(f"Closed PyAV writer: {self._frame_count} frames written")

        finally:
            # Close output container
            if self._output:
                self._output.close()
                self._output = None

            # Don't close input_container - caller owns it
            self._input_container = None
            self._video_stream = None
            self._audio_streams = []
            self._audio_output_streams = []

    def cancel(self) -> None:
        """Cancel writing."""
        self._cancelled = True


class PyAVImageWriter(FrameWriter):
    """Image sequence writer using PIL for image output.

    Writes frames as individual image files in a directory.
    """

    def __init__(
        self,
        image_format: str = "png",
        quality: int = 95,
    ):
        """Initialize PyAV image writer.

        Args:
            image_format: Image format (png, jpg, tiff, etc.)
            quality: JPEG quality (1-100)
        """
        self._format = image_format.lower()
        self._quality = quality
        self._output_dir: Optional[Path] = None
        self._frame_count = 0
        self._cancelled = False

    def open(
        self,
        output_path: Union[str, Path],
        metadata: VideoMetadata,
    ) -> None:
        """Open output directory for writing.

        Args:
            output_path: Output path (used as base directory for images)
            metadata: Video metadata
        """
        from PIL import Image

        output_path = Path(output_path)

        # Create output directory
        self._output_dir = output_path.parent
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0

        logger.info(
            f"Writing {metadata.total_frames} frames as {self._format.upper()} "
            f"sequence to {self._output_dir}"
        )

    def write_frame(self, frame: ProcessedFrameData) -> None:
        """Write single frame as image.

        Args:
            frame: Processed frame data
        """
        from PIL import Image

        if self._output_dir is None:
            raise RuntimeError("Writer not opened")

        if self._cancelled:
            return

        # Convert to uint8 numpy array
        arr = frame.to_numpy()
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

        # Build filename using source frame index
        filename = f"frame_{frame.source_frame_idx:06d}.{self._format}"
        filepath = self._output_dir / filename

        # Save image
        img = Image.fromarray(arr)
        if self._format in ("jpg", "jpeg"):
            img.save(filepath, quality=self._quality)
        else:
            img.save(filepath)

        self._frame_count += 1

    def write_frames(
        self,
        frames: Iterator[ProcessedFrameData],
        output_path: Union[str, Path],
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Write all frames as image sequence."""
        self.open(output_path, metadata)

        frame_count = 0
        total_frames = metadata.total_frames

        for frame in frames:
            if self._cancelled:
                break

            self.write_frame(frame)
            frame_count += 1

            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, total_frames)

        logger.info(f"Saved {frame_count} {self._format.upper()} images to {self._output_dir}")

    def close(self) -> None:
        """Close writer."""
        logger.debug(f"Closed image writer: {self._frame_count} frames written")
        self._output_dir = None

    def cancel(self) -> None:
        """Cancel writing."""
        self._cancelled = True


# =============================================================================
# Factory
# =============================================================================


class FrameWriterFactory:
    """Factory for creating appropriate frame writer."""

    @staticmethod
    def create_writer(
        output_path: Union[str, Path] = None,
        codec_config: Optional[dict] = None,
        output_mode: Optional[str] = None,
        **kwargs,
    ) -> FrameWriter:
        """Create appropriate writer for output.

        Args:
            output_path: Output file path (optional for backward compatibility)
            codec_config: Optional codec configuration
            output_mode: Output mode ("video" or "images"), auto-detected if None
            **kwargs: Additional arguments (ignored, for forward compatibility)

        Returns:
            FrameWriter instance (PyAVVideoWriter for video, PyAVImageWriter for images)
        """
        # Determine output mode
        effective_output_mode = output_mode or (codec_config or {}).get(
            "output_mode", "video"
        )

        if effective_output_mode == "images":
            # Image sequence output
            image_format = (codec_config or {}).get("image_format", "png")
            quality = (codec_config or {}).get("image_quality", 95)
            return PyAVImageWriter(image_format=image_format, quality=quality)

        # Video output
        return PyAVVideoWriter(codec_config=codec_config or {})

    @staticmethod
    def is_image_output(output_path: Union[str, Path] = None) -> bool:
        """Check if output path indicates image sequence.

        Args:
            output_path: Output path to check (optional)

        Returns:
            True if image sequence output
        """
        from core.codec_manager import get_codec_manager

        codec_manager = get_codec_manager()
        return codec_manager.is_image_output()


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description="Frame Writer CLI")
    parser.add_argument("--input", required=True, help="Input video (for metadata)")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--codec", default="libx265", help="Codec name")
    parser.add_argument("--quality", type=int, default=22, help="CRF/CQ quality")
    parser.add_argument("--preset", default="medium", help="Encoder preset")
    parser.add_argument("--frames", type=int, default=30, help="Number of test frames")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    try:
        import av

        # Read metadata from input video
        input_container = av.open(args.input)
        video_stream = input_container.streams.video[0]

        # Create VideoMetadata using local class for compatibility
        metadata = VideoMetadata(
            width=video_stream.width,
            height=video_stream.height,
            fps=float(video_stream.average_rate),
            total_frames=video_stream.frames or args.frames,
        )

        # Create writer
        codec_config = {
            "codec": args.codec,
            "quality": args.quality,
            "preset": args.preset,
            "audio_copy": True,
        }

        writer = PyAVVideoWriter(codec_config=codec_config)

        # Generate test frames (solid color gradient)
        start_time = time.perf_counter()

        writer.open(args.output, metadata, input_container=input_container)

        frame_count = min(args.frames, metadata.total_frames)
        for i in range(frame_count):
            # Create test frame (gradient pattern)
            r = np.linspace(0, 255, metadata.width, dtype=np.uint8)
            g = np.linspace(0, 255, metadata.height, dtype=np.uint8)
            r_grid, g_grid = np.meshgrid(r, g)
            b = np.full_like(r_grid, (i * 255 // frame_count), dtype=np.uint8)
            frame_data = np.stack([r_grid, g_grid, b], axis=-1)

            # Create ProcessedFrameData using local class for compatibility
            frame = ProcessedFrameData(data=frame_data, source_frame_idx=i)
            writer.write_frame(frame)

        writer.close()
        input_container.close()

        elapsed = time.perf_counter() - start_time

        result = {
            "success": True,
            "output": args.output,
            "codec": args.codec,
            "frames_written": frame_count,
            "resolution": f"{metadata.width}x{metadata.height}",
            "fps": metadata.fps,
            "elapsed_seconds": round(elapsed, 3),
            "fps_achieved": round(frame_count / elapsed, 2),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Output: {result['output']}")
            print(f"Codec: {result['codec']}")
            print(f"Frames: {result['frames_written']}")
            print(f"Resolution: {result['resolution']}")
            print(f"Time: {result['elapsed_seconds']}s ({result['fps_achieved']} fps)")

    except Exception as e:
        result = {"success": False, "error": str(e)}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {e}")
        raise
