"""Video frame writer component.

Handles all video output operations, writing processed frames to files.
Separates IO concerns from backend processing logic.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional, Callable, List, Union, BinaryIO, Tuple
import subprocess
import warnings

import numpy as np
import torch
from loguru import logger

from core.io.frame_data import (
    FrameData,
    ProcessedFrameData,
    VideoMetadata,
    FrameFormat,
    VideoFrameSequence,
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


class VideoFrameWriter(FrameWriter):
    """Video file writer using FFmpeg."""
    
    def __init__(
        self,
        codec_config: Optional[dict] = None,
        input_pixel_format: str = "rgb24",
    ):
        """Initialize video frame writer.
        
        Args:
            codec_config: Codec configuration
            input_pixel_format: Input pixel format for FFmpeg
        """
        self.codec_config = codec_config or {}
        self.input_pixel_format = input_pixel_format
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False
    
    def open(
        self,
        output_path: Union[str, Path],
        metadata: VideoMetadata,
    ) -> None:
        """Open output for writing.
        
        Args:
            output_path: Output file path
            metadata: Video metadata
        """
        from core.codec_manager import get_codec_manager, CodecConfig
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        codec_manager = get_codec_manager()
        
        # Create codec config from settings
        codec_config = CodecConfig.from_dict(self.codec_config)
        codec_manager.set_config(codec_config)
        
        # Build command
        cmd = codec_manager.build_ffmpeg_encode_args(
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            input_source="pipe:",
            output_path=str(output_path),
            include_audio=False,
        )
        
        logger.info(f"Encoding {metadata.total_frames} frames to video using {codec_config.codec}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        # Start FFmpeg process
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    
    def write_frame(self, frame: ProcessedFrameData) -> None:
        """Write single frame.
        
        Args:
            frame: Processed frame data
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Writer not opened")
        
        if self._cancelled:
            return
        
        # Convert to uint8 numpy array
        frame_data = frame.to_numpy()
        
        # Ensure uint8 format
        if frame_data.dtype != np.uint8:
            frame_uint8 = (np.clip(frame_data, 0, 1) * 255).astype(np.uint8)
        else:
            frame_uint8 = frame_data
        
        # Write to FFmpeg
        self._process.stdin.write(frame_uint8.tobytes())
    
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
    
    def close(self) -> None:
        """Close writer."""
        if self._process is not None:
            if self._process.stdin:
                self._process.stdin.close()
            
            if self._cancelled:
                self._process.terminate()
            else:
                self._process.wait()
                
                if self._process.returncode != 0:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else "Unknown error"
                    logger.error(f"FFmpeg error: {stderr}")
            
            self._process = None
    
    def cancel(self) -> None:
        """Cancel writing."""
        self._cancelled = True
        if self._process:
            self._process.terminate()


class ImageSequenceWriter(FrameWriter):
    """Image sequence writer."""
    
    def __init__(
        self,
        image_format: str = "png",
        quality: int = 95,
        frame_format: FrameFormat = FrameFormat.RGB,
    ):
        """Initialize image sequence writer.
        
        Args:
            image_format: Image format (png, jpg, tiff, etc.)
            quality: JPEG quality (1-100)
            frame_format: Expected input frame format
        """
        self.image_format = image_format.lower()
        self.quality = quality
        self.frame_format = frame_format
        self._output_dir: Optional[Path] = None
        self._base_name: str = "frame"
        self._cancelled = False
        self._frame_count = 0
    
    def open(
        self,
        output_path: Union[str, Path],
        metadata: VideoMetadata,
    ) -> None:
        """Open output directory for writing.

        Args:
            output_path: Output path (used as base for images)
            metadata: Video metadata (not used for images)
        """
        output_path = Path(output_path)

        # Create output directory
        self._output_dir = output_path.parent
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Get base filename
        self._base_name = output_path.stem
        self._frame_count = 0

        logger.info(f"Writing {metadata.total_frames} frames as {self.image_format.upper()} sequence to {self._output_dir}")
    
    def write_frame(self, frame: ProcessedFrameData) -> None:
        """Write single frame as image."""
        from PIL import Image
        
        if self._output_dir is None:
            raise RuntimeError("Writer not opened")
        
        if self._cancelled:
            return
        
        # Convert to numpy
        frame_data = frame.to_numpy()
        
        # Ensure uint8 format
        if frame_data.dtype != np.uint8:
            frame_uint8 = (np.clip(frame_data, 0, 1) * 255).astype(np.uint8)
        else:
            frame_uint8 = frame_data
        
        # Create PIL Image
        img = Image.fromarray(frame_uint8)
        
        # Build filename
        frame_path = self._output_dir / f"{self._base_name}_{self._frame_count:06d}.{self.image_format}"
        
        # Save
        if self.image_format in ("jpg", "jpeg"):
            img.save(frame_path, quality=self.quality)
        else:
            img.save(frame_path)
        
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
        
        logger.info(f"Saved {frame_count} {self.image_format.upper()} images to {self._output_dir}")
    
    def close(self) -> None:
        """Close writer."""
        self._output_dir = None
        self._frame_count = 0
    
    def cancel(self) -> None:
        """Cancel writing."""
        self._cancelled = True


class ParallelImageSequenceWriter(FrameWriter):
    """Parallel image sequence writer using multi-threading.

    Uses ThreadPoolExecutor to save multiple frames concurrently,
    significantly improving performance for image sequence output.
    """

    def __init__(
        self,
        image_format: str = "png",
        quality: int = 95,
        frame_format: FrameFormat = FrameFormat.RGB,
        max_workers: int = 4,
    ):
        """Initialize parallel image sequence writer.

        Args:
            image_format: Image format (png, jpg, tiff, etc.)
            quality: JPEG quality (1-100)
            frame_format: Expected input frame format
            max_workers: Maximum number of concurrent save threads
        """
        self.image_format = image_format.lower()
        self.quality = quality
        self.frame_format = frame_format
        self.max_workers = max_workers
        self._output_dir: Optional[Path] = None
        self._base_name: str = "frame"
        self._cancelled = False
        self._frame_count = 0

    def open(
        self,
        output_path: Union[str, Path],
        metadata: VideoMetadata,
    ) -> None:
        """Open output directory for writing.

        Args:
            output_path: Output path (used as base for images)
            metadata: Video metadata (not used for images)
        """
        output_path = Path(output_path)

        # Create output directory
        self._output_dir = output_path.parent
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Get base filename
        self._base_name = output_path.stem
        self._frame_count = 0

        logger.info(f"Writing image sequence to {self._output_dir} (parallel={self.max_workers})")

    def _save_single_frame(self, args: Tuple[int, np.ndarray]) -> int:
        """Save a single frame (called by worker threads).

        Args:
            args: Tuple of (frame_index, frame_data)

        Returns:
            Frame index that was saved
        """
        from PIL import Image

        if self._cancelled:
            return -1

        frame_idx, frame_uint8 = args

        # Create PIL Image
        img = Image.fromarray(frame_uint8)

        # Build filename
        frame_path = self._output_dir / f"{self._base_name}_{frame_idx:06d}.{self.image_format}"

        # Save
        if self.image_format in ("jpg", "jpeg"):
            img.save(frame_path, quality=self.quality)
        else:
            img.save(frame_path)

        return frame_idx

    def write_frames(
        self,
        frames: Iterator[ProcessedFrameData],
        output_path: Union[str, Path],
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Write all frames as image sequence using parallel processing."""
        from PIL import Image

        self.open(output_path, metadata)

        # Collect all frames first (convert to numpy in main thread)
        frame_list: List[Tuple[int, np.ndarray]] = []
        frame_idx = 0

        for frame in frames:
            if self._cancelled:
                break

            # Convert to numpy
            frame_data = frame.to_numpy()

            # Ensure uint8 format
            if frame_data.dtype != np.uint8:
                frame_uint8 = (np.clip(frame_data, 0, 1) * 255).astype(np.uint8)
            else:
                frame_uint8 = frame_data

            frame_list.append((frame_idx, frame_uint8))
            frame_idx += 1

        total_frames = len(frame_list)
        logger.info(f"Saving {total_frames} frames with {self.max_workers} workers...")

        # Use ThreadPoolExecutor for parallel saving
        saved_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._save_single_frame, args): args[0]
                for args in frame_list
            }

            # Process completed tasks
            for future in as_completed(future_to_idx):
                if self._cancelled:
                    executor.shutdown(wait=False)
                    break

                idx = future_to_idx[future]
                try:
                    future.result()
                    saved_count += 1

                    if progress_callback and saved_count % 10 == 0:
                        progress_callback(saved_count, total_frames)

                except Exception as e:
                    logger.error(f"Failed to save frame {idx}: {e}")

        logger.info(f"Wrote {saved_count} images to {self._output_dir}")

    def close(self) -> None:
        """Close writer."""
        self._output_dir = None
        self._frame_count = 0

    def cancel(self) -> None:
        """Cancel writing."""
        self._cancelled = True


class FrameWriterFactory:
    """Factory for creating appropriate frame writer."""
    
    @staticmethod
    def create_writer(
        output_path: Union[str, Path],
        codec_config: Optional[dict] = None,
    ) -> FrameWriter:
        """Create appropriate writer for output path.
        
        Args:
            output_path: Output file path
            codec_config: Optional codec configuration
            
        Returns:
            FrameWriter instance
        """
        output_path = Path(output_path)
        
        # Check if it's an image sequence output
        from core.codec_manager import get_codec_manager, CodecConfig
        
        codec_manager = get_codec_manager()
        if codec_config:
            codec_manager.set_config(CodecConfig.from_dict(codec_config))
        
        if codec_manager.is_image_output():
            # Get image format from codec config
            image_format = codec_config.get("image_format", "png") if codec_config else "png"
            quality = codec_config.get("image_quality", 95) if codec_config else 95
            return ImageSequenceWriter(
                image_format=image_format,
                quality=quality,
            )
        
        # Video output
        return VideoFrameWriter(codec_config=codec_config)
    
    @staticmethod
    def is_image_output(output_path: Union[str, Path]) -> bool:
        """Check if output path indicates image sequence.
        
        Args:
            output_path: Output path to check
            
        Returns:
            True if image sequence output
        """
        from core.codec_manager import get_codec_manager
        
        codec_manager = get_codec_manager()
        return codec_manager.is_image_output()
