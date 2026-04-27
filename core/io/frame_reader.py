"""Video frame reader component.

Handles all video input operations, reading frames into memory
for processing by backend components.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Callable, List, Union
import warnings

import numpy as np
import torch
from loguru import logger

from core.io.frame_data import FrameData, FrameBatch, VideoMetadata, VideoFrameSequence, FrameFormat


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
    ) -> Iterator[FrameData]:
        """Read frames as iterator (memory efficient).
        
        Args:
            source: Video file path or image sequence pattern
            
        Yields:
            FrameData objects
        """
        pass
    
    @abstractmethod
    def get_metadata(self, source: Union[str, Path]) -> VideoMetadata:
        """Get video metadata without reading frames.
        
        Args:
            source: Video file path
            
        Returns:
            VideoMetadata
        """
        pass


class VideoFrameReader(FrameReader):
    """Video file frame reader using OpenCV."""
    
    def __init__(
        self,
        target_format: FrameFormat = FrameFormat.RGB_FLOAT,
        device: str = "cpu",
    ):
        """Initialize video frame reader.
        
        Args:
            target_format: Target frame format
            device: Target device for tensors
        """
        self.target_format = target_format
        self.device = device
    
    def read_frames(
        self,
        source: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VideoFrameSequence:
        """Read all frames from video file."""
        import cv2
        
        source_path = Path(source)
        
        # Open video
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source_path}")
        
        # Get metadata
        metadata = self._extract_metadata(cap)
        
        # Read all frames
        frames: List[FrameData] = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to target format
                frame_data = self._convert_frame(frame, frame_idx)
                frames.append(frame_data)
                
                frame_idx += 1
                if progress_callback and frame_idx % 10 == 0:
                    progress_callback(frame_idx, metadata.total_frames)
        finally:
            cap.release()
        
        logger.info(f"Read {len(frames)} frames from {source_path}")
        
        return VideoFrameSequence(
            frames=frames,
            metadata=metadata,
            source_path=source_path,
        )
    
    def read_frames_iter(
        self,
        source: Union[str, Path],
    ) -> Iterator[FrameData]:
        """Read frames as iterator."""
        import cv2
        
        source_path = Path(source)
        
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source_path}")
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_data = self._convert_frame(frame, frame_idx)
                yield frame_data
                frame_idx += 1
        finally:
            cap.release()
    
    def get_metadata(self, source: Union[str, Path]) -> VideoMetadata:
        """Get video metadata."""
        import cv2
        
        source_path = Path(source)
        
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source_path}")
        
        try:
            return self._extract_metadata(cap)
        finally:
            cap.release()
    
    def _extract_metadata(self, cap) -> VideoMetadata:
        """Extract metadata from OpenCV capture."""
        import cv2
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0.0
        
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
        )
    
    def _convert_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameData:
        """Convert OpenCV BGR frame to target format."""
        import cv2
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to target format
        if self.target_format == FrameFormat.RGB:
            data = frame_rgb
        elif self.target_format == FrameFormat.RGB_FLOAT:
            data = frame_rgb.astype(np.float32) / 255.0
        elif self.target_format == FrameFormat.TENSOR_NHWC:
            data = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).to(self.device)
        else:
            # Default to RGB float
            data = frame_rgb.astype(np.float32) / 255.0
        
        return FrameData(
            data=data,
            frame_idx=frame_idx,
            format=self.target_format,
            metadata={"original_format": "bgr"},
        )


class ImageSequenceReader(FrameReader):
    """Image sequence reader."""
    
    def __init__(
        self,
        target_format: FrameFormat = FrameFormat.RGB_FLOAT,
        device: str = "cpu",
        pattern: Optional[str] = None,
    ):
        """Initialize image sequence reader.
        
        Args:
            target_format: Target frame format
            device: Target device for tensors
            pattern: Optional glob pattern for filtering files
        """
        self.target_format = target_format
        self.device = device
        self.pattern = pattern
    
    def read_frames(
        self,
        source: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VideoFrameSequence:
        """Read image sequence from directory."""
        from PIL import Image
        from core.utils.file_utils import sort_files_naturally
        
        source_path = Path(source)
        
        # Get image files
        if source_path.is_dir():
            if self.pattern:
                image_files = sorted(source_path.glob(self.pattern))
            else:
                from core.utils.file_utils import IMAGE_EXTENSIONS
                image_files = [
                    f for f in source_path.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ]
                image_files = sort_files_naturally(image_files)
        else:
            # Single file or pattern
            parent = source_path.parent
            pattern = source_path.name
            image_files = sorted(parent.glob(pattern))
        
        if not image_files:
            raise RuntimeError(f"No images found at: {source}")
        
        # Read first image to get dimensions
        first_img = Image.open(image_files[0])
        width, height = first_img.size
        
        # Create metadata (fps unknown for image sequences)
        metadata = VideoMetadata(
            width=width,
            height=height,
            fps=30.0,  # Default assumption
            total_frames=len(image_files),
        )
        
        # Read all frames
        frames: List[FrameData] = []
        
        for frame_idx, img_path in enumerate(image_files):
            frame_data = self._load_image(img_path, frame_idx)
            frames.append(frame_data)
            
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, len(image_files))
        
        logger.info(f"Read {len(frames)} images from {source_path}")
        
        return VideoFrameSequence(
            frames=frames,
            metadata=metadata,
            source_path=source_path if source_path.is_dir() else source_path.parent,
        )
    
    def read_frames_iter(
        self,
        source: Union[str, Path],
    ) -> Iterator[FrameData]:
        """Read images as iterator."""
        from PIL import Image
        from core.utils.file_utils import sort_files_naturally
        
        source_path = Path(source)
        
        # Get image files
        if source_path.is_dir():
            if self.pattern:
                image_files = sorted(source_path.glob(self.pattern))
            else:
                from core.utils.file_utils import IMAGE_EXTENSIONS
                image_files = [
                    f for f in source_path.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ]
                image_files = sort_files_naturally(image_files)
        else:
            parent = source_path.parent
            pattern = source_path.name
            image_files = sorted(parent.glob(pattern))
        
        for frame_idx, img_path in enumerate(image_files):
            yield self._load_image(img_path, frame_idx)
    
    def get_metadata(self, source: Union[str, Path]) -> VideoMetadata:
        """Get image sequence metadata."""
        from PIL import Image
        
        source_path = Path(source)
        
        # Get image files
        if source_path.is_dir():
            if self.pattern:
                image_files = list(source_path.glob(self.pattern))
            else:
                from core.utils.file_utils import IMAGE_EXTENSIONS
                image_files = [
                    f for f in source_path.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ]
        else:
            parent = source_path.parent
            pattern = source_path.name
            image_files = list(parent.glob(pattern))
        
        if not image_files:
            raise RuntimeError(f"No images found at: {source}")
        
        # Get dimensions from first image
        first_img = Image.open(image_files[0])
        width, height = first_img.size
        
        return VideoMetadata(
            width=width,
            height=height,
            fps=30.0,
            total_frames=len(image_files),
        )
    
    def _load_image(self, img_path: Path, frame_idx: int) -> FrameData:
        """Load single image."""
        from PIL import Image
        
        img = Image.open(img_path).convert("RGB")
        frame_rgb = np.array(img)
        
        # Convert to target format
        if self.target_format == FrameFormat.RGB:
            data = frame_rgb
        elif self.target_format == FrameFormat.RGB_FLOAT:
            data = frame_rgb.astype(np.float32) / 255.0
        elif self.target_format == FrameFormat.TENSOR_NHWC:
            data = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).to(self.device)
        else:
            data = frame_rgb.astype(np.float32) / 255.0
        
        return FrameData(
            data=data,
            frame_idx=frame_idx,
            format=self.target_format,
            metadata={"source_file": str(img_path)},
        )


class FrameReaderFactory:
    """Factory for creating appropriate frame reader."""
    
    @staticmethod
    def create_reader(
        source: Union[str, Path],
        target_format: FrameFormat = FrameFormat.RGB_FLOAT,
        device: str = "cpu",
    ) -> FrameReader:
        """Create appropriate reader for source.
        
        Args:
            source: Video file or image sequence path
            target_format: Target frame format
            device: Target device
            
        Returns:
            FrameReader instance
        """
        source_path = Path(source)
        
        # Check if it's a video file
        from core.utils.file_utils import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS
        
        if source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTENSIONS:
            return VideoFrameReader(target_format, device)
        
        # Check if it's an image sequence
        if source_path.is_dir():
            return ImageSequenceReader(target_format, device)
        
        if source_path.suffix.lower() in IMAGE_EXTENSIONS:
            return ImageSequenceReader(target_format, device)
        
        # Default to video reader
        return VideoFrameReader(target_format, device)
