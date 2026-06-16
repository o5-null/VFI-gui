"""IO module for VFI-gui.

Provides centralized data export/import functionality with support for:
- JSON, YAML, TOML formats
- Async IO operations
- Batch processing
- Data validation and transformation
- Video frame reading/writing (PyAV primary, OpenCV/PIL fallback)
"""

from core.io.export_import_manager import ExportImportManager, ExportOptions, ExportFormat
from core.io.serializers import JsonSerializer, YamlSerializer, TomlSerializer
from core.io.async_io import AsyncFileHandler
from core.io.data_validator import DataValidator

# Frame data and video IO
from core.io.frame_data import (
    FrameData,
    FrameBatch,
    ProcessedFrameData,
    VideoMetadata,
    VideoFrameSequence,
    FrameFormat,
)
from core.io.frame_reader import (
    FrameReader,
    PyAVVideoReader,
    PyAVImageReader,
    FrameReaderFactory,
)
from core.io.frame_writer import (
    FrameWriter,
    PyAVVideoWriter,
    PyAVImageWriter,
    FrameWriterFactory,
)
from core.io.streaming_reader import StreamingFramePairReader
from core.io.frame_cache import FrameCache
from core.io.ordered_buffer import OrderedResultBuffer
from core.io.frame_lifecycle import FrameLifecycle

__all__ = [
    # Config/Data IO
    "ExportImportManager",
    "ExportOptions",
    "ExportFormat",
    "JsonSerializer",
    "YamlSerializer",
    "TomlSerializer",
    "AsyncFileHandler",
    "DataValidator",
    # Frame Data
    "FrameData",
    "FrameBatch",
    "ProcessedFrameData",
    "VideoMetadata",
    "VideoFrameSequence",
    "FrameFormat",
    # Frame Readers (PyAV primary)
    "PyAVVideoReader",
    "PyAVImageReader",
    # Frame Readers (base)
    "FrameReader",
    "FrameReaderFactory",
    # Frame Writers (PyAV primary)
    "PyAVVideoWriter",
    "PyAVImageWriter",
    # Frame Writers (base)
    "FrameWriter",
    "FrameWriterFactory",
    # Streaming Reader
    "StreamingFramePairReader",
    # Frame Cache
    "FrameCache",
    # Ordered Buffer (parallel results, sequential write)
    "OrderedResultBuffer",
    # Frame Lifecycle (write-once, consumer tracking)
    "FrameLifecycle",
]
