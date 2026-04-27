"""VFI-gui core module exports."""

# Path manager - import first as other modules may depend on it
from core.paths import paths, PathManager

from core.config import Config, DEFAULT_SETTINGS
from core.config_provider import get_config, set_config, reset_config
from core.processor import (
    Processor,
    VideoProcessor,  # Backward compatibility alias
    SUPPORTED_INTERPOLATION_MODELS,
    register_backend,
)
from core.backends import (
    # Types
    BackendType,
    BackendConfig,
    ProcessingConfig,
    ProcessingResult,
    # Base classes
    BaseBackend,
    BackendFactory,
)
from core.pipeline import PipelineBuilder
from core.models import ModelManager, EngineInfo, ModelTypeInfo, CheckpointInfo, ModelStatus, MODEL_DEFINITIONS
from core.queue_manager import QueueManager, QueueItem, QueueItemStatus
from core.dependency_manager import DependencyManager, FFmpegManager, DependencyInfo
from core.codec_manager import (
    CodecManager,
    CodecConfig,
    CodecInfo,
    CodecType,
    CodecVendor,
    get_codec_manager,
)
from core.i18n import (
    I18NManager,
    init_i18n,
    get_i18n,
    tr,
    tr_n,
)
from core.logger import logger, setup_logger, get_logger
from core.runtime_manager import (
    RuntimeManager,
    RuntimeType,
    RuntimeInfo,
    runtime_manager,
)
from core.config.runtime_config import RuntimeConfig, DEFAULT_RUNTIME_SETTINGS

# Utils
from core.utils import (
    natural_sort_key,
    sort_files_naturally,
    get_image_sequence_files,
    detect_image_sequence_pattern,
    is_image_file,
    is_video_file,
    get_media_type,
)

__all__ = [
    # Paths
    "paths",
    "PathManager",
    # Config
    "Config",
    "DEFAULT_SETTINGS",
    "get_config",
    "set_config",
    "reset_config",
    # Processor (new unified interface)
    "Processor",
    "VideoProcessor",  # Backward compatibility
    "SUPPORTED_INTERPOLATION_MODELS",
    "register_backend",
    # Backend types and factory
    "BackendType",
    "BackendConfig",
    "ProcessingConfig",
    "ProcessingResult",
    "BaseBackend",
    "BackendFactory",
    # Pipeline
    "PipelineBuilder",
    # Models
    "ModelManager",
    "EngineInfo",
    "ModelTypeInfo",
    "CheckpointInfo",
    "ModelStatus",
    "MODEL_DEFINITIONS",
    # Queue
    "QueueManager",
    "QueueItem",
    "QueueItemStatus",
    # Dependency Manager
    "DependencyManager",
    "FFmpegManager",
    "DependencyInfo",
    # Codec Manager
    "CodecManager",
    "CodecConfig",
    "CodecInfo",
    "CodecType",
    "CodecVendor",
    "get_codec_manager",
    # i18n
    "I18NManager",
    "init_i18n",
    "get_i18n",
    "tr",
    "tr_n",
    # Logger
    "logger",
    "setup_logger",
    "get_logger",
    # Runtime
    "RuntimeManager",
    "RuntimeType",
    "RuntimeInfo",
    "runtime_manager",
    "RuntimeConfig",
    "DEFAULT_RUNTIME_SETTINGS",
    # Utils
    "natural_sort_key",
    "sort_files_naturally",
    "get_image_sequence_files",
    "detect_image_sequence_pattern",
    "is_image_file",
    "is_video_file",
    "get_media_type",
]