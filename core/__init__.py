"""VFI-gui core module exports."""

# Backward compatibility stub - defined FIRST to avoid circular imports
# This must be before any imports that might depend on it
class Processor:
    """Stub Processor for backward compatibility.

    .. deprecated:: Use TaskOrchestrator instead.
    """

    def __init__(self, config: object = None, backend_type: object = None) -> None:
        self._config: object = config
        self._backend_type: object = backend_type

    def cancel(self):
        pass


# Path manager - import first as other modules may depend on it
from core.paths import paths, PathManager

from core.config import Config, DEFAULT_SETTINGS
from core.config_provider import get_config, set_config, reset_config
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
from core.model_manager import ModelManager, EngineInfo, ModelTypeInfo, CheckpointInfo, ModelStatus, MODEL_DEFINITIONS
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
    RuntimeInfo,
    runtime_manager,
)
from core.config.runtime_config import RuntimeConfig, DEFAULT_RUNTIME_SETTINGS

# Unified device management
from core.device_type import DeviceType, RuntimeType, get_device_type_priority
from core.device_manager import (
    DeviceManager,
    device_manager,
    get_available_devices,
    get_best_device,
    is_cuda_available,
    is_rocm_available,
    is_xpu_available,
    resolve_device,
    get_torch_device,
)
from core.benchmark.device_detector import DeviceInfo, SystemInfo

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
    # Backend types and factory
    "BackendType",
    "BackendConfig",
    "ProcessingConfig",
    "ProcessingResult",
    "BaseBackend",
    "BackendFactory",
    # Backward compatibility
    "Processor",
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
    "RuntimeInfo",
    "runtime_manager",
    "RuntimeConfig",
    "DEFAULT_RUNTIME_SETTINGS",
    # Device management (unified)
    "DeviceType",
    "RuntimeType",  # Backward compatibility alias for DeviceType
    "get_device_type_priority",
    "DeviceManager",
    "device_manager",
    "DeviceInfo",
    "SystemInfo",
    "get_available_devices",
    "get_best_device",
    "is_cuda_available",
    "is_rocm_available",
    "is_xpu_available",
    "resolve_device",
    "get_torch_device",
    # Utils
    "natural_sort_key",
    "sort_files_naturally",
    "get_image_sequence_files",
    "detect_image_sequence_pattern",
    "is_image_file",
    "is_video_file",
    "get_media_type",
]
