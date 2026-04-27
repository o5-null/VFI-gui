"""Configuration module with domain-specific config classes."""

from core.config.base_config import BaseConfig
from core.config.pipeline_config import PipelineConfig, DEFAULT_PIPELINE_SETTINGS
from core.config.ui_config import UIConfig, DEFAULT_UI_SETTINGS
from core.config.network_config import NetworkConfig, DEFAULT_NETWORK_SETTINGS
from core.config.vapoursynth_config import VapourSynthConfig, DEFAULT_VAPOURSYNTH_SETTINGS
from core.config.output_config import OutputConfig, DEFAULT_OUTPUT_SETTINGS
from core.config.runtime_config import RuntimeConfig, DEFAULT_RUNTIME_SETTINGS
from core.config.paths_config import PathsConfig, DEFAULT_PATHS_SETTINGS
from core.config.performance_config import PerformanceConfig, DEFAULT_PERFORMANCE_SETTINGS
from core.config.config_facade import ConfigFacade, Config

# Legacy compatibility - combine all defaults
DEFAULT_SETTINGS = {
    "pipeline": DEFAULT_PIPELINE_SETTINGS,
    "ui": DEFAULT_UI_SETTINGS,
    "network": DEFAULT_NETWORK_SETTINGS,
    "vapoursynth": DEFAULT_VAPOURSYNTH_SETTINGS,
    "output": DEFAULT_OUTPUT_SETTINGS,
    "runtime": DEFAULT_RUNTIME_SETTINGS,
    "paths": DEFAULT_PATHS_SETTINGS,
    "performance": DEFAULT_PERFORMANCE_SETTINGS,
}

__all__ = [
    "BaseConfig",
    "PipelineConfig",
    "UIConfig",
    "NetworkConfig",
    "VapourSynthConfig",
    "OutputConfig",
    "RuntimeConfig",
    "PathsConfig",
    "PerformanceConfig",
    "ConfigFacade",
    "Config",
    "DEFAULT_SETTINGS",
    "DEFAULT_PIPELINE_SETTINGS",
    "DEFAULT_UI_SETTINGS",
    "DEFAULT_NETWORK_SETTINGS",
    "DEFAULT_VAPOURSYNTH_SETTINGS",
    "DEFAULT_OUTPUT_SETTINGS",
    "DEFAULT_RUNTIME_SETTINGS",
    "DEFAULT_PATHS_SETTINGS",
    "DEFAULT_PERFORMANCE_SETTINGS",
]
