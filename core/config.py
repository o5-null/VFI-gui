"""JSON-based settings management for VFI-gui.

This module re-exports the new domain-specific configuration system
for backward compatibility. New code should import from core.config directly.

Example:
    # Old way (still works)
    from core.config import Config
    config = Config()
    
    # New recommended way
    from core.config import ConfigFacade
    config = ConfigFacade()
    pipeline_cfg = config.pipeline.get_interpolation_config()
"""

# Re-export all configuration classes from the new module
from core.config import (
    BaseConfig,
    PipelineConfig,
    UIConfig,
    NetworkConfig,
    VapourSynthConfig,
    OutputConfig,
    ConfigFacade,
    Config,  # Alias for backward compatibility
)

__all__ = [
    "BaseConfig",
    "PipelineConfig",
    "UIConfig",
    "NetworkConfig",
    "VapourSynthConfig",
    "OutputConfig",
    "ConfigFacade",
    "Config",  # Backward compatibility
]
