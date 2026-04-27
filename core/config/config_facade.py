"""Configuration Facade - provides unified access to all domain configs.

This class maintains backward compatibility with the old Config class
while delegating to domain-specific configuration classes.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from core.paths import paths
from core.io import ExportImportManager

from core.config.pipeline_config import PipelineConfig
from core.config.ui_config import UIConfig
from core.config.network_config import NetworkConfig
from core.config.vapoursynth_config import VapourSynthConfig
from core.config.output_config import OutputConfig
from core.config.runtime_config import RuntimeConfig
from core.config.paths_config import PathsConfig
from core.config.performance_config import PerformanceConfig


class ConfigFacade:
    """Facade for accessing all domain-specific configurations.
    
    This class provides a unified interface to all configuration domains
    while maintaining backward compatibility with the legacy Config API.
    
    Usage:
        # New recommended way - access specific configs directly
        config = ConfigFacade()
        pipeline = config.pipeline.get_interpolation_config()
        
        # Legacy compatibility - still works
        config.get("pipeline.interpolation.model_type")
    """

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize all domain configurations.
        
        Args:
            config_dir: Directory for configuration files
        """
        self._config_dir = Path(config_dir) if config_dir else paths.config_dir
        self._config_dir.mkdir(parents=True, exist_ok=True)
        
        # Shared IO manager for efficiency
        self._io_manager = ExportImportManager()
        
        # Initialize domain-specific configs
        self.pipeline = PipelineConfig(
            str(self._config_dir / "pipeline.json"),
            self._io_manager,
        )
        self.ui = UIConfig(
            str(self._config_dir / "ui.json"),
            self._io_manager,
        )
        self.network = NetworkConfig(
            str(self._config_dir / "network.json"),
            self._io_manager,
        )
        self.vapoursynth = VapourSynthConfig(
            str(self._config_dir / "vapoursynth.json"),
            self._io_manager,
        )
        self.output = OutputConfig(
            str(self._config_dir / "output.json"),
            self._io_manager,
        )
        self.runtime = RuntimeConfig(
            str(self._config_dir / "runtime.json"),
            self._io_manager,
        )
        self.paths = PathsConfig(
            str(self._config_dir / "paths.json"),
            self._io_manager,
        )
        self.performance = PerformanceConfig(
            str(self._config_dir / "performance.json"),
            self._io_manager,
        )
        
        logger.debug("ConfigFacade initialized with all domain configs")

    # Legacy API compatibility methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (legacy API).
        
        Args:
            key: Dot-notation key with domain prefix
                 (e.g., "pipeline.interpolation.model_type")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        if not parts:
            return default
            
        domain = parts[0]
        subkey = ".".join(parts[1:]) if len(parts) > 1 else ""
        
        domain_configs = {
            "pipeline": self.pipeline,
            "ui": self.ui,
            "network": self.network,
            "vapoursynth": self.vapoursynth,
            "output": self.output,
            "runtime": self.runtime,
            "paths": self.paths,
            "performance": self.performance,
        }
        
        if domain in domain_configs:
            if subkey:
                return domain_configs[domain].get(subkey, default)
            else:
                return domain_configs[domain].get_all()
                
        return default

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """Set a configuration value using dot notation (legacy API).
        
        Args:
            key: Dot-notation key with domain prefix
            value: Value to set
            auto_save: Whether to save after setting
        """
        parts = key.split(".")
        if not parts:
            return
            
        domain = parts[0]
        subkey = ".".join(parts[1:]) if len(parts) > 1 else ""
        
        domain_configs = {
            "pipeline": self.pipeline,
            "ui": self.ui,
            "network": self.network,
            "vapoursynth": self.vapoursynth,
            "output": self.output,
            "runtime": self.runtime,
            "paths": self.paths,
            "performance": self.performance,
        }
        
        if domain in domain_configs and subkey:
            domain_configs[domain].set(subkey, value, auto_save)

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values (legacy API).
        
        Returns:
            Dictionary with all configuration domains
        """
        return {
            "pipeline": self.pipeline.get_all(),
            "ui": self.ui.get_all(),
            "network": self.network.get_all(),
            "vapoursynth": self.vapoursynth.get_all(),
            "output": self.output.get_all(),
            "runtime": self.runtime.get_all(),
            "paths": self.paths.get_all(),
            "performance": self.performance.get_all(),
        }

    def save(self) -> None:
        """Save all configurations (legacy API)."""
        self.pipeline.save()
        self.ui.save()
        self.network.save()
        self.vapoursynth.save()
        self.output.save()
        self.runtime.save()
        self.paths.save()
        self.performance.save()
        logger.debug("All configurations saved")

    def reset_to_defaults(self) -> None:
        """Reset all configurations to defaults (legacy API)."""
        self.pipeline.reset_to_defaults()
        self.ui.reset_to_defaults()
        self.network.reset_to_defaults()
        self.vapoursynth.reset_to_defaults()
        self.output.reset_to_defaults()
        self.runtime.reset_to_defaults()
        self.paths.reset_to_defaults()
        self.performance.reset_to_defaults()
        logger.info("All configurations reset to defaults")

    # Convenience methods for common operations
    def get_language(self) -> str:
        """Get UI language (convenience method)."""
        return self.ui.get_language()

    def set_language(self, language: str) -> None:
        """Set UI language (convenience method)."""
        self.ui.set_language(language)

    def get_proxy_config(self) -> dict:
        """Get proxy configuration (convenience method)."""
        return self.network.get_proxy_config()

    def get_vapoursynth_config(self) -> Dict[str, Any]:
        """Get VapourSynth configuration (convenience method)."""
        return {
            "portable_path": self.vapoursynth.get_portable_path(),
            "num_threads": self.vapoursynth.get_num_threads(),
            "models_dir": self.vapoursynth.get_models_dir(),
            "temp_dir": self.vapoursynth.get_temp_dir(),
            "output_dir": self.vapoursynth.get_output_dir(),
        }

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration (convenience method)."""
        return self.output.get_all()

    def set_output_config(self, config: Dict[str, Any]) -> None:
        """Set output configuration (convenience method)."""
        for key, value in config.items():
            self.output.set(key, value, auto_save=False)
        self.output.save()

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration (convenience method).
        
        Returns combined pipeline and output config for UI compatibility.
        """
        return {
            **self.pipeline.get_all(),
            "output": self.output.get_all(),
        }

    def set_pipeline_config(self, config: Dict[str, Any]) -> None:
        """Set pipeline configuration (convenience method)."""
        # Separate output config from pipeline config
        pipeline_keys = ["interpolation", "upscaling", "scene_detection"]
        for key, value in config.items():
            if key in pipeline_keys:
                self.pipeline.set(key, value, auto_save=False)
            elif key == "output":
                for out_key, out_value in value.items():
                    self.output.set(out_key, out_value, auto_save=False)
        self.pipeline.save()
        self.output.save()

    # Legacy compatibility property
    @property
    def settings(self) -> Dict[str, Any]:
        """Legacy settings property for backward compatibility.
        
        Returns a combined dictionary of all configuration domains.
        Note: Direct modification of this dictionary won't trigger saves.
              Use set() method for modifications.
        """
        return self.get_all()

    @settings.setter
    def settings(self, value: Dict[str, Any]) -> None:
        """Legacy settings setter for backward compatibility.
        
        Distributes settings to appropriate domain configs.
        """
        domain_configs = {
            "pipeline": self.pipeline,
            "ui": self.ui,
            "network": self.network,
            "vapoursynth": self.vapoursynth,
            "output": self.output,
            "runtime": self.runtime,
            "paths": self.paths,
            "performance": self.performance,
        }
        
        for key, config_value in value.items():
            if key in domain_configs:
                if isinstance(config_value, dict):
                    for sub_key, sub_value in config_value.items():
                        domain_configs[key].set(sub_key, sub_value, auto_save=False)
                else:
                    domain_configs[key]._settings = config_value
        
        self.save()


# Backward compatibility alias
Config = ConfigFacade
