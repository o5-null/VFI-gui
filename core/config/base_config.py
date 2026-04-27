"""Base configuration class with common functionality."""

from pathlib import Path
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from loguru import logger

from core.io import ExportImportManager, ExportOptions


class BaseConfig(ABC):
    """Base class for domain-specific configurations.
    
    Provides common functionality for loading, saving, and accessing
    configuration values with dot notation support.
    """

    def __init__(
        self,
        config_path: str,
        io_manager: Optional[ExportImportManager] = None,
    ):
        self.config_path = config_path
        self._io_manager = io_manager or ExportImportManager()
        self._save_options = ExportOptions(
            validate=True,
            debounce_delay=0.5,
        )
        self._settings: Dict[str, Any] = {}
        self._load_defaults()
        self.load()

    @abstractmethod
    def _load_defaults(self) -> None:
        """Load default settings for this configuration domain."""
        pass

    def load(self) -> None:
        """Load settings from JSON file."""
        # Ensure config directory exists
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = self._io_manager.import_config(self.config_path, validate=True)
        if data:
            self._merge_settings(data)
            logger.debug(f"{self.__class__.__name__} loaded from: {self.config_path}")
        else:
            logger.debug(f"Using default {self.__class__.__name__} settings, creating config file")
            # Save default settings immediately to create the file
            self.save(immediate=True)

    def _merge_settings(self, loaded: Dict[str, Any]) -> None:
        """Merge loaded settings with defaults."""
        for key, value in loaded.items():
            if key in self._settings and isinstance(value, dict):
                self._settings[key].update(value)
            else:
                self._settings[key] = value

    def save(self, immediate: bool = False) -> None:
        """Save settings to JSON file.
        
        Args:
            immediate: If True, save immediately without debouncing
        """
        options = self._save_options
        if immediate:
            from core.io import ExportOptions
            options = ExportOptions(
                validate=True,
                debounce_delay=0.0,  # No debounce for immediate save
            )
        
        success = self._io_manager.export_config(
            self._settings,
            self.config_path,
            options,
        )
        if success:
            if immediate:
                logger.debug(f"{self.__class__.__name__} saved immediately")
            else:
                logger.debug(f"{self.__class__.__name__} save scheduled")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Dot-notation key (e.g., "interpolation.model_type")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """Set a configuration value using dot notation.
        
        Args:
            key: Dot-notation key
            value: Value to set
            auto_save: Whether to save after setting
        """
        keys = key.split(".")
        target = self._settings
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
            
        target[keys[-1]] = value
        
        if auto_save:
            self.save()

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._settings.copy()

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._settings = {}
        self._load_defaults()
        self.save()
        logger.info(f"{self.__class__.__name__} reset to defaults")
