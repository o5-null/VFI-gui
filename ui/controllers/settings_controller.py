"""Settings Controller for VFI-gui.

Handles settings management operations, separating settings logic from MainWindow.
"""

from typing import Optional, Dict, Any, Callable
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget
from loguru import logger

from core import Config
from core.i18n import I18NManager, get_i18n


class SettingsController(QObject):
    """Controller for application settings management.
    
    Encapsulates settings logic that was previously in MainWindow,
    providing a clean interface for loading, saving, and applying settings.
    """
    
    # Signals
    settings_loaded = pyqtSignal()
    settings_saved = pyqtSignal()
    language_changed = pyqtSignal(str)  # new language code
    config_changed = pyqtSignal(str, Any)  # key, value
    
    def __init__(self, config: Optional[Config] = None, parent=None):
        super().__init__(parent)
        self._config = config or Config()
        self._i18n = get_i18n()
    
    def get_config(self) -> Config:
        """Get the configuration object."""
        return self._config
    
    def load_settings(self) -> None:
        """Load settings from file."""
        self._config.load()
        logger.info("Settings loaded")
        self.settings_loaded.emit()
    
    def save_settings(self) -> bool:
        """Save settings to file.
        
        Returns:
            True if saved successfully
        """
        try:
            self._config.save()
            logger.info("Settings saved")
            self.settings_saved.emit()
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key.
        
        Args:
            key: Dot-notation key (e.g., "pipeline.interpolation.model_type")
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        return self._config.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value.
        
        Args:
            key: Dot-notation key
            value: Value to set
        """
        self._config.set(key, value)
        self.config_changed.emit(key, value)
        logger.debug(f"Setting changed: {key} = {value}")
    
    def get_language(self) -> str:
        """Get current language code."""
        return self._config.get("ui.language", "")
    
    def set_language(self, language_code: str) -> bool:
        """Set application language.
        
        Args:
            language_code: Language code (e.g., "en", "zh_CN")
            
        Returns:
            True if language was changed
        """
        current = self.get_language()
        if current == language_code:
            return False
        
        self._config.set("ui.language", language_code)
        
        # Apply language change
        if self._i18n:
            self._i18n.set_language(language_code)
        
        logger.info(f"Language changed to: {language_code}")
        self.language_changed.emit(language_code)
        self.config_changed.emit("ui.language", language_code)
        return True
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages.
        
        Returns:
            Dictionary of language_code -> display_name
        """
        if self._i18n:
            languages = {}
            for code in self._i18n.get_available_languages():
                display = self._i18n.get_language_display_name(code)
                languages[code] = display
            return languages
        return {}
    
    def get_window_geometry(self) -> Optional[bytes]:
        """Get saved window geometry."""
        geometry = self._config.get("window.geometry")
        if geometry:
            try:
                return bytes.fromhex(geometry)
            except ValueError:
                logger.warning("Invalid window geometry in settings")
        return None
    
    def set_window_geometry(self, geometry: bytes) -> None:
        """Save window geometry."""
        self._config.set("window.geometry", geometry.hex())
    
    def get_models_dir(self) -> str:
        """Get models directory path."""
        return self._config.get("paths.models_dir", "")
    
    def set_models_dir(self, path: str) -> None:
        """Set models directory path."""
        self._config.set("paths.models_dir", path)
    
    def get_output_dir(self) -> str:
        """Get output directory path."""
        return self._config.get("paths.output_dir", "")
    
    def set_output_dir(self, path: str) -> None:
        """Set output directory path."""
        self._config.set("paths.output_dir", path)
    
    def apply_language_preference(self) -> None:
        """Apply saved language preference."""
        language = self.get_language()
        if language and self._i18n:
            self._i18n.set_language(language)
            logger.info(f"Applied language preference: {language}")

    def get_proxy_config(self) -> Dict[str, str]:
        """Get proxy configuration.
        
        Returns:
            Dictionary with http_proxy, https_proxy, no_proxy
        """
        return self._config.get("network.proxy", {
            "http_proxy": "",
            "https_proxy": "",
            "no_proxy": "",
        })

    def set_proxy_config(self, http_proxy: str = "", https_proxy: str = "", no_proxy: str = "") -> None:
        """Set proxy configuration.
        
        Args:
            http_proxy: HTTP proxy URL (e.g., "http://127.0.0.1:7890")
            https_proxy: HTTPS proxy URL (e.g., "http://127.0.0.1:7890")
            no_proxy: Comma-separated list of hosts to bypass proxy
        """
        self._config.set("network.proxy", {
            "http_proxy": http_proxy,
            "https_proxy": https_proxy,
            "no_proxy": no_proxy,
        })
        logger.info(f"Proxy configuration updated")

    def get_network_timeout(self) -> int:
        """Get network timeout in seconds."""
        return self._config.get("network.timeout", 300)

    def set_network_timeout(self, timeout: int) -> None:
        """Set network timeout in seconds."""
        self._config.set("network.timeout", timeout)

    def get_max_retries(self) -> int:
        """Get max retry attempts for failed downloads."""
        return self._config.get("network.max_retries", 3)

    def set_max_retries(self, retries: int) -> None:
        """Set max retry attempts for failed downloads."""
        self._config.set("network.max_retries", retries)
