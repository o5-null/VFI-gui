"""SettingsController - handles UI settings persistence.

Stateless controller for window geometry and UI preferences.
"""

from typing import Optional

from loguru import logger

from core.config.config_facade import ConfigFacade


class SettingsController:
    """Stateless controller for UI settings.
    
    This controller handles:
    - Window geometry persistence
    - UI preferences (language, theme)
    
    Controllers do NOT hold state - they delegate to ConfigFacade.
    """
    
    def __init__(self, config: ConfigFacade):
        """Initialize SettingsController.
        
        Args:
            config: ConfigFacade instance
        """
        self._config = config
    
    def save_window_geometry(self, geometry: bytes) -> None:
        """Save window geometry to config.
        
        Args:
            geometry: Serialized window geometry (from QWidget.saveGeometry())
        """
        # Store as hex string for JSON compatibility
        geometry_hex = geometry.hex()
        self._config.ui.set("window_geometry", geometry_hex)
        logger.debug("Window geometry saved")
    
    def load_window_geometry(self) -> Optional[bytes]:
        """Load window geometry from config.
        
        Returns:
            Geometry bytes or None if not stored
        """
        geometry_hex = self._config.ui.get("window_geometry")
        
        if geometry_hex:
            try:
                geometry = bytes.fromhex(geometry_hex)
                logger.debug("Window geometry loaded")
                return geometry
            except ValueError:
                logger.warning("Invalid geometry data in config")
                return None
        
        return None
    
    def save_window_state(self, state: bytes) -> None:
        """Save window state (toolbars, docks) to config.
        
        Args:
            state: Serialized window state (from QMainWindow.saveState())
        """
        state_hex = state.hex()
        self._config.ui.set("window_state", state_hex)
        logger.debug("Window state saved")
    
    def load_window_state(self) -> Optional[bytes]:
        """Load window state from config.
        
        Returns:
            State bytes or None if not stored
        """
        state_hex = self._config.ui.get("window_state")
        
        if state_hex:
            try:
                state = bytes.fromhex(state_hex)
                logger.debug("Window state loaded")
                return state
            except ValueError:
                logger.warning("Invalid state data in config")
                return None
        
        return None
    
    def set_language(self, language: str) -> None:
        """Set UI language.
        
        Args:
            language: Language code (e.g., "en", "zh_CN")
        """
        self._config.set_language(language)
        logger.info(f"Language set to: {language}")
    
    def get_language(self) -> str:
        """Get current UI language.
        
        Returns:
            Language code
        """
        return self._config.get_language()
    
    def set_theme(self, theme: str) -> None:
        """Set UI theme.
        
        Args:
            theme: Theme name (e.g., "dark", "light")
        """
        self._config.ui.set("theme", theme)
        logger.info(f"Theme set to: {theme}")
    
    def get_theme(self) -> str:
        """Get current UI theme.
        
        Returns:
            Theme name
        """
        return self._config.ui.get("theme", "dark")
    
    def set_last_open_dir(self, directory: str) -> None:
        """Save last opened directory for file dialogs.
        
        Args:
            directory: Directory path
        """
        self._config.ui.set("last_open_dir", directory)
    
    def get_last_open_dir(self) -> str:
        """Get last opened directory.
        
        Returns:
            Directory path or empty string
        """
        return self._config.ui.get("last_open_dir", "")
    
    def set_last_save_dir(self, directory: str) -> None:
        """Save last save directory for file dialogs.
        
        Args:
            directory: Directory path
        """
        self._config.ui.set("last_save_dir", directory)
    
    def get_last_save_dir(self) -> str:
        """Get last save directory.
        
        Returns:
            Directory path or empty string
        """
        return self._config.ui.get("last_save_dir", "")
    
    def reset_ui_settings(self) -> None:
        """Reset UI settings to defaults."""
        self._config.ui.reset_to_defaults()
        logger.info("UI settings reset to defaults")


__all__ = ["SettingsController"]