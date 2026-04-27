"""UI-specific configuration."""

from core.config.base_config import BaseConfig


DEFAULT_UI_SETTINGS = {
    "language": "",
    "theme": "dark",
    "window_geometry": None,
    "recent_files": [],
}


class UIConfig(BaseConfig):
    """Configuration for UI-related settings.
    
    Manages language, theme, window state, and recent files.
    """

    def _load_defaults(self) -> None:
        """Load default UI settings."""
        self._settings = DEFAULT_UI_SETTINGS.copy()

    def get_language(self) -> str:
        """Get UI language setting.
        
        Returns:
            Language code or empty string for auto-detect
        """
        return self._settings.get("language", "")

    def set_language(self, language: str) -> None:
        """Set UI language.
        
        Args:
            language: Language code or empty for auto-detect
        """
        self._settings["language"] = language
        self.save()

    def get_theme(self) -> str:
        """Get UI theme."""
        return self._settings.get("theme", "dark")

    def set_theme(self, theme: str) -> None:
        """Set UI theme."""
        self._settings["theme"] = theme
        self.save()

    def get_recent_files(self) -> list:
        """Get list of recent files."""
        return self._settings.get("recent_files", []).copy()

    def add_recent_file(self, file_path: str, max_files: int = 10) -> None:
        """Add a file to recent files list.
        
        Args:
            file_path: Path to add
            max_files: Maximum number of files to keep
        """
        recent = self.get_recent_files()
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        self._settings["recent_files"] = recent[:max_files]
        self.save()
