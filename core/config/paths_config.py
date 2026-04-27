"""Paths-specific configuration.

This configuration domain handles all path-related settings including:
- Models directory for AI model storage
- FFmpeg directory for FFmpeg binaries
- VapourSynth portable path
- Temp directory for temporary files
- Output directory for processed videos
"""

from typing import Any, Dict

from core.config.base_config import BaseConfig
from core.paths import paths


DEFAULT_PATHS_SETTINGS: Dict[str, Any] = {
    "models_dir": str(paths.models_dir),
    "ffmpeg_dir": "",
    "vs_portable_path": str(paths.vs_portable_dir),
    "temp_dir": str(paths.temp_dir),
    "output_dir": str(paths.output_dir),
}


class PathsConfig(BaseConfig):
    """Configuration for path-related settings.

    Manages directories for models, FFmpeg, VapourSynth, temp, and output.
    """

    def _load_defaults(self) -> None:
        """Load default paths settings."""
        self._settings = DEFAULT_PATHS_SETTINGS.copy()

    def get_models_dir(self) -> str:
        """Get models directory path."""
        return self._settings.get("models_dir", str(paths.models_dir))

    def set_models_dir(self, path: str) -> None:
        """Set models directory path."""
        self._settings["models_dir"] = path
        self.save()

    def get_ffmpeg_dir(self) -> str:
        """Get FFmpeg directory path."""
        return self._settings.get("ffmpeg_dir", "")

    def set_ffmpeg_dir(self, path: str) -> None:
        """Set FFmpeg directory path."""
        self._settings["ffmpeg_dir"] = path
        self.save()

    def get_vs_portable_path(self) -> str:
        """Get VapourSynth portable path."""
        return self._settings.get("vs_portable_path", str(paths.vs_portable_dir))

    def set_vs_portable_path(self, path: str) -> None:
        """Set VapourSynth portable path."""
        self._settings["vs_portable_path"] = path
        self.save()

    def get_temp_dir(self) -> str:
        """Get temporary directory path."""
        return self._settings.get("temp_dir", str(paths.temp_dir))

    def set_temp_dir(self, path: str) -> None:
        """Set temporary directory path."""
        self._settings["temp_dir"] = path
        self.save()

    def get_output_dir(self) -> str:
        """Get output directory path."""
        return self._settings.get("output_dir", str(paths.output_dir))

    def set_output_dir(self, path: str) -> None:
        """Set output directory path."""
        self._settings["output_dir"] = path
        self.save()

    def get_all_paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return {
            "models_dir": self.get_models_dir(),
            "ffmpeg_dir": self.get_ffmpeg_dir(),
            "vs_portable_path": self.get_vs_portable_path(),
            "temp_dir": self.get_temp_dir(),
            "output_dir": self.get_output_dir(),
        }