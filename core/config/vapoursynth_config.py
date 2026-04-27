"""VapourSynth-specific configuration."""

from typing import Any, Dict

from core.config.base_config import BaseConfig
from core.paths import paths


VS_PORTABLE_DEFAULT = str(paths.vs_portable_dir)

DEFAULT_VAPOURSYNTH_SETTINGS = {
    "portable_path": VS_PORTABLE_DEFAULT,
    "num_threads": 4,
    "models_dir": str(paths.models_dir),
    "temp_dir": str(paths.temp_dir),
    "output_dir": str(paths.output_dir),
}


class VapourSynthConfig(BaseConfig):
    """Configuration for VapourSynth-related settings.
    
    Manages VapourSynth paths, threading, and directory locations.
    """

    def _load_defaults(self) -> None:
        """Load default VapourSynth settings."""
        self._settings = DEFAULT_VAPOURSYNTH_SETTINGS.copy()

    def get_portable_path(self) -> str:
        """Get VapourSynth portable path."""
        return self._settings.get("portable_path", VS_PORTABLE_DEFAULT)

    def set_portable_path(self, path: str) -> None:
        """Set VapourSynth portable path."""
        self._settings["portable_path"] = path
        self.save()

    def get_num_threads(self) -> int:
        """Get number of VapourSynth threads."""
        return self._settings.get("num_threads", 4)

    def set_num_threads(self, threads: int) -> None:
        """Set number of VapourSynth threads."""
        self._settings["num_threads"] = threads
        self.save()

    def get_models_dir(self) -> str:
        """Get models directory path."""
        return self._settings.get("models_dir", str(paths.models_dir))

    def set_models_dir(self, path: str) -> None:
        """Set models directory path."""
        self._settings["models_dir"] = path
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
            "portable_path": self.get_portable_path(),
            "models_dir": self.get_models_dir(),
            "temp_dir": self.get_temp_dir(),
            "output_dir": self.get_output_dir(),
        }
