"""Pipeline-specific configuration."""

from typing import Any, Dict

from core.config.base_config import BaseConfig


DEFAULT_PIPELINE_SETTINGS = {
    "interpolation": {
        "enabled": False,
        "model_type": "rife",
        "model_version": "4.9",
        "multi": 2,
        "scale": 1.0,
        "scene_change": False,
    },
    "upscaling": {
        "enabled": False,
        "engine": "",
        "tile_size": 0,
        "overlap": 0,
        "num_streams": 3,
    },
    "scene_detection": {
        "enabled": False,
        "model": 12,
        "threshold": 0.5,
        "fp16": True,
    },
}


class PipelineConfig(BaseConfig):
    """Configuration for video processing pipeline settings.
    
    Manages interpolation, upscaling, and scene detection settings.
    """

    def _load_defaults(self) -> None:
        """Load default pipeline settings."""
        self._settings = DEFAULT_PIPELINE_SETTINGS.copy()

    def get_interpolation_config(self) -> Dict[str, Any]:
        """Get interpolation configuration."""
        return self._settings.get("interpolation", DEFAULT_PIPELINE_SETTINGS["interpolation"]).copy()

    def set_interpolation_config(self, config: Dict[str, Any]) -> None:
        """Set interpolation configuration."""
        self._settings["interpolation"] = config
        self.save()

    def get_upscaling_config(self) -> Dict[str, Any]:
        """Get upscaling configuration."""
        return self._settings.get("upscaling", DEFAULT_PIPELINE_SETTINGS["upscaling"]).copy()

    def set_upscaling_config(self, config: Dict[str, Any]) -> None:
        """Set upscaling configuration."""
        self._settings["upscaling"] = config
        self.save()

    def get_scene_detection_config(self) -> Dict[str, Any]:
        """Get scene detection configuration."""
        return self._settings.get("scene_detection", DEFAULT_PIPELINE_SETTINGS["scene_detection"]).copy()

    def set_scene_detection_config(self, config: Dict[str, Any]) -> None:
        """Set scene detection configuration."""
        self._settings["scene_detection"] = config
        self.save()

    def is_interpolation_enabled(self) -> bool:
        """Check if interpolation is enabled."""
        return self.get("interpolation.enabled", False)

    def is_upscaling_enabled(self) -> bool:
        """Check if upscaling is enabled."""
        return self.get("upscaling.enabled", False)

    def is_scene_detection_enabled(self) -> bool:
        """Check if scene detection is enabled."""
        return self.get("scene_detection.enabled", False)
