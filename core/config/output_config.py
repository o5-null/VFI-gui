"""Output-specific configuration."""

from typing import Any, Dict

from core.config.base_config import BaseConfig


DEFAULT_OUTPUT_SETTINGS = {
    "codec": "hevc_nvenc",
    "quality": 22,
    "preset": "p4",
    "audio_copy": True,
    "pixel_format": "yuv420p",
    "color_range": "tv",
}


class OutputConfig(BaseConfig):
    """Configuration for output video settings.
    
    Manages codec, quality, preset, and audio settings.
    """

    def _load_defaults(self) -> None:
        """Load default output settings."""
        self._settings = DEFAULT_OUTPUT_SETTINGS.copy()

    def get_codec(self) -> str:
        """Get output codec."""
        return self._settings.get("codec", "hevc_nvenc")

    def set_codec(self, codec: str) -> None:
        """Set output codec."""
        self._settings["codec"] = codec
        self.save()

    def get_quality(self) -> int:
        """Get output quality (CRF for x264/x265, CQ for NVENC)."""
        return self._settings.get("quality", 22)

    def set_quality(self, quality: int) -> None:
        """Set output quality."""
        self._settings["quality"] = quality
        self.save()

    def get_preset(self) -> str:
        """Get encoding preset."""
        return self._settings.get("preset", "p4")

    def set_preset(self, preset: str) -> None:
        """Set encoding preset."""
        self._settings["preset"] = preset
        self.save()

    def is_audio_copy_enabled(self) -> bool:
        """Check if audio copy is enabled."""
        return self._settings.get("audio_copy", True)

    def set_audio_copy(self, enabled: bool) -> None:
        """Set audio copy enabled state."""
        self._settings["audio_copy"] = enabled
        self.save()

    def get_pixel_format(self) -> str:
        """Get output pixel format."""
        return self._settings.get("pixel_format", "yuv420p")

    def set_pixel_format(self, fmt: str) -> None:
        """Set output pixel format."""
        self._settings["pixel_format"] = fmt
        self.save()

    def get_ffmpeg_args(self) -> Dict[str, Any]:
        """Get FFmpeg arguments dictionary.
        
        Returns:
            Dictionary suitable for FFmpeg command construction
        """
        return {
            "codec": self.get_codec(),
            "quality": self.get_quality(),
            "preset": self.get_preset(),
            "pix_fmt": self.get_pixel_format(),
            "audio_copy": self.is_audio_copy_enabled(),
        }
