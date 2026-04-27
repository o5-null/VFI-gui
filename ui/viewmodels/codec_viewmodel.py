"""Codec ViewModel for VFI-gui.

Acts as intermediary between CodecSettingsWidget and CodecManager,
providing a clean interface for UI-codec operations.
"""

from typing import Dict, Any, Optional, List
from PyQt6.QtCore import QObject, pyqtSignal

from core.codec_manager import CodecManager, CodecConfig, CodecType


class CodecViewModel(QObject):
    """ViewModel for codec configuration management.
    
    Decouples UI widgets from CodecManager implementation details.
    """
    
    # Signals for UI updates
    codec_changed = pyqtSignal(str)  # codec name
    config_changed = pyqtSignal()  # any config change
    quality_changed = pyqtSignal(int)  # quality value
    preset_changed = pyqtSignal(str)  # preset name
    
    def __init__(self, codec_manager: Optional[CodecManager] = None, parent=None):
        super().__init__(parent)
        self._codec_manager = codec_manager or CodecManager()
        self._current_config = CodecConfig()
        self._current_codec = "hevc_nvenc"
    
    def get_available_codecs(self) -> List[str]:
        """Get list of available codec names."""
        return self._codec_manager.get_available_codecs()
    
    def get_codec_info(self, codec_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific codec."""
        return self._codec_manager.get_codec_info(codec_name)
    
    def get_current_codec(self) -> str:
        """Get currently selected codec."""
        return self._current_codec
    
    def set_codec(self, codec_name: str) -> bool:
        """Set the current codec.
        
        Args:
            codec_name: Name of the codec to select
            
        Returns:
            True if codec was changed, False if invalid
        """
        if codec_name not in self.get_available_codecs():
            return False
        
        self._current_codec = codec_name
        self._current_config.codec = codec_name
        self.codec_changed.emit(codec_name)
        self.config_changed.emit()
        return True
    
    def get_quality_range(self) -> tuple:
        """Get valid quality range for current codec."""
        info = self.get_codec_info(self._current_codec)
        if info:
            return (info.get("min_quality", 0), info.get("max_quality", 51))
        return (0, 51)
    
    def get_quality(self) -> int:
        """Get current quality setting."""
        return self._current_config.quality
    
    def set_quality(self, quality: int) -> None:
        """Set quality value."""
        min_q, max_q = self.get_quality_range()
        self._current_config.quality = max(min_q, min(max_q, quality))
        self.quality_changed.emit(self._current_config.quality)
        self.config_changed.emit()
    
    def get_available_presets(self) -> List[str]:
        """Get available presets for current codec."""
        info = self.get_codec_info(self._current_codec)
        if info:
            return info.get("presets", [])
        return []
    
    def get_preset(self) -> str:
        """Get current preset."""
        return self._current_config.preset
    
    def set_preset(self, preset: str) -> None:
        """Set preset value."""
        self._current_config.preset = preset
        self.preset_changed.emit(preset)
        self.config_changed.emit()
    
    def get_config(self) -> CodecConfig:
        """Get current codec configuration."""
        return self._current_config
    
    def set_config(self, config: CodecConfig) -> None:
        """Set codec configuration from external source."""
        self._current_config = config
        self._current_codec = config.codec
        self.codec_changed.emit(config.codec)
        self.config_changed.emit()
    
    def get_ffmpeg_args(self) -> List[str]:
        """Get FFmpeg arguments for current configuration."""
        return self._codec_manager.build_ffmpeg_args(self._current_config)
    
    def validate_config(self) -> tuple[bool, str]:
        """Validate current configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self._codec_manager.validate_config(self._current_config)
    
    def load_from_dict(self, settings: Dict[str, Any]) -> None:
        """Load configuration from settings dictionary."""
        self._current_config = CodecConfig.from_dict(settings)
        self._current_codec = self._current_config.codec
        self.config_changed.emit()
    
    def save_to_dict(self) -> Dict[str, Any]:
        """Save configuration to dictionary."""
        return self._current_config.to_dict()
