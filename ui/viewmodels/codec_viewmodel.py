"""CodecViewModel for output codec configuration.

Wraps OutputConfig and CodecManager to provide Qt-signals for UI binding.
Supports explicit persist() for configuration saving.
"""

from typing import Any, Dict, List

from PyQt6.QtCore import QObject, pyqtSignal

from core.config.config_facade import ConfigFacade
from core.codec_manager import get_codec_manager, CodecInfo


class CodecViewModel(QObject):
    """ViewModel for codec/output configuration.
    
    Signals:
        codec_changed: Emitted when codec selection changes
        quality_changed: Emitted when quality value changes (int)
        preset_changed: Emitted when preset selection changes
        audio_copy_changed: Emitted when audio copy mode changes (bool)
        available_codecs_changed: Emitted when available codecs list changes
    
    Properties:
        codec: Current codec ID string
        quality: Current quality value (int)
        preset: Current preset string
        audio_copy: Whether to copy audio stream
        available_codecs: List of available codec IDs
    
    Methods:
        persist(): Save configuration to ConfigFacade
        to_config(): Export as ProcessingConfig.output compatible dict
    """
    
    # Signals
    codec_changed = pyqtSignal(str)
    quality_changed = pyqtSignal(int)
    preset_changed = pyqtSignal(str)
    audio_copy_changed = pyqtSignal(bool)
    available_codecs_changed = pyqtSignal(list)
    
    def __init__(
        self,
        config: ConfigFacade,
        codec_manager=None,
        parent=None,
    ):
        """Initialize CodecViewModel.
        
        Args:
            config: ConfigFacade instance for persistence
            codec_manager: CodecManager instance (optional, uses global if None)
            parent: Parent QObject
        """
        super().__init__(parent)
        self._config = config
        self._codec_manager = codec_manager or get_codec_manager()
        
        # Private fields (loaded from config)
        self._codec: str = ""
        self._quality: int = 0
        self._preset: str = ""
        self._audio_copy: bool = True
        self._available_codecs: List[str] = []
        
        # Load initial state from config
        self._load_from_config()
    
    def _load_from_config(self) -> None:
        """Load configuration values from ConfigFacade."""
        output_config = self._config.output
        
        self._codec = output_config.get_codec()
        self._quality = output_config.get_quality()
        self._preset = output_config.get_preset()
        self._audio_copy = output_config.is_audio_copy_enabled()
        
        # Get available codecs from CodecManager
        all_codecs = self._codec_manager.get_all_codecs()
        self._available_codecs = list(all_codecs.keys())
    
    # ====================
    # Properties
    # ====================
    
    @property
    def codec(self) -> str:
        """Get current codec ID."""
        return self._codec
    
    @property
    def quality(self) -> int:
        """Get current quality value."""
        return self._quality
    
    @property
    def preset(self) -> str:
        """Get current preset."""
        return self._preset
    
    @property
    def audio_copy(self) -> bool:
        """Get audio copy mode."""
        return self._audio_copy
    
    @property
    def available_codecs(self) -> List[str]:
        """Get list of available codec IDs."""
        return self._available_codecs.copy()
    
    # ====================
    # Setters (emit signals, no auto-persist)
    # ====================
    
    def set_codec(self, codec: str) -> None:
        """Set codec selection.
        
        Args:
            codec: Codec ID string
        """
        if codec != self._codec:
            self._codec = codec
            self.codec_changed.emit(codec)
            
            # Update preset to default for new codec
            codec_info = self._codec_manager.get_codec_info(codec)
            if codec_info and codec_info.default_preset:
                self._preset = codec_info.default_preset
                self.preset_changed.emit(self._preset)
    
    def set_quality(self, quality: int) -> None:
        """Set quality value.
        
        Args:
            quality: Quality integer value
        """
        if quality != self._quality:
            self._quality = quality
            self.quality_changed.emit(quality)
    
    def set_preset(self, preset: str) -> None:
        """Set preset selection.
        
        Args:
            preset: Preset string
        """
        if preset != self._preset:
            self._preset = preset
            self.preset_changed.emit(preset)
    
    def set_audio_copy(self, enabled: bool) -> None:
        """Set audio copy mode.
        
        Args:
            enabled: True to copy audio stream
        """
        if enabled != self._audio_copy:
            self._audio_copy = enabled
            self.audio_copy_changed.emit(enabled)
    
    # ====================
    # Persistence
    # ====================
    
    def persist(self) -> None:
        """Save configuration to ConfigFacade."""
        output_config = self._config.output
        
        output_config.set_codec(self._codec)
        output_config.set_quality(self._quality)
        output_config.set_preset(self._preset)
        output_config.set_audio_copy(self._audio_copy)
    
    def to_config(self) -> Dict[str, Any]:
        """Export as ProcessingConfig.output compatible dict.
        
        Returns:
            Dictionary with codec configuration for ProcessingConfig
        """
        return {
            "codec": self._codec,
            "quality": self._quality,
            "preset": self._preset,
            "audio_copy": self._audio_copy,
        }
    
    # ====================
    # Utility Methods
    # ====================
    
    def get_codec_info(self, codec_id: str) -> Dict[str, Any]:
        """Get codec information for display.
        
        Args:
            codec_id: Codec ID to query
            
        Returns:
            Dictionary with codec info (name, description, presets, etc.)
        """
        info = self._codec_manager.get_codec_info(codec_id)
        if info:
            return {
                "id": info.id,
                "name": info.name,
                "description": info.description,
                "type": info.type.value,
                "vendor": info.vendor.value,
                "presets": info.presets,
                "preset_names": info.preset_names,
                "default_preset": info.default_preset,
                "quality_range": info.quality_range,
                "quality_default": info.quality_default,
            }
        return {}
    
    def get_preset_display_name(self, preset: str) -> str:
        """Get display name for a preset.
        
        Args:
            preset: Preset string
            
        Returns:
            Human-readable preset name
        """
        codec_info = self._codec_manager.get_codec_info(self._codec)
        if codec_info and preset in codec_info.preset_names:
            return codec_info.preset_names[preset]
        return preset
    
    def refresh_available_codecs(self) -> None:
        """Refresh available codecs list from CodecManager."""
        all_codecs = self._codec_manager.get_all_codecs()
        new_list = list(all_codecs.keys())
        
        if new_list != self._available_codecs:
            self._available_codecs = new_list
            self.available_codecs_changed.emit(new_list)


__all__ = ["CodecViewModel"]