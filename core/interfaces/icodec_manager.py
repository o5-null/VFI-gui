"""ICodecManager interface for VFI-gui.

Abstract interface for codec management operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class ICodecManager(ABC):
    """Abstract interface for codec management.
    
    Defines the contract that any codec manager implementation must fulfill.
    """
    
    @abstractmethod
    def get_available_codecs(self) -> List[str]:
        """Get list of available codec names.
        
        Returns:
            List of codec identifiers
        """
        pass
    
    @abstractmethod
    def get_codec_info(self, codec_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific codec.
        
        Args:
            codec_name: Name of the codec
            
        Returns:
            Codec information dictionary, or None if not found
        """
        pass
    
    @abstractmethod
    def build_ffmpeg_args(self, config: Any) -> List[str]:
        """Build FFmpeg arguments from codec configuration.
        
        Args:
            config: CodecConfig object
            
        Returns:
            List of FFmpeg command-line arguments
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Any) -> tuple[bool, str]:
        """Validate codec configuration.
        
        Args:
            config: CodecConfig object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Any:
        """Get default codec configuration.
        
        Returns:
            Default CodecConfig object
        """
        pass
