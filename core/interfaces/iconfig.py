"""IConfig interface for VFI-gui.

Abstract interface for configuration management operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class IConfig(ABC):
    """Abstract interface for configuration management.
    
    Defines the contract that any configuration implementation must fulfill.
    This enables:
    - Different storage backends (JSON, YAML, database, etc.)
    - Mock configurations for testing
    - Layered configurations (user + system + defaults)
    """
    
    @abstractmethod
    def load(self) -> None:
        """Load configuration from storage."""
        pass
    
    @abstractmethod
    def save(self) -> None:
        """Save configuration to storage."""
        pass
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Dot-notation key (e.g., "pipeline.interpolation.model_type")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Dot-notation key
            value: Value to set
        """
        pass
    
    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.
        
        Returns:
            Dictionary of all settings
        """
        pass
    
    @abstractmethod
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        pass
    
    @abstractmethod
    def get_config_path(self) -> str:
        """Get the configuration file path.
        
        Returns:
            Path to configuration file
        """
        pass
