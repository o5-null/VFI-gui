"""Configuration provider - singleton pattern for global config access.

This module provides a centralized way to access configuration,
ensuring all components use the same config instance.
"""

from typing import Optional
from core.config import ConfigFacade

# Global config instance
_config_instance: Optional[ConfigFacade] = None


def get_config() -> ConfigFacade:
    """Get the global configuration instance.
    
    Creates the instance on first call, returns the same instance thereafter.
    
    Returns:
        ConfigFacade instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigFacade()
    return _config_instance


def set_config(config: ConfigFacade) -> None:
    """Set the global configuration instance.
    
    Used for dependency injection in tests or when custom config is needed.
    
    Args:
        config: ConfigFacade instance to use globally
    """
    global _config_instance
    _config_instance = config


def reset_config() -> None:
    """Reset the global configuration instance.
    
    Forces creation of a new instance on next get_config() call.
    Useful for testing.
    """
    global _config_instance
    _config_instance = None
