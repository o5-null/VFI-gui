"""IModelManager interface for VFI-gui.

Abstract interface for model management operations.
This decouples the GUI from specific model management implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from enum import Enum, auto


class ModelStatus(Enum):
    """Model installation status."""
    NOT_INSTALLED = auto()
    INSTALLING = auto()
    INSTALLED = auto()
    UPDATE_AVAILABLE = auto()
    ERROR = auto()


class ModelInfo:
    """Information about a model."""
    
    def __init__(
        self,
        model_id: str,
        name: str,
        version: str,
        description: str = "",
        size_mb: float = 0.0,
        status: ModelStatus = ModelStatus.NOT_INSTALLED,
    ):
        self.model_id = model_id
        self.name = name
        self.version = version
        self.description = description
        self.size_mb = size_mb
        self.status = status


class IModelManager(ABC):
    """Abstract interface for model management.
    
    Defines the contract that any model manager implementation must fulfill.
    This enables:
    - Different model sources (local, remote, bundled)
    - Mock model managers for testing
    - Different download strategies
    """
    
    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of all available models.
        
        Returns:
            List of model information objects
        """
        pass
    
    @abstractmethod
    def get_installed_models(self) -> List[ModelInfo]:
        """Get list of installed models.
        
        Returns:
            List of installed model information
        """
        pass
    
    @abstractmethod
    def get_model_status(self, model_id: str) -> ModelStatus:
        """Get status of a specific model.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Current model status
        """
        pass
    
    @abstractmethod
    def install_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> bool:
        """Install a model.
        
        Args:
            model_id: Model to install
            progress_callback: Optional callback(progress_percent, status_message)
            
        Returns:
            True if installation succeeded
        """
        pass
    
    @abstractmethod
    def uninstall_model(self, model_id: str) -> bool:
        """Uninstall a model.
        
        Args:
            model_id: Model to uninstall
            
        Returns:
            True if uninstallation succeeded
        """
        pass
    
    @abstractmethod
    def update_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> bool:
        """Update a model to latest version.
        
        Args:
            model_id: Model to update
            progress_callback: Optional callback(progress_percent, status_message)
            
        Returns:
            True if update succeeded
        """
        pass
    
    @abstractmethod
    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get filesystem path to model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model files or None if not installed
        """
        pass
    
    @abstractmethod
    def check_for_updates(self) -> Dict[str, str]:
        """Check for available model updates.
        
        Returns:
            Dictionary mapping model_id to latest version
        """
        pass
    
    @abstractmethod
    def get_total_size_mb(self) -> float:
        """Get total size of all installed models.
        
        Returns:
            Total size in megabytes
        """
        pass
    
    @abstractmethod
    def is_model_available(self, model_id: str) -> bool:
        """Check if a model can be used for inference.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model is installed and ready
        """
        pass
