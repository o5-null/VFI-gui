"""Model ViewModel for VFI-gui.

Acts as intermediary between Model panels and ModelManager,
providing a clean interface for UI-model operations.
"""

from typing import Dict, Any, Optional, List
from PyQt6.QtCore import QObject, pyqtSignal

from core.models import ModelManager, CheckpointInfo, ModelTypeInfo


class ModelViewModel(QObject):
    """ViewModel for model management.
    
    Decouples UI widgets from ModelManager implementation details.
    Provides observable properties for model state.
    """
    
    # Signals for UI updates
    models_refreshed = pyqtSignal()  # model list changed
    download_started = pyqtSignal(str, int)  # model_name, total_bytes
    download_progress = pyqtSignal(str, int, int)  # model_name, downloaded, total
    download_finished = pyqtSignal(str, bool)  # model_name, success
    download_cancelled = pyqtSignal(str)  # model_name
    checkpoint_deleted = pyqtSignal(str)  # checkpoint_name
    
    def __init__(self, model_manager: Optional[ModelManager] = None, parent=None):
        super().__init__(parent)
        self._model_manager = model_manager or ModelManager()
        self._selected_model_type: Optional[str] = None
        self._selected_checkpoint: Optional[str] = None
    
    def refresh_models(self) -> None:
        """Refresh model information from disk."""
        self._model_manager.refresh()
        self.models_refreshed.emit()
    
    def get_model_types(self) -> List[str]:
        """Get list of available model type names."""
        return [mt.name for mt in self._model_manager.get_model_types_summary()]
    
    def get_model_type_info(self, model_type: str) -> Optional[ModelTypeInfo]:
        """Get detailed information about a model type."""
        for mt in self._model_manager.get_model_types_summary():
            if mt.name == model_type:
                return mt
        return None
    
    def get_checkpoints(self, model_type: Optional[str] = None) -> List[CheckpointInfo]:
        """Get list of checkpoints for a model type.
        
        Args:
            model_type: Model type filter, or None for all
            
        Returns:
            List of checkpoint information
        """
        if model_type:
            info = self.get_model_type_info(model_type)
            if info:
                return info.checkpoints
            return []
        
        # Return all checkpoints
        all_checkpoints = []
        for mt in self._model_manager.get_model_types_summary():
            all_checkpoints.extend(mt.checkpoints)
        return all_checkpoints
    
    def get_installed_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of installed checkpoints."""
        return self._model_manager.get_installed_checkpoints()
    
    def get_missing_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of missing (downloadable) checkpoints."""
        return self._model_manager.get_missing_checkpoints()
    
    def is_checkpoint_installed(self, checkpoint_name: str) -> bool:
        """Check if a checkpoint is installed."""
        return self._model_manager.is_checkpoint_installed(checkpoint_name)
    
    def get_checkpoint_path(self, checkpoint_name: str) -> Optional[str]:
        """Get path to checkpoint file."""
        return self._model_manager.get_checkpoint_path(checkpoint_name)
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        success = self._model_manager.delete_checkpoint(checkpoint_name)
        if success:
            self.checkpoint_deleted.emit(checkpoint_name)
            self.models_refreshed.emit()
        return success
    
    def get_total_size_mb(self) -> float:
        """Get total size of all installed models in MB."""
        return self._model_manager.get_total_size_mb()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model status.
        
        Returns:
            Dictionary with installed count, total count, size info
        """
        installed = len(self.get_installed_checkpoints())
        total = len(self.get_checkpoints())
        
        return {
            "installed": installed,
            "total": total,
            "missing": total - installed,
            "size_mb": self.get_total_size_mb(),
        }
