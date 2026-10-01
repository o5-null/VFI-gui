"""Model selection management for VFI-gui.

Centralizes model selection state and provides a unified interface
for model type and checkpoint selection across the application.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from core.model_manager import (
    ModelManager,
    ModelTypeInfo,
    CheckpointInfo,
    MODEL_DEFINITIONS,
    CHECKPOINT_VERSION_MAP,
)


@dataclass
class ModelSelection:
    """Represents a model selection state.

    Attributes:
        model_type: 模型类型（如 ``"rife"``）
        checkpoint_name: checkpoint 文件名（UI 展示用，如 ``"rife49.pth"``）
        version: 版本令牌（config 语义，spec §7，如 ``"4.9"``/``"fp32"``）
        display_name: 模型类型的人类可读名
    """
    model_type: str = "rife"
    checkpoint_name: Optional[str] = None
    version: Optional[str] = None
    display_name: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Check if selection is valid (has checkpoint)."""
        return self.checkpoint_name is not None


class ModelSelectionManager(QObject):
    """Centralized manager for model selection state.
    
    Provides:
    - Unified model selection state management
    - Available model types and checkpoints discovery
    - Version to checkpoint mapping
    - Config integration for persistence
    - Signals for UI updates
    
    Usage:
        manager = ModelSelectionManager(config)
        
        # Get available model types
        types = manager.get_available_model_types()
        
        # Set selection
        manager.set_model_type("rife")
        manager.set_checkpoint("rife49.pth")
        
        # Get current selection
        selection = manager.get_selection()
    """
    
    # Signals
    model_type_changed = pyqtSignal(str)  # new_model_type
    checkpoint_changed = pyqtSignal(str, str)  # model_type, checkpoint_name
    available_models_changed = pyqtSignal()  # models installed/uninstalled
    selection_changed = pyqtSignal()  # any selection change
    
    def __init__(self, config, model_manager: Optional[ModelManager] = None, parent=None):
        """Initialize the selection manager.
        
        Args:
            config: Config instance for persistence
            model_manager: Optional ModelManager instance (creates one if None)
            parent: Parent QObject
        """
        super().__init__(parent)
        self._config = config
        
        # Use provided model manager or create one
        if model_manager is None:
            from core.config_provider import get_config
            temp_config = get_config()
            self._model_manager = ModelManager(temp_config)
        else:
            self._model_manager = model_manager

        # Connect to ModelManager signals for real-time updates
        # Using Blinker signals instead of Qt signals
        from core.events import models_updated, engines_updated
        models_updated.connect(self._on_model_manager_updated, sender=self._model_manager)
        engines_updated.connect(self._on_model_manager_updated, sender=self._model_manager)

        # Flag to prevent recursion during signal handling
        self._handling_model_update = False

        # Current selection
        self._current_type: str = ""
        self._current_checkpoint: Optional[str] = None
        
        # Load saved selection from config
        self._load_from_config()
    
    def _load_from_config(self):
        """Load model selection from config."""
        self._current_type = self._config.get("pipeline.interpolation.model_type", "rife").lower()
        version = self._config.get("pipeline.interpolation.model_version", "")
        
        # Convert version to checkpoint name
        if version:
            self._current_checkpoint = self._version_to_checkpoint(self._current_type, version)
        
        # Validate selection
        self._validate_selection()
    
    def _save_to_config(self):
        """Save current selection to config."""
        self._config.set("pipeline.interpolation.model_type", self._current_type)
        
        if self._current_checkpoint:
            version = self._checkpoint_to_version(self._current_type, self._current_checkpoint)
            self._config.set("pipeline.interpolation.model_version", version)
    
    def _on_model_manager_updated(self, sender):
        """Handle ModelManager models_updated or engines_updated signal.
        
        This is called when ModelManager finishes scanning models/engines,
        ensuring the selection manager stays synchronized with available models.
        
        Note: ModelManager has already refreshed its data before emitting this signal,
        so we validate selection without calling refresh() to avoid recursion.
        
        Args:
            sender: The sender of the signal (ModelManager instance)
        """
        # Prevent recursion: if we're already handling an update, skip
        if self._handling_model_update:
            return
        
        self._handling_model_update = True
        try:
            old_type = self._current_type
            old_checkpoint = self._current_checkpoint
            
            # Validate selection using current ModelManager data (no refresh needed)
            self._validate_selection_no_refresh()
            
            # Emit signals to notify UI components
            if self._current_type != old_type:
                self.model_type_changed.emit(self._current_type)
            if self._current_checkpoint != old_checkpoint:
                if self._current_checkpoint:
                    self.checkpoint_changed.emit(self._current_type, self._current_checkpoint)
            
            self.available_models_changed.emit()
        finally:
            self._handling_model_update = False
    
    def _validate_selection_no_refresh(self):
        """Validate current selection using existing ModelManager data (no refresh).
        
        Used by signal handlers when ModelManager has already refreshed.
        """
        # Get current model types (ModelManager already has fresh data)
        model_types = self._model_manager.get_model_types()
        
        # Check if current type is valid
        if self._current_type not in model_types:
            # Select first available type with installed models
            for mt, info in model_types.items():
                if info.installed_count > 0:
                    self._current_type = mt
                    self._current_checkpoint = None
                    break
            else:
                # No models installed, use first type
                self._current_type = list(model_types.keys())[0] if model_types else "rife"
                self._current_checkpoint = None
        
        # Check if current checkpoint is valid
        if self._current_checkpoint:
            type_info = model_types.get(self._current_type)
            if type_info:
                valid_checkpoints = [c.name for c in type_info.checkpoints if c.is_installed]
                if self._current_checkpoint not in valid_checkpoints:
                    # Select first available checkpoint
                    self._current_checkpoint = valid_checkpoints[0] if valid_checkpoints else None
            else:
                self._current_checkpoint = None
        
        # If no checkpoint selected, try to select default
        if not self._current_checkpoint:
            self._select_default_checkpoint()
    
    def _validate_selection(self):
        """Validate current selection and fix if needed."""
        # Refresh model manager
        self._model_manager.refresh()
        
        # Check if current type is valid
        model_types = self._model_manager.get_model_types()
        
        if self._current_type not in model_types:
            # Select first available type with installed models
            for mt, info in model_types.items():
                if info.installed_count > 0:
                    self._current_type = mt
                    self._current_checkpoint = None
                    break
            else:
                # No models installed, use first type
                self._current_type = list(model_types.keys())[0] if model_types else "rife"
                self._current_checkpoint = None
        
        # Check if current checkpoint is valid
        if self._current_checkpoint:
            type_info = model_types.get(self._current_type)
            if type_info:
                valid_checkpoints = [c.name for c in type_info.checkpoints if c.is_installed]
                if self._current_checkpoint not in valid_checkpoints:
                    # Select first available checkpoint
                    self._current_checkpoint = valid_checkpoints[0] if valid_checkpoints else None
            else:
                self._current_checkpoint = None
        
        # If no checkpoint selected, try to select default
        if not self._current_checkpoint:
            self._select_default_checkpoint()
    
    def _select_default_checkpoint(self):
        """Select default checkpoint for current type."""
        model_types = self._model_manager.get_model_types()
        type_info = model_types.get(self._current_type)
        
        if type_info:
            installed = [c for c in type_info.checkpoints if c.is_installed]
            if installed:
                # Prefer specific defaults
                defaults = {
                    "rife": "rife49.pth",
                    "film": "film_net_fp32.pt",
                    "amt": "amt-s.pth",
                }
                default_ckpt = defaults.get(self._current_type)
                if default_ckpt and any(c.name == default_ckpt for c in installed):
                    self._current_checkpoint = default_ckpt
                else:
                    self._current_checkpoint = installed[0].name
    
    def _version_to_checkpoint(self, model_type: str, version: str) -> Optional[str]:
        """Convert version token to checkpoint name.

        Single source of truth is :mod:`core.models.asset_resolver` (spec §4/§7).

        Args:
            model_type: Model type name
            version: Version token (e.g., "4.9", "fp32")

        Returns:
            Checkpoint filename or None
        """
        try:
            from core.models.asset_resolver import version_to_checkpoint
            return version_to_checkpoint(model_type, version)
        except ImportError:  # 过渡期回退：复用 model_manager 的派生映射（同源）
            for ckpt, ver in CHECKPOINT_VERSION_MAP.get(model_type, {}).items():
                if ver == version:
                    return ckpt
            return None

    def _checkpoint_to_version(self, model_type: str, checkpoint: str) -> str:
        """Convert checkpoint name to version token.

        Single source of truth is :mod:`core.models.asset_resolver` (spec §4/§7).
        Unknown checkpoints fall back to their own name so round-tripping is safe.

        Args:
            model_type: Model type name
            checkpoint: Checkpoint filename

        Returns:
            Version token
        """
        try:
            from core.models.asset_resolver import checkpoint_to_version
            version = checkpoint_to_version(model_type, checkpoint)
            if version is not None:
                return version
        except ImportError:  # 过渡期回退：复用 model_manager 的派生映射（同源）
            if checkpoint in CHECKPOINT_VERSION_MAP.get(model_type, {}):
                return CHECKPOINT_VERSION_MAP[model_type][checkpoint]
        return checkpoint

    def checkpoint_to_version(self, model_type: str, checkpoint: str) -> str:
        """Public wrapper: checkpoint 名 -> 版本令牌（spec §7 语义）。"""
        return self._checkpoint_to_version(model_type, checkpoint)

    def version_to_checkpoint(self, model_type: str, version: str) -> Optional[str]:
        """Public wrapper: 版本令牌 -> checkpoint 名（UI 展示用）。"""
        return self._version_to_checkpoint(model_type, version)
    
    # Public API
    
    def refresh(self):
        """Refresh available models from disk."""
        self._model_manager.refresh()
        old_type = self._current_type
        old_checkpoint = self._current_checkpoint
        
        self._validate_selection()
        
        # Emit signals if changed
        if self._current_type != old_type:
            self.model_type_changed.emit(self._current_type)
        if self._current_checkpoint != old_checkpoint:
            if self._current_checkpoint:
                self.checkpoint_changed.emit(self._current_type, self._current_checkpoint)
        
        self.available_models_changed.emit()
        self.selection_changed.emit()
    
    def get_available_model_types(self) -> List[ModelTypeInfo]:
        """Get list of model types with installed models.
        
        Returns:
            List of ModelTypeInfo for types that have installed checkpoints,
            sorted by sort_order (lower = more recommended)
        """
        self._model_manager.refresh()
        result = []
        for mt, info in self._model_manager.get_model_types().items():
            if info.installed_count > 0:
                result.append(info)
        # Sort by sort_order (lower = displayed first)
        return sorted(result, key=lambda x: x.sort_order)
    
    def get_all_model_types(self) -> List[ModelTypeInfo]:
        """Get all model types regardless of installation status.
        
        Returns:
            List of all ModelTypeInfo, sorted by sort_order
        """
        self._model_manager.refresh()
        all_types = list(self._model_manager.get_model_types().values())
        # Sort by sort_order (lower = displayed first)
        return sorted(all_types, key=lambda x: x.sort_order)
    
    def get_available_checkpoints(self, model_type: Optional[str] = None) -> List[CheckpointInfo]:
        """Get available checkpoints for a model type.
        
        Args:
            model_type: Model type name, or None for current type
            
        Returns:
            List of installed CheckpointInfo
        """
        mt = model_type or self._current_type
        model_types = self._model_manager.get_model_types()
        
        if mt in model_types:
            return [c for c in model_types[mt].checkpoints if c.is_installed]
        return []
    
    def get_selection(self) -> ModelSelection:
        """Get current selection.
        
        Returns:
            ModelSelection with current state
        """
        display_name = ""
        if self._current_type:
            model_types = self._model_manager.get_model_types()
            if self._current_type in model_types:
                display_name = model_types[self._current_type].display_name
        
        return ModelSelection(
            model_type=self._current_type,
            checkpoint_name=self._current_checkpoint,
            version=self.get_version_string() or None,
            display_name=display_name,
        )
    
    def set_model_type(self, model_type: str) -> bool:
        """Set the current model type.
        
        Args:
            model_type: Model type name
            
        Returns:
            True if successful
        """
        model_types = self._model_manager.get_model_types()
        
        if model_type not in model_types:
            return False
        
        if model_type == self._current_type:
            return True
        
        self._current_type = model_type
        
        # Select default checkpoint for new type
        old_checkpoint = self._current_checkpoint
        self._current_checkpoint = None
        self._select_default_checkpoint()
        
        self._save_to_config()
        
        self.model_type_changed.emit(model_type)
        if self._current_checkpoint != old_checkpoint:
            self.checkpoint_changed.emit(model_type, self._current_checkpoint or "")
        self.selection_changed.emit()
        
        return True
    
    def set_checkpoint(self, checkpoint_name: str) -> bool:
        """Set the current checkpoint.
        
        Args:
            checkpoint_name: Checkpoint filename
            
        Returns:
            True if successful
        """
        available = self.get_available_checkpoints()
        valid_names = [c.name for c in available]
        
        if checkpoint_name not in valid_names:
            return False
        
        if checkpoint_name == self._current_checkpoint:
            return True
        
        self._current_checkpoint = checkpoint_name
        self._save_to_config()
        
        self.checkpoint_changed.emit(self._current_type, checkpoint_name)
        self.selection_changed.emit()
        
        return True
    
    def get_checkpoint_path(self) -> Optional[str]:
        """Get full path to current checkpoint.
        
        Returns:
            Absolute path to checkpoint file, or None if not selected
        """
        if not self._current_checkpoint:
            return None
        return self._model_manager.get_checkpoint_path(self._current_type, self._current_checkpoint)
    
    def get_model_manager(self) -> ModelManager:
        """Get the underlying ModelManager.
        
        Returns:
            ModelManager instance
        """
        return self._model_manager
    
    def is_model_available(self, model_type: str, checkpoint_name: str) -> bool:
        """Check if a specific model is available.
        
        Args:
            model_type: Model type name
            checkpoint_name: Checkpoint filename
            
        Returns:
            True if model is installed
        """
        return self._model_manager.is_checkpoint_installed(model_type, checkpoint_name)
    
    def get_version_string(self) -> str:
        """Get version string for current selection.
        
        Returns:
            Version string for config storage
        """
        if not self._current_checkpoint:
            return ""
        return self._checkpoint_to_version(self._current_type, self._current_checkpoint)
