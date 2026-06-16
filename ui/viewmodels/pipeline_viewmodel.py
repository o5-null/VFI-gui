"""PipelineViewModel for processing pipeline configuration.

Wraps PipelineConfig and ModelSelectionManager to provide Qt-signals for UI binding.
The most complex ViewModel, handling interpolation, upscaling, and scene detection settings.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.config.config_facade import ConfigFacade
from core.model_selection import ModelSelectionManager


class PipelineViewModel(QObject):
    """ViewModel for processing pipeline configuration.
    
    Signals:
        interp_enabled_changed: Interpolation enabled state (bool)
        model_type_changed: Model type selection (str)
        checkpoint_changed: Model checkpoint selection (str)
        multiplier_changed: Frame multiplier (int)
        scale_changed: Scale factor (float)
        upscale_enabled_changed: Upscaling enabled state (bool)
        upscale_engine_changed: Upscaling engine selection (str)
        scene_detect_enabled_changed: Scene detection enabled state (bool)
        scene_detect_method_changed: Scene detection method (str)
        scene_detect_threshold_changed: Detection threshold (float)
        video_path_changed: Input video path (str)
        input_type_changed: Input type ("video" or "image_sequence")
        available_models_changed: Available model types list
        available_checkpoints_changed: Available checkpoints for current model
    
    Properties:
        All configuration values as properties
    
    Methods:
        persist(): Save configuration to ConfigFacade
        validate(): Return validation errors
        to_pipeline_config(): Export as ProcessingConfig compatible dict
    """
    
    # Signals for interpolation
    interp_enabled_changed = pyqtSignal(bool)
    model_type_changed = pyqtSignal(str)
    checkpoint_changed = pyqtSignal(str)
    multiplier_changed = pyqtSignal(int)
    scale_changed = pyqtSignal(float)
    
    # Signals for upscaling
    upscale_enabled_changed = pyqtSignal(bool)
    upscale_engine_changed = pyqtSignal(str)
    
    # Signals for scene detection
    scene_detect_enabled_changed = pyqtSignal(bool)
    scene_detect_method_changed = pyqtSignal(str)
    scene_detect_threshold_changed = pyqtSignal(float)
    
    # Signals for input
    video_path_changed = pyqtSignal(str)
    input_type_changed = pyqtSignal(str)
    
    # Signals for model selection
    available_models_changed = pyqtSignal(list)
    available_checkpoints_changed = pyqtSignal(list)
    
    def __init__(
        self,
        config: ConfigFacade,
        model_selection: ModelSelectionManager,
        codec_viewmodel,  # CodecViewModel - avoid circular import
        parent=None,
    ):
        """Initialize PipelineViewModel.
        
        Args:
            config: ConfigFacade instance for persistence
            model_selection: ModelSelectionManager for model/checkpoint state
            codec_viewmodel: CodecViewModel instance for output config
            parent: Parent QObject
        """
        super().__init__(parent)
        self._config = config
        self._model_selection = model_selection
        self._codec_vm = codec_viewmodel
        
        # Private fields - interpolation
        self._interp_enabled: bool = False
        self._model_type: str = "rife"
        self._checkpoint: str = ""
        self._multiplier: int = 2
        self._scale: float = 1.0
        
        # Private fields - upscaling
        self._upscale_enabled: bool = False
        self._upscale_engine: str = ""
        
        # Private fields - scene detection
        self._scene_detect_enabled: bool = False
        self._scene_detect_method: str = "neural"
        self._scene_detect_threshold: float = 0.5
        
        # Private fields - input
        self._video_path: str = ""
        self._input_type: str = "video"
        
        # Private fields - model selection
        self._available_models: List[str] = []
        self._available_checkpoints: List[str] = []
        
        # Load initial state from config
        self._load_from_config()
        
        # Forward ModelSelectionManager signals
        self._forward_model_selection_signals()
    
    def _load_from_config(self) -> None:
        """Load configuration values from ConfigFacade."""
        pipeline = self._config.pipeline
        
        # Load interpolation config
        interp = pipeline.get_interpolation_config()
        self._interp_enabled = interp.get("enabled", False)
        self._model_type = interp.get("model_type", "rife")
        self._checkpoint = interp.get("model_version", "")
        self._multiplier = interp.get("multi", 2)
        self._scale = interp.get("scale", 1.0)
        
        # Load upscaling config
        upscale = pipeline.get_upscaling_config()
        self._upscale_enabled = upscale.get("enabled", False)
        self._upscale_engine = upscale.get("engine", "")
        
        # Load scene detection config
        scene = pipeline.get_scene_detection_config()
        self._scene_detect_enabled = scene.get("enabled", False)
        self._scene_detect_method = str(scene.get("model", "neural"))
        self._scene_detect_threshold = scene.get("threshold", 0.5)
        
        # Sync with ModelSelectionManager
        selection = self._model_selection.get_selection()
        if selection.checkpoint_name:
            self._checkpoint = selection.checkpoint_name
        self._model_type = selection.model_type
        
        # Get available models
        self._refresh_available_models()
    
    def _forward_model_selection_signals(self) -> None:
        """Connect ModelSelectionManager signals to ViewModel signals."""
        # Forward model type changes
        self._model_selection.model_type_changed.connect(self._on_model_type_changed)
        
        # Forward checkpoint changes
        self._model_selection.checkpoint_changed.connect(self._on_checkpoint_changed)
        
        # Forward available models changes
        self._model_selection.available_models_changed.connect(self._on_available_models_changed)
    
    def _on_model_type_changed(self, model_type: str) -> None:
        """Handle model type change from ModelSelectionManager."""
        if model_type != self._model_type:
            self._model_type = model_type
            self.model_type_changed.emit(model_type)
            self._refresh_available_checkpoints()
    
    def _on_checkpoint_changed(self, model_type: str, checkpoint: str) -> None:
        """Handle checkpoint change from ModelSelectionManager."""
        if checkpoint != self._checkpoint:
            self._checkpoint = checkpoint
            self.checkpoint_changed.emit(checkpoint)
    
    def _on_available_models_changed(self) -> None:
        """Handle available models update from ModelSelectionManager."""
        self._refresh_available_models()
    
    def _refresh_available_models(self) -> None:
        """Refresh available model types list."""
        model_types = self._model_selection.get_available_model_types()
        new_list = [mt.name for mt in model_types]
        
        if new_list != self._available_models:
            self._available_models = new_list
            self.available_models_changed.emit(new_list)
        
        self._refresh_available_checkpoints()
    
    def _refresh_available_checkpoints(self) -> None:
        """Refresh available checkpoints for current model type."""
        checkpoints = self._model_selection.get_available_checkpoints(self._model_type)
        new_list = [c.name for c in checkpoints]
        
        if new_list != self._available_checkpoints:
            self._available_checkpoints = new_list
            self.available_checkpoints_changed.emit(new_list)
    
    # ====================
    # Properties - Interpolation
    # ====================
    
    @property
    def interp_enabled(self) -> bool:
        """Get interpolation enabled state."""
        return self._interp_enabled
    
    @property
    def model_type(self) -> str:
        """Get current model type."""
        return self._model_type
    
    @property
    def checkpoint(self) -> str:
        """Get current checkpoint."""
        return self._checkpoint
    
    @property
    def multiplier(self) -> int:
        """Get frame multiplier."""
        return self._multiplier
    
    @property
    def scale(self) -> float:
        """Get scale factor."""
        return self._scale
    
    # ====================
    # Properties - Upscaling
    # ====================
    
    @property
    def upscale_enabled(self) -> bool:
        """Get upscaling enabled state."""
        return self._upscale_enabled
    
    @property
    def upscale_engine(self) -> str:
        """Get upscaling engine."""
        return self._upscale_engine
    
    # ====================
    # Properties - Scene Detection
    # ====================
    
    @property
    def scene_detect_enabled(self) -> bool:
        """Get scene detection enabled state."""
        return self._scene_detect_enabled
    
    @property
    def scene_detect_method(self) -> str:
        """Get scene detection method."""
        return self._scene_detect_method
    
    @property
    def scene_detect_threshold(self) -> float:
        """Get detection threshold."""
        return self._scene_detect_threshold
    
    # ====================
    # Properties - Input
    # ====================
    
    @property
    def video_path(self) -> str:
        """Get input video path."""
        return self._video_path
    
    @property
    def input_type(self) -> str:
        """Get input type ('video' or 'image_sequence')."""
        return self._input_type
    
    # ====================
    # Properties - Model Selection
    # ====================
    
    @property
    def available_models(self) -> List[str]:
        """Get available model types."""
        return self._available_models.copy()
    
    @property
    def available_checkpoints(self) -> List[str]:
        """Get available checkpoints for current model."""
        return self._available_checkpoints.copy()
    
    # ====================
    # Setters (emit signals, no auto-persist)
    # ====================
    
    def set_interp_enabled(self, enabled: bool) -> None:
        """Set interpolation enabled state."""
        if enabled != self._interp_enabled:
            self._interp_enabled = enabled
            self.interp_enabled_changed.emit(enabled)
    
    def set_model_type(self, model_type: str) -> None:
        """Set model type.
        
        Also updates ModelSelectionManager and refreshes checkpoints.
        """
        if model_type != self._model_type:
            self._model_selection.set_model_type(model_type)
    
    def set_checkpoint(self, checkpoint: str) -> None:
        """Set checkpoint.
        
        Also updates ModelSelectionManager.
        """
        if checkpoint != self._checkpoint:
            self._model_selection.set_checkpoint(checkpoint)
    
    def set_multiplier(self, multiplier: int) -> None:
        """Set frame multiplier."""
        if multiplier != self._multiplier:
            self._multiplier = multiplier
            self.multiplier_changed.emit(multiplier)
    
    def set_scale(self, scale: float) -> None:
        """Set scale factor."""
        if scale != self._scale:
            self._scale = scale
            self.scale_changed.emit(scale)
    
    def set_upscale_enabled(self, enabled: bool) -> None:
        """Set upscaling enabled state."""
        if enabled != self._upscale_enabled:
            self._upscale_enabled = enabled
            self.upscale_enabled_changed.emit(enabled)
    
    def set_upscale_engine(self, engine: str) -> None:
        """Set upscaling engine."""
        if engine != self._upscale_engine:
            self._upscale_engine = engine
            self.upscale_engine_changed.emit(engine)
    
    def set_scene_detect_enabled(self, enabled: bool) -> None:
        """Set scene detection enabled state."""
        if enabled != self._scene_detect_enabled:
            self._scene_detect_enabled = enabled
            self.scene_detect_enabled_changed.emit(enabled)
    
    def set_scene_detect_method(self, method: str) -> None:
        """Set scene detection method."""
        if method != self._scene_detect_method:
            self._scene_detect_method = method
            self.scene_detect_method_changed.emit(method)
    
    def set_scene_detect_threshold(self, threshold: float) -> None:
        """Set detection threshold."""
        if threshold != self._scene_detect_threshold:
            self._scene_detect_threshold = threshold
            self.scene_detect_threshold_changed.emit(threshold)
    
    def set_video_path(self, path: str) -> None:
        """Set input video path.
        
        Also updates input_type based on path extension.
        """
        if path != self._video_path:
            self._video_path = path
            
            # Detect input type
            if path:
                ext = Path(path).suffix.lower()
                image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                if ext in image_exts:
                    self._input_type = "image_sequence"
                else:
                    self._input_type = "video"
            else:
                self._input_type = "video"
            
            self.video_path_changed.emit(path)
            self.input_type_changed.emit(self._input_type)
    
    # ====================
    # Persistence
    # ====================
    
    def persist(self) -> None:
        """Save configuration to ConfigFacade."""
        # Save interpolation config
        interp_config = {
            "enabled": self._interp_enabled,
            "model_type": self._model_type,
            "model_version": self._checkpoint,
            "multi": self._multiplier,
            "scale": self._scale,
            "scene_change": self._scene_detect_enabled,
        }
        self._config.pipeline.set_interpolation_config(interp_config)
        
        # Save upscaling config
        upscale_config = {
            "enabled": self._upscale_enabled,
            "engine": self._upscale_engine,
            "tile_size": 0,
            "overlap": 0,
            "num_streams": 3,
        }
        self._config.pipeline.set_upscaling_config(upscale_config)
        
        # Save scene detection config
        scene_config = {
            "enabled": self._scene_detect_enabled,
            "model": int(self._scene_detect_method) if self._scene_detect_method.isdigit() else 12,
            "threshold": self._scene_detect_threshold,
            "fp16": True,
        }
        self._config.pipeline.set_scene_detection_config(scene_config)
        
        # Persist codec config
        self._codec_vm.persist()
    
    def validate(self) -> List[str]:
        """Validate configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors: List[str] = []
        
        # Check video path
        if self._video_path:
            if not Path(self._video_path).exists():
                errors.append(f"Video file not found: {self._video_path}")
        else:
            errors.append("No video file selected")
        
        # Check interpolation settings
        if self._interp_enabled:
            if not self._checkpoint:
                errors.append("No model checkpoint selected")
            
            if self._multiplier < 2:
                errors.append("Multiplier must be at least 2")
        
        # Check upscaling settings
        if self._upscale_enabled and not self._upscale_engine:
            errors.append("No upscaling engine selected")
        
        # Check scene detection settings
        if self._scene_detect_enabled:
            if self._scene_detect_threshold < 0.0 or self._scene_detect_threshold > 1.0:
                errors.append("Threshold must be between 0 and 1")
        
        return errors
    
    def to_pipeline_config(self) -> Dict[str, Any]:
        """Export as ProcessingConfig compatible dict.
        
        Returns:
            Dictionary with full pipeline configuration
        """
        return {
            "interpolation": {
                "enabled": self._interp_enabled,
                "model_type": self._model_type,
                "model_version": self._checkpoint,
                "multi": self._multiplier,
                "scale": self._scale,
                "scene_change": self._scene_detect_enabled,
            },
            "upscaling": {
                "enabled": self._upscale_enabled,
                "engine": self._upscale_engine,
                "num_streams": 3,
                "tile_size": 0,
                "overlap": 0,
            },
            "scene_detection": {
                "enabled": self._scene_detect_enabled,
                "model": int(self._scene_detect_method) if self._scene_detect_method.isdigit() else 12,
                "threshold": self._scene_detect_threshold,
                "fp16": True,
            },
            "output": self._codec_vm.to_config(),
        }
    
    # ====================
    # Utility Methods
    # ====================
    
    def refresh_models(self) -> None:
        """Refresh available models from ModelSelectionManager."""
        self._model_selection.refresh()
    
    def get_model_display_name(self, model_type: str) -> str:
        """Get display name for a model type.
        
        Args:
            model_type: Model type name
            
        Returns:
            Human-readable display name
        """
        model_types = self._model_selection.get_all_model_types()
        for mt in model_types:
            if mt.name == model_type:
                return mt.display_name
        return model_type
    
    def get_checkpoint_path(self) -> Optional[str]:
        """Get full path to current checkpoint.
        
        Returns:
            Absolute path or None if not selected
        """
        return self._model_selection.get_checkpoint_path()


__all__ = ["PipelineViewModel"]