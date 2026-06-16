"""InterpolationGroup - Interpolation settings widget.

A QGroupBox for configuring video frame interpolation settings.
Bidirectional binding with PipelineViewModel.
"""

from typing import List

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
)

from ui.styles.theme import Theme
from ui.viewmodels.pipeline_viewmodel import PipelineViewModel


class InterpolationGroup(QGroupBox):
    """Interpolation settings widget with ViewModel binding.
    
    Features:
        - Enable/disable toggle for interpolation
        - Model type selection (populated from ViewModel)
        - Checkpoint/version selection (populated from ViewModel)
        - Multiplier selection (2x, 4x, 8x)
        - Scale factor adjustment
        - Scene-change toggle
    
    All widgets use QFormLayout for clean alignment.
    Bidirectional binding ensures UI ↔ ViewModel synchronization.
    """

    def __init__(self, vm: PipelineViewModel, parent=None):
        """Initialize InterpolationGroup.
        
        Args:
            vm: PipelineViewModel instance for data binding
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        
        # Initialize and bind widgets
        self._setup_ui()
        self._bind_viewmodel()
    
    def _setup_ui(self) -> None:
        """Create and arrange widgets using QFormLayout."""
        self.setTitle(self.tr("Interpolation"))
        
        layout = QFormLayout(self)
        layout.setSpacing(Theme.SPACING_MD)
        layout.setContentsMargins(
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
        )
        
        # Enable/disable checkbox
        self._enabled_check = QCheckBox(self.tr("Enable Interpolation"))
        layout.addRow(self._enabled_check)
        
        # Model type selection
        self._model_combo = QComboBox()
        self._model_combo.setToolTip(self.tr("Select interpolation model type"))
        layout.addRow(self.tr("Model Type:"), self._model_combo)
        
        # Checkpoint/version selection
        self._checkpoint_combo = QComboBox()
        self._checkpoint_combo.setToolTip(self.tr("Select model checkpoint/version"))
        layout.addRow(self.tr("Checkpoint:"), self._checkpoint_combo)
        
        # Multiplier selection
        self._multiplier_combo = QComboBox()
        self._multiplier_combo.addItems(["2x", "4x", "8x"])
        self._multiplier_combo.setToolTip(self.tr("Frame multiplier for interpolation"))
        layout.addRow(self.tr("Multiplier:"), self._multiplier_combo)
        
        # Scale factor
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.5, 4.0)
        self._scale_spin.setSingleStep(0.1)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setToolTip(self.tr("Scale factor for model inference"))
        layout.addRow(self.tr("Scale:"), self._scale_spin)
        
        # Scene-change detection toggle
        self._scene_change_check = QCheckBox(self.tr("Scene Change Detection"))
        self._scene_change_check.setToolTip(
            self.tr("Enable scene change detection for better interpolation")
        )
        layout.addRow(self._scene_change_check)
        
        # Populate initial values
        self._populate_models(self._vm.available_models)
        self._populate_checkpoints(self._vm.available_checkpoints)
        
        # Set initial state from ViewModel
        self._enabled_check.setChecked(self._vm.interp_enabled)
        self._set_current_model(self._vm.model_type)
        self._set_current_checkpoint(self._vm.checkpoint)
        self._set_current_multiplier(self._vm.multiplier)
        self._scale_spin.setValue(self._vm.scale)
        self._scene_change_check.setChecked(self._vm.scene_detect_enabled)
        
        # Update enabled state
        self._update_enabled_state(self._vm.interp_enabled)
    
    def _bind_viewmodel(self) -> None:
        """Establish bidirectional binding with ViewModel."""
        # ViewModel → UI (with blockSignals to prevent feedback loops)
        self._vm.interp_enabled_changed.connect(self._on_interp_enabled_changed)
        self._vm.model_type_changed.connect(self._on_model_type_changed)
        self._vm.checkpoint_changed.connect(self._on_checkpoint_changed)
        self._vm.multiplier_changed.connect(self._on_multiplier_changed)
        self._vm.scale_changed.connect(self._on_scale_changed)
        self._vm.scene_detect_enabled_changed.connect(self._on_scene_detect_enabled_changed)
        self._vm.available_models_changed.connect(self._on_available_models_changed)
        self._vm.available_checkpoints_changed.connect(self._on_available_checkpoints_changed)
        
        # UI → ViewModel (direct setter calls)
        self._enabled_check.checkStateChanged.connect(self._on_enabled_check_changed)
        self._model_combo.currentTextChanged.connect(self._on_model_combo_changed)
        self._checkpoint_combo.currentTextChanged.connect(self._on_checkpoint_combo_changed)
        self._multiplier_combo.currentIndexChanged.connect(self._on_multiplier_combo_changed)
        self._scale_spin.valueChanged.connect(self._on_scale_spin_changed)
        self._scene_change_check.checkStateChanged.connect(self._on_scene_change_check_changed)
    
    # ====================
    # ViewModel → UI handlers (with blockSignals)
    # ====================
    
    def _on_interp_enabled_changed(self, enabled: bool) -> None:
        """Handle interp_enabled signal from ViewModel."""
        self._enabled_check.blockSignals(True)
        self._enabled_check.setChecked(enabled)
        self._enabled_check.blockSignals(False)
        self._update_enabled_state(enabled)
    
    def _on_model_type_changed(self, model_type: str) -> None:
        """Handle model_type signal from ViewModel."""
        self._set_current_model(model_type)
    
    def _on_checkpoint_changed(self, checkpoint: str) -> None:
        """Handle checkpoint signal from ViewModel."""
        self._set_current_checkpoint(checkpoint)
    
    def _on_multiplier_changed(self, multiplier: int) -> None:
        """Handle multiplier signal from ViewModel."""
        self._set_current_multiplier(multiplier)
    
    def _on_scale_changed(self, scale: float) -> None:
        """Handle scale signal from ViewModel."""
        self._scale_spin.blockSignals(True)
        self._scale_spin.setValue(scale)
        self._scale_spin.blockSignals(False)
    
    def _on_scene_detect_enabled_changed(self, enabled: bool) -> None:
        """Handle scene_detect_enabled signal from ViewModel."""
        self._scene_change_check.blockSignals(True)
        self._scene_change_check.setChecked(enabled)
        self._scene_change_check.blockSignals(False)
    
    def _on_available_models_changed(self, models: List[str]) -> None:
        """Handle available_models signal from ViewModel."""
        self._populate_models(models)
    
    def _on_available_checkpoints_changed(self, checkpoints: List[str]) -> None:
        """Handle available_checkpoints signal from ViewModel."""
        self._populate_checkpoints(checkpoints)
    
    # ====================
    # UI → ViewModel handlers
    # ====================
    
    def _on_enabled_check_changed(self, state: Qt.CheckState) -> None:
        """Handle enable checkbox state change."""
        enabled = state == Qt.CheckState.Checked
        self._vm.set_interp_enabled(enabled)
        self._update_enabled_state(enabled)
    
    def _on_model_combo_changed(self, model_type: str) -> None:
        """Handle model combo selection change."""
        if model_type:
            self._vm.set_model_type(model_type)
    
    def _on_checkpoint_combo_changed(self, checkpoint: str) -> None:
        """Handle checkpoint combo selection change."""
        if checkpoint:
            self._vm.set_checkpoint(checkpoint)
    
    def _on_multiplier_combo_changed(self, index: int) -> None:
        """Handle multiplier combo index change."""
        # Map index to multiplier value: 0→2, 1→4, 2→8
        multiplier_values = [2, 4, 8]
        if 0 <= index < len(multiplier_values):
            self._vm.set_multiplier(multiplier_values[index])
    
    def _on_scale_spin_changed(self, value: float) -> None:
        """Handle scale spinbox value change."""
        self._vm.set_scale(value)
    
    def _on_scene_change_check_changed(self, state: Qt.CheckState) -> None:
        """Handle scene change checkbox state change."""
        enabled = state == Qt.CheckState.Checked
        self._vm.set_scene_detect_enabled(enabled)
    
    # ====================
    # Helper methods
    # ====================
    
    def _populate_models(self, models: List[str]) -> None:
        """Populate model combo box with available models."""
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        
        for model in models:
            # Try to get display name from ViewModel
            display_name = self._vm.get_model_display_name(model)
            self._model_combo.addItem(display_name, model)
        
        # Restore current selection
        self._set_current_model(self._vm.model_type)
        self._model_combo.blockSignals(False)
    
    def _populate_checkpoints(self, checkpoints: List[str]) -> None:
        """Populate checkpoint combo box with available checkpoints."""
        self._checkpoint_combo.blockSignals(True)
        self._checkpoint_combo.clear()
        self._checkpoint_combo.addItems(checkpoints)
        
        # Restore current selection
        self._set_current_checkpoint(self._vm.checkpoint)
        self._checkpoint_combo.blockSignals(False)
    
    def _set_current_model(self, model_type: str) -> None:
        """Set current model in combo without triggering signals."""
        self._model_combo.blockSignals(True)
        
        # Find by data value (actual model type)
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == model_type:
                self._model_combo.setCurrentIndex(i)
                break
        
        self._model_combo.blockSignals(False)
    
    def _set_current_checkpoint(self, checkpoint: str) -> None:
        """Set current checkpoint in combo without triggering signals."""
        self._checkpoint_combo.blockSignals(True)
        
        index = self._checkpoint_combo.findText(checkpoint)
        if index >= 0:
            self._checkpoint_combo.setCurrentIndex(index)
        
        self._checkpoint_combo.blockSignals(False)
    
    def _set_current_multiplier(self, multiplier: int) -> None:
        """Set current multiplier index in combo without triggering signals."""
        self._multiplier_combo.blockSignals(True)
        
        # Map multiplier value to index: 2→0, 4→1, 8→2
        multiplier_map = {2: 0, 4: 1, 8: 2}
        if multiplier in multiplier_map:
            self._multiplier_combo.setCurrentIndex(multiplier_map[multiplier])
        
        self._multiplier_combo.blockSignals(False)
    
    def _update_enabled_state(self, enabled: bool) -> None:
        """Update enabled/disabled state of child widgets."""
        self._model_combo.setEnabled(enabled)
        self._checkpoint_combo.setEnabled(enabled)
        self._multiplier_combo.setEnabled(enabled)
        self._scale_spin.setEnabled(enabled)
        # Scene change checkbox is independent of interpolation enabled
    
    # ====================
    # i18n support
    # ====================
    
    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle change events including language change."""
        if a0 and a0.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(a0)
    
    def retranslate_ui(self) -> None:
        """Update all UI strings for translation."""
        self.setTitle(self.tr("Interpolation"))
        self._enabled_check.setText(self.tr("Enable Interpolation"))
        self._model_combo.setToolTip(self.tr("Select interpolation model type"))
        self._checkpoint_combo.setToolTip(self.tr("Select model checkpoint/version"))
        self._multiplier_combo.setToolTip(self.tr("Frame multiplier for interpolation"))
        self._scale_spin.setToolTip(self.tr("Scale factor for model inference"))
        self._scene_change_check.setText(self.tr("Scene Change Detection"))
        self._scene_change_check.setToolTip(
            self.tr("Enable scene change detection for better interpolation")
        )


__all__ = ["InterpolationGroup"]