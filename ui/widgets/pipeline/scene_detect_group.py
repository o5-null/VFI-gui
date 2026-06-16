"""SceneDetectGroup - Scene detection settings widget.

A QGroupBox for configuring scene detection settings.
Bidirectional binding with PipelineViewModel.
"""

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox

from ui.styles.theme import Theme
from ui.viewmodels.pipeline_viewmodel import PipelineViewModel


class SceneDetectGroup(QGroupBox):
    """Scene detection settings widget with ViewModel binding.
    
    Features:
        - Enable/disable toggle for scene detection
        - Method selection (planestats/neural/vapoursynth)
        - Threshold adjustment (0.0 - 1.0)
    
    All widgets use QFormLayout for clean alignment.
    Bidirectional binding ensures UI ↔ ViewModel synchronization.
    """

    # Available detection methods
    DETECTION_METHODS = [
        ("neural", "Neural Network"),
        ("planestats", "Plane Statistics"),
        ("vapoursynth", "VapourSynth Built-in"),
    ]

    def __init__(self, vm: PipelineViewModel, parent=None):
        """Initialize SceneDetectGroup.
        
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
        self.setTitle(self.tr("Scene Detection"))
        
        layout = QFormLayout(self)
        layout.setSpacing(Theme.SPACING_MD)
        layout.setContentsMargins(
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
        )
        
        # Enable/disable checkbox
        self._enabled_check = QCheckBox(self.tr("Enable Scene Detection"))
        layout.addRow(self._enabled_check)
        
        # Method selection
        self._method_combo = QComboBox()
        self._method_combo.setToolTip(self.tr("Select scene detection method"))
        
        # Populate methods
        for method_id, method_name in self.DETECTION_METHODS:
            self._method_combo.addItem(method_name, method_id)
        
        layout.addRow(self.tr("Method:"), self._method_combo)
        
        # Threshold adjustment
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(0.5)
        self._threshold_spin.setToolTip(
            self.tr("Detection threshold (higher = more sensitive)")
        )
        layout.addRow(self.tr("Threshold:"), self._threshold_spin)
        
        # Set initial state from ViewModel
        self._enabled_check.setChecked(self._vm.scene_detect_enabled)
        self._set_current_method(self._vm.scene_detect_method)
        self._threshold_spin.setValue(self._vm.scene_detect_threshold)
        
        # Update enabled state
        self._update_enabled_state(self._vm.scene_detect_enabled)
    
    def _bind_viewmodel(self) -> None:
        """Establish bidirectional binding with ViewModel."""
        # ViewModel → UI (with blockSignals to prevent feedback loops)
        self._vm.scene_detect_enabled_changed.connect(self._on_scene_detect_enabled_changed)
        self._vm.scene_detect_method_changed.connect(self._on_scene_detect_method_changed)
        self._vm.scene_detect_threshold_changed.connect(self._on_scene_detect_threshold_changed)
        
        # UI → ViewModel (direct setter calls)
        self._enabled_check.checkStateChanged.connect(self._on_enabled_check_changed)
        self._method_combo.currentIndexChanged.connect(self._on_method_combo_changed)
        self._threshold_spin.valueChanged.connect(self._on_threshold_spin_changed)
    
    # ====================
    # ViewModel → UI handlers (with blockSignals)
    # ====================
    
    def _on_scene_detect_enabled_changed(self, enabled: bool) -> None:
        """Handle scene_detect_enabled signal from ViewModel."""
        self._enabled_check.blockSignals(True)
        self._enabled_check.setChecked(enabled)
        self._enabled_check.blockSignals(False)
        self._update_enabled_state(enabled)
    
    def _on_scene_detect_method_changed(self, method: str) -> None:
        """Handle scene_detect_method signal from ViewModel."""
        self._set_current_method(method)
    
    def _on_scene_detect_threshold_changed(self, threshold: float) -> None:
        """Handle scene_detect_threshold signal from ViewModel."""
        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(threshold)
        self._threshold_spin.blockSignals(False)
    
    # ====================
    # UI → ViewModel handlers
    # ====================
    
    def _on_enabled_check_changed(self, state: Qt.CheckState) -> None:
        """Handle enable checkbox state change."""
        enabled = state == Qt.CheckState.Checked
        self._vm.set_scene_detect_enabled(enabled)
        self._update_enabled_state(enabled)
    
    def _on_method_combo_changed(self, index: int) -> None:
        """Handle method combo index change."""
        if index >= 0:
            method_id = self._method_combo.itemData(index)
            if method_id:
                self._vm.set_scene_detect_method(method_id)
    
    def _on_threshold_spin_changed(self, value: float) -> None:
        """Handle threshold spinbox value change."""
        self._vm.set_scene_detect_threshold(value)
    
    # ====================
    # Helper methods
    # ====================
    
    def _set_current_method(self, method: str) -> None:
        """Set current method in combo without triggering signals."""
        self._method_combo.blockSignals(True)
        
        # Find by data value (method ID)
        for i in range(self._method_combo.count()):
            if self._method_combo.itemData(i) == method:
                self._method_combo.setCurrentIndex(i)
                break
        
        # If not found, default to neural (index 0)
        if self._method_combo.currentIndex() < 0:
            self._method_combo.setCurrentIndex(0)
        
        self._method_combo.blockSignals(False)
    
    def _update_enabled_state(self, enabled: bool) -> None:
        """Update enabled/disabled state of child widgets."""
        self._method_combo.setEnabled(enabled)
        self._threshold_spin.setEnabled(enabled)
    
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
        self.setTitle(self.tr("Scene Detection"))
        self._enabled_check.setText(self.tr("Enable Scene Detection"))
        self._method_combo.setToolTip(self.tr("Select scene detection method"))
        self._threshold_spin.setToolTip(
            self.tr("Detection threshold (higher = more sensitive)")
        )
        
        # Rebuild method combo with translated names
        current_index = self._method_combo.currentIndex()
        self._method_combo.blockSignals(True)
        self._method_combo.clear()
        
        for method_id, method_name in self.DETECTION_METHODS:
            self._method_combo.addItem(self.tr(method_name), method_id)
        
        if current_index >= 0 and current_index < self._method_combo.count():
            self._method_combo.setCurrentIndex(current_index)
        
        self._method_combo.blockSignals(False)


__all__ = ["SceneDetectGroup"]