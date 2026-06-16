"""UpscalingGroup - Upscaling settings widget.

A QGroupBox for configuring video upscaling settings.
Bidirectional binding with PipelineViewModel.
"""

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox

from ui.styles.theme import Theme
from ui.viewmodels.pipeline_viewmodel import PipelineViewModel


class UpscalingGroup(QGroupBox):
    """Upscaling settings widget with ViewModel binding.
    
    Features:
        - Enable/disable toggle for upscaling
        - Engine selection (populated from available engines)
        - Placeholder for future settings (tile size, overlap, etc.)
    
    All widgets use QFormLayout for clean alignment.
    Bidirectional binding ensures UI ↔ ViewModel synchronization.
    """

    # Available upscaling engines (static for now, can be dynamic later)
    AVAILABLE_ENGINES = [
        ("esrgan", "ESRGAN"),
        ("realcugan", "Real-CUGAN"),
        ("waifu2x", "Waifu2x"),
    ]

    def __init__(self, vm: PipelineViewModel, parent=None):
        """Initialize UpscalingGroup.
        
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
        self.setTitle(self.tr("Upscaling"))
        
        layout = QFormLayout(self)
        layout.setSpacing(Theme.SPACING_MD)
        layout.setContentsMargins(
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
        )
        
        # Enable/disable checkbox
        self._enabled_check = QCheckBox(self.tr("Enable Upscaling"))
        layout.addRow(self._enabled_check)
        
        # Engine selection
        self._engine_combo = QComboBox()
        self._engine_combo.setToolTip(self.tr("Select upscaling engine"))
        
        # Populate engines
        for engine_id, engine_name in self.AVAILABLE_ENGINES:
            self._engine_combo.addItem(engine_name, engine_id)
        
        layout.addRow(self.tr("Engine:"), self._engine_combo)
        
        # Set initial state from ViewModel
        self._enabled_check.setChecked(self._vm.upscale_enabled)
        self._set_current_engine(self._vm.upscale_engine)
        
        # Update enabled state
        self._update_enabled_state(self._vm.upscale_enabled)
    
    def _bind_viewmodel(self) -> None:
        """Establish bidirectional binding with ViewModel."""
        # ViewModel → UI (with blockSignals to prevent feedback loops)
        self._vm.upscale_enabled_changed.connect(self._on_upscale_enabled_changed)
        self._vm.upscale_engine_changed.connect(self._on_upscale_engine_changed)
        
        # UI → ViewModel (direct setter calls)
        self._enabled_check.checkStateChanged.connect(self._on_enabled_check_changed)
        self._engine_combo.currentIndexChanged.connect(self._on_engine_combo_changed)
    
    # ====================
    # ViewModel → UI handlers (with blockSignals)
    # ====================
    
    def _on_upscale_enabled_changed(self, enabled: bool) -> None:
        """Handle upscale_enabled signal from ViewModel."""
        self._enabled_check.blockSignals(True)
        self._enabled_check.setChecked(enabled)
        self._enabled_check.blockSignals(False)
        self._update_enabled_state(enabled)
    
    def _on_upscale_engine_changed(self, engine: str) -> None:
        """Handle upscale_engine signal from ViewModel."""
        self._set_current_engine(engine)
    
    # ====================
    # UI → ViewModel handlers
    # ====================
    
    def _on_enabled_check_changed(self, state: Qt.CheckState) -> None:
        """Handle enable checkbox state change."""
        enabled = state == Qt.CheckState.Checked
        self._vm.set_upscale_enabled(enabled)
        self._update_enabled_state(enabled)
    
    def _on_engine_combo_changed(self, index: int) -> None:
        """Handle engine combo index change."""
        if index >= 0:
            engine_id = self._engine_combo.itemData(index)
            if engine_id:
                self._vm.set_upscale_engine(engine_id)
    
    # ====================
    # Helper methods
    # ====================
    
    def _set_current_engine(self, engine: str) -> None:
        """Set current engine in combo without triggering signals."""
        self._engine_combo.blockSignals(True)
        
        # Find by data value (engine ID)
        for i in range(self._engine_combo.count()):
            if self._engine_combo.itemData(i) == engine:
                self._engine_combo.setCurrentIndex(i)
                break
        
        self._engine_combo.blockSignals(False)
    
    def _update_enabled_state(self, enabled: bool) -> None:
        """Update enabled/disabled state of child widgets."""
        self._engine_combo.setEnabled(enabled)
    
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
        self.setTitle(self.tr("Upscaling"))
        self._enabled_check.setText(self.tr("Enable Upscaling"))
        self._engine_combo.setToolTip(self.tr("Select upscaling engine"))
        
        # Rebuild engine combo with translated names
        current_index = self._engine_combo.currentIndex()
        self._engine_combo.blockSignals(True)
        self._engine_combo.clear()
        
        # Engine names can be translated if needed
        for engine_id, engine_name in self.AVAILABLE_ENGINES:
            self._engine_combo.addItem(self.tr(engine_name), engine_id)
        
        if current_index >= 0 and current_index < self._engine_combo.count():
            self._engine_combo.setCurrentIndex(current_index)
        
        self._engine_combo.blockSignals(False)


__all__ = ["UpscalingGroup"]