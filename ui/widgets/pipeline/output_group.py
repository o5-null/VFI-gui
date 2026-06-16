"""OutputGroup - Output codec settings widget.

A QGroupBox for configuring output codec and encoding settings.
Bidirectional binding with both PipelineViewModel and CodecViewModel.
"""

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QSpinBox

from ui.styles.theme import Theme
from ui.viewmodels.codec_viewmodel import CodecViewModel
from ui.viewmodels.pipeline_viewmodel import PipelineViewModel


class OutputGroup(QGroupBox):
    """Output settings widget with dual ViewModel binding.
    
    Features:
        - Codec selection (populated from CodecViewModel)
        - Quality adjustment (CRF/CQ value range)
        - Preset selection (populated based on selected codec)
        - Audio copy toggle
    
    All widgets use QFormLayout for clean alignment.
    Bidirectional binding ensures UI ↔ ViewModel synchronization.
    
    Note: This group requires both PipelineViewModel and CodecViewModel
    because output settings are split across both ViewModels.
    """

    def __init__(
        self,
        pipeline_vm: PipelineViewModel,
        codec_vm: CodecViewModel,
        parent=None,
    ):
        """Initialize OutputGroup.
        
        Args:
            pipeline_vm: PipelineViewModel instance (for reference, not directly used)
            codec_vm: CodecViewModel instance for codec settings binding
            parent: Parent widget
        """
        super().__init__(parent)
        self._pipeline_vm = pipeline_vm  # Kept for future extensibility
        self._codec_vm = codec_vm
        
        # Initialize and bind widgets
        self._setup_ui()
        self._bind_viewmodel()
    
    def _setup_ui(self) -> None:
        """Create and arrange widgets using QFormLayout."""
        self.setTitle(self.tr("Output"))
        
        layout = QFormLayout(self)
        layout.setSpacing(Theme.SPACING_MD)
        layout.setContentsMargins(
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
        )
        
        # Codec selection
        self._codec_combo = QComboBox()
        self._codec_combo.setToolTip(self.tr("Select output codec"))
        layout.addRow(self.tr("Codec:"), self._codec_combo)
        
        # Quality adjustment (CRF/CQ values)
        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(0, 51)
        self._quality_spin.setValue(22)  # Default CRF
        self._quality_spin.setToolTip(
            self.tr("Quality value (0=best, 51=worst for most codecs)")
        )
        layout.addRow(self.tr("Quality:"), self._quality_spin)
        
        # Preset selection
        self._preset_combo = QComboBox()
        self._preset_combo.setToolTip(self.tr("Encoding preset (speed vs quality)"))
        layout.addRow(self.tr("Preset:"), self._preset_combo)
        
        # Audio copy toggle
        self._audio_copy_check = QCheckBox(self.tr("Copy Audio Stream"))
        self._audio_copy_check.setToolTip(
            self.tr("Copy original audio stream without re-encoding")
        )
        layout.addRow(self._audio_copy_check)
        
        # Populate initial values
        self._populate_codecs(self._codec_vm.available_codecs)
        
        # Set initial state from ViewModel
        self._set_current_codec(self._codec_vm.codec)
        self._quality_spin.setValue(self._codec_vm.quality)
        self._set_current_preset(self._codec_vm.preset)
        self._audio_copy_check.setChecked(self._codec_vm.audio_copy)
    
    def _bind_viewmodel(self) -> None:
        """Establish bidirectional binding with CodecViewModel."""
        # CodecViewModel → UI (with blockSignals to prevent feedback loops)
        self._codec_vm.codec_changed.connect(self._on_codec_changed)
        self._codec_vm.quality_changed.connect(self._on_quality_changed)
        self._codec_vm.preset_changed.connect(self._on_preset_changed)
        self._codec_vm.audio_copy_changed.connect(self._on_audio_copy_changed)
        self._codec_vm.available_codecs_changed.connect(self._on_available_codecs_changed)
        
        # UI → CodecViewModel (direct setter calls)
        self._codec_combo.currentIndexChanged.connect(self._on_codec_combo_changed)
        self._quality_spin.valueChanged.connect(self._on_quality_spin_changed)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_combo_changed)
        self._audio_copy_check.checkStateChanged.connect(self._on_audio_copy_check_changed)
    
    # ====================
    # ViewModel → UI handlers (with blockSignals)
    # ====================
    
    def _on_codec_changed(self, codec: str) -> None:
        """Handle codec signal from CodecViewModel."""
        self._set_current_codec(codec)
        self._update_presets_for_codec(codec)
    
    def _on_quality_changed(self, quality: int) -> None:
        """Handle quality signal from CodecViewModel."""
        self._quality_spin.blockSignals(True)
        self._quality_spin.setValue(quality)
        self._quality_spin.blockSignals(False)
    
    def _on_preset_changed(self, preset: str) -> None:
        """Handle preset signal from CodecViewModel."""
        self._set_current_preset(preset)
    
    def _on_audio_copy_changed(self, enabled: bool) -> None:
        """Handle audio_copy signal from CodecViewModel."""
        self._audio_copy_check.blockSignals(True)
        self._audio_copy_check.setChecked(enabled)
        self._audio_copy_check.blockSignals(False)
    
    def _on_available_codecs_changed(self, codecs: list[str]) -> None:
        """Handle available_codecs signal from CodecViewModel."""
        self._populate_codecs(codecs)
    
    # ====================
    # UI → ViewModel handlers
    # ====================
    
    def _on_codec_combo_changed(self, index: int) -> None:
        """Handle codec combo index change."""
        if index >= 0:
            codec_id = self._codec_combo.itemData(index)
            if codec_id:
                self._codec_vm.set_codec(codec_id)
                self._update_presets_for_codec(codec_id)
    
    def _on_quality_spin_changed(self, value: int) -> None:
        """Handle quality spinbox value change."""
        self._codec_vm.set_quality(value)
    
    def _on_preset_combo_changed(self, index: int) -> None:
        """Handle preset combo index change."""
        if index >= 0:
            preset = self._preset_combo.itemData(index)
            if preset:
                self._codec_vm.set_preset(preset)
    
    def _on_audio_copy_check_changed(self, state: Qt.CheckState) -> None:
        """Handle audio copy checkbox state change."""
        enabled = state == Qt.CheckState.Checked
        self._codec_vm.set_audio_copy(enabled)
    
    # ====================
    # Helper methods
    # ====================
    
    def _populate_codecs(self, codecs: list[str]) -> None:
        """Populate codec combo box with available codecs."""
        self._codec_combo.blockSignals(True)
        self._codec_combo.clear()
        
        for codec_id in codecs:
            # Get display name from CodecViewModel
            codec_info = self._codec_vm.get_codec_info(codec_id)
            display_name = codec_info.get("name", codec_id)
            self._codec_combo.addItem(display_name, codec_id)
        
        # Restore current selection
        self._set_current_codec(self._codec_vm.codec)
        self._codec_combo.blockSignals(False)
        
        # Update presets for current codec
        if self._codec_vm.codec:
            self._update_presets_for_codec(self._codec_vm.codec)
    
    def _update_presets_for_codec(self, codec: str) -> None:
        """Update preset combo based on selected codec."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        
        codec_info = self._codec_vm.get_codec_info(codec)
        
        presets = codec_info.get("presets", [])
        preset_names = codec_info.get("preset_names", {})
        
        for preset in presets:
            display_name = preset_names.get(preset, preset)
            self._preset_combo.addItem(display_name, preset)
        
        # Set quality range based on codec
        quality_range = codec_info.get("quality_range", (0, 51))
        self._quality_spin.setRange(quality_range[0], quality_range[1])
        
        # Restore current preset or use default
        default_preset = codec_info.get("default_preset", "")
        current_preset = self._codec_vm.preset or default_preset
        self._set_current_preset(current_preset)
        
        self._preset_combo.blockSignals(False)
    
    def _set_current_codec(self, codec: str) -> None:
        """Set current codec in combo without triggering signals."""
        self._codec_combo.blockSignals(True)
        
        # Find by data value (codec ID)
        for i in range(self._codec_combo.count()):
            if self._codec_combo.itemData(i) == codec:
                self._codec_combo.setCurrentIndex(i)
                break
        
        self._codec_combo.blockSignals(False)
    
    def _set_current_preset(self, preset: str) -> None:
        """Set current preset in combo without triggering signals."""
        self._preset_combo.blockSignals(True)
        
        # Find by data value (preset string)
        for i in range(self._preset_combo.count()):
            if self._preset_combo.itemData(i) == preset:
                self._preset_combo.setCurrentIndex(i)
                break
        
        self._preset_combo.blockSignals(False)
    
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
        self.setTitle(self.tr("Output"))
        self._codec_combo.setToolTip(self.tr("Select output codec"))
        self._quality_spin.setToolTip(
            self.tr("Quality value (0=best, 51=worst for most codecs)")
        )
        self._preset_combo.setToolTip(self.tr("Encoding preset (speed vs quality)"))
        self._audio_copy_check.setText(self.tr("Copy Audio Stream"))
        self._audio_copy_check.setToolTip(
            self.tr("Copy original audio stream without re-encoding")
        )


__all__ = ["OutputGroup"]