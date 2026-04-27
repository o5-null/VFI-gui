"""Codec Settings Widget for VFI-gui.

Provides comprehensive UI for video encoding configuration.
Uses CodecManager for codec definitions and FFmpeg command building.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QLabel,
    QScrollArea,
    QFrame,
    QLineEdit,
    QPushButton,
    QSlider,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal

from core import tr
from core.codec_manager import (
    CodecManager,
    CodecConfig,
    CodecType,
    get_codec_manager,
)


class CodecSettingsWidget(QWidget):
    """Widget for comprehensive video encoding settings."""

    # Signals
    config_changed = pyqtSignal()
    codec_changed = pyqtSignal(str)  # New codec name

    def __init__(
        self,
        config: Optional[Any] = None,
        codec_manager: Optional[CodecManager] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._config = config
        self._codec_manager = codec_manager or get_codec_manager()
        self._setup_ui()
        self._load_defaults()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        scroll_layout.setSpacing(12)

        # Output mode group (first - most important)
        self._create_output_mode_group(scroll_layout)

        # Output path group
        self._create_output_path_group(scroll_layout)

        # Codec selection group
        self._create_codec_selection_group(scroll_layout)

        # Quality settings group
        self._create_quality_group(scroll_layout)

        # Advanced settings group
        self._create_advanced_group(scroll_layout)

        # Audio settings group
        self._create_audio_group(scroll_layout)

        # Custom parameters group
        self._create_custom_params_group(scroll_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def _create_output_mode_group(self, parent_layout: QVBoxLayout):
        """Create output mode selection group."""
        group = QGroupBox(tr("Output Mode"))
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Output mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel(tr("Output:"))
        mode_layout.addWidget(mode_label)

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem(tr("Video"), "video")
        self.output_mode_combo.addItem(tr("Image Sequence"), "images")
        self.output_mode_combo.currentIndexChanged.connect(self._on_output_mode_changed)
        mode_layout.addWidget(self.output_mode_combo, 1)
        layout.addLayout(mode_layout)

        # Image format settings (hidden by default)
        self.image_settings_widget = QWidget()
        image_layout = QHBoxLayout(self.image_settings_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(8)

        format_label = QLabel(tr("Image Format:"))
        image_layout.addWidget(format_label)

        self.image_format_combo = QComboBox()
        self.image_format_combo.addItem("PNG", "png")
        self.image_format_combo.addItem("JPEG", "jpg")
        self.image_format_combo.addItem("TIFF", "tiff")
        self.image_format_combo.addItem("EXR", "exr")
        self.image_format_combo.currentIndexChanged.connect(self._on_image_format_changed)
        image_layout.addWidget(self.image_format_combo)

        # JPEG quality
        self.jpeg_quality_label = QLabel(tr("Quality:"))
        image_layout.addWidget(self.jpeg_quality_label)

        self.image_quality_spin = QSpinBox()
        self.image_quality_spin.setRange(1, 100)
        self.image_quality_spin.setValue(95)
        self.image_quality_spin.valueChanged.connect(self._on_config_changed)
        image_layout.addWidget(self.image_quality_spin)

        self.image_settings_widget.setVisible(False)
        layout.addWidget(self.image_settings_widget)

        # Info label
        self.output_mode_info = QLabel()
        self.output_mode_info.setStyleSheet("color: #808080; font-size: 9pt;")
        self._update_output_mode_info()
        layout.addWidget(self.output_mode_info)

        parent_layout.addWidget(group)

    def _create_output_path_group(self, parent_layout: QVBoxLayout):
        """Create output path configuration group."""
        group = QGroupBox(tr("Output Path"))
        self.output_path_group = group
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Custom output directory
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(8)

        self.custom_dir_check = QCheckBox(tr("Use custom output directory"))
        self.custom_dir_check.stateChanged.connect(self._on_custom_dir_changed)
        dir_layout.addWidget(self.custom_dir_check)

        self.custom_dir_edit = QLineEdit()
        self.custom_dir_edit.setPlaceholderText(tr("Leave empty for default"))
        self.custom_dir_edit.setEnabled(False)
        self.custom_dir_edit.textChanged.connect(self._on_config_changed)
        dir_layout.addWidget(self.custom_dir_edit, 1)

        self.browse_dir_btn = QPushButton(tr("Browse..."))
        self.browse_dir_btn.setEnabled(False)
        self.browse_dir_btn.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(self.browse_dir_btn)

        layout.addLayout(dir_layout)

        # Custom subdirectory
        subdir_layout = QHBoxLayout()
        subdir_layout.setSpacing(8)

        subdir_label = QLabel(tr("Subdirectory:"))
        subdir_layout.addWidget(subdir_label)

        self.subdir_edit = QLineEdit()
        self.subdir_edit.setPlaceholderText(tr("Leave empty for default"))
        self.subdir_edit.textChanged.connect(self._on_config_changed)
        subdir_layout.addWidget(self.subdir_edit, 1)

        layout.addLayout(subdir_layout)

        # Custom filename pattern
        filename_layout = QHBoxLayout()
        filename_layout.setSpacing(8)

        filename_label = QLabel(tr("Filename pattern:"))
        filename_layout.addWidget(filename_label)

        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText(tr("e.g., {input}_4x_{backend}"))
        self.filename_edit.textChanged.connect(self._on_config_changed)
        filename_layout.addWidget(self.filename_edit, 1)

        layout.addLayout(filename_layout)

        # Placeholder hint
        hint = QLabel(tr("Placeholders: {input} = input filename, {backend} = backend type"))
        hint.setStyleSheet("color: #808080; font-size: 9pt;")
        layout.addWidget(hint)

        parent_layout.addWidget(group)

    def _on_custom_dir_changed(self, state):
        """Handle custom directory checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self.custom_dir_edit.setEnabled(enabled)
        self.browse_dir_btn.setEnabled(enabled)
        self._on_config_changed()

    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            tr("Select Output Directory"),
            self.custom_dir_edit.text() or ""
        )
        if dir_path:
            self.custom_dir_edit.setText(dir_path)

    def _on_output_mode_changed(self):
        """Handle output mode change."""
        is_images = self.output_mode_combo.currentData() == "images"
        self.image_settings_widget.setVisible(is_images)
        self.codec_selection_group.setVisible(not is_images)
        self.quality_group.setVisible(not is_images)
        self.advanced_group.setVisible(not is_images)
        self.audio_group.setVisible(not is_images)
        self._update_output_mode_info()
        self._on_config_changed()

    def _on_image_format_changed(self):
        """Handle image format change."""
        is_jpeg = self.image_format_combo.currentData() == "jpg"
        self.jpeg_quality_label.setVisible(is_jpeg)
        self.image_quality_spin.setVisible(is_jpeg)
        self._on_config_changed()

    def _update_output_mode_info(self):
        """Update output mode info label."""
        if self.output_mode_combo.currentData() == "images":
            self.output_mode_info.setText(
                tr("Output will be saved as image sequence.\n"
                   "Useful for further processing or high-quality archival.")
            )
        else:
            self.output_mode_info.setText(
                tr("Output will be encoded as video file.\n"
                   "Recommended for most use cases.")
            )

    def _create_codec_selection_group(self, parent_layout: QVBoxLayout):
        """Create codec selection group."""
        group = QGroupBox(tr("Codec Selection"))
        self.codec_selection_group = group
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Codec dropdown
        codec_label = QLabel(tr("Encoder:"))
        layout.addWidget(codec_label, 0, 0)

        self.codec_combo = QComboBox()
        self._populate_codecs()
        self.codec_combo.currentIndexChanged.connect(self._on_codec_changed)
        layout.addWidget(self.codec_combo, 0, 1, 1, 2)

        # Codec description
        self.codec_desc_label = QLabel()
        self.codec_desc_label.setWordWrap(True)
        self.codec_desc_label.setStyleSheet("color: #808080; font-size: 10pt;")
        layout.addWidget(self.codec_desc_label, 1, 0, 1, 3)

        # Hardware acceleration indicator
        self.hw_accel_label = QLabel()
        self.hw_accel_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.hw_accel_label, 2, 0, 1, 3)

        parent_layout.addWidget(group)

    def _create_quality_group(self, parent_layout: QVBoxLayout):
        """Create quality/rate control settings group."""
        self.quality_group = QGroupBox(tr("Quality Settings"))
        layout = QGridLayout(self.quality_group)
        layout.setSpacing(8)

        # Rate control mode
        rate_control_label = QLabel(tr("Rate Control:"))
        layout.addWidget(rate_control_label, 0, 0)

        self.rate_control_combo = QComboBox()
        self._populate_rate_control_modes()
        self.rate_control_combo.currentIndexChanged.connect(self._on_rate_control_changed)
        layout.addWidget(self.rate_control_combo, 0, 1, 1, 2)

        # Quality value (CRF/CQ)
        self.quality_label = QLabel(tr("Quality (CRF/CQ):"))
        layout.addWidget(self.quality_label, 1, 0)

        quality_widget = QWidget()
        quality_layout = QHBoxLayout(quality_widget)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(8)

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(0, 51)
        self.quality_slider.setValue(22)
        self.quality_slider.valueChanged.connect(self._on_quality_slider_changed)
        quality_layout.addWidget(self.quality_slider, 1)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(0, 63)
        self.quality_spin.setValue(22)
        self.quality_spin.valueChanged.connect(self._on_quality_spin_changed)
        quality_layout.addWidget(self.quality_spin)

        quality_hint = QLabel(tr("(Lower = better)"))
        quality_hint.setStyleSheet("color: #808080; font-size: 9pt;")
        quality_layout.addWidget(quality_hint)

        layout.addWidget(quality_widget, 1, 1, 1, 2)

        # Bitrate (for VBR/CBR modes)
        self.bitrate_label = QLabel(tr("Target Bitrate:"))
        layout.addWidget(self.bitrate_label, 2, 0)

        bitrate_widget = QWidget()
        bitrate_layout = QHBoxLayout(bitrate_widget)
        bitrate_layout.setContentsMargins(0, 0, 0, 0)
        bitrate_layout.setSpacing(8)

        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(100, 100000)
        self.bitrate_spin.setValue(8000)
        self.bitrate_spin.setSuffix(" kbps")
        self.bitrate_spin.valueChanged.connect(self._on_config_changed)
        bitrate_layout.addWidget(self.bitrate_spin, 1)

        self.max_bitrate_check = QCheckBox(tr("Max Bitrate:"))
        self.max_bitrate_check.stateChanged.connect(self._on_config_changed)
        bitrate_layout.addWidget(self.max_bitrate_check)

        self.max_bitrate_spin = QSpinBox()
        self.max_bitrate_spin.setRange(100, 100000)
        self.max_bitrate_spin.setValue(10000)
        self.max_bitrate_spin.setSuffix(" kbps")
        self.max_bitrate_spin.setEnabled(False)
        self.max_bitrate_spin.valueChanged.connect(self._on_config_changed)
        bitrate_layout.addWidget(self.max_bitrate_spin)

        self.max_bitrate_check.stateChanged.connect(
            lambda state: self.max_bitrate_spin.setEnabled(state == Qt.CheckState.Checked.value)
        )

        layout.addWidget(bitrate_widget, 2, 1, 1, 2)

        # Preset
        self.preset_label = QLabel(tr("Preset:"))
        layout.addWidget(self.preset_label, 3, 0)

        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._on_config_changed)
        layout.addWidget(self.preset_combo, 3, 1, 1, 2)

        parent_layout.addWidget(self.quality_group)

    def _create_advanced_group(self, parent_layout: QVBoxLayout):
        """Create advanced encoding settings group."""
        self.advanced_group = QGroupBox(tr("Advanced Settings"))
        layout = QGridLayout(self.advanced_group)
        layout.setSpacing(8)

        # Pixel format
        pixel_format_label = QLabel(tr("Pixel Format:"))
        layout.addWidget(pixel_format_label, 0, 0)

        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.addItem(tr("Auto"), "auto")
        layout.addWidget(self.pixel_format_combo, 0, 1, 1, 2)

        # Profile (for H.264/H.265)
        self.profile_label = QLabel(tr("Profile:"))
        layout.addWidget(self.profile_label, 1, 0)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem(tr("Auto"), "auto")
        self.profile_combo.currentIndexChanged.connect(self._on_config_changed)
        layout.addWidget(self.profile_combo, 1, 1, 1, 2)

        # Level (for H.264/H.265)
        self.level_label = QLabel(tr("Level:"))
        layout.addWidget(self.level_label, 2, 0)

        self.level_combo = QComboBox()
        self.level_combo.addItem(tr("Auto"), "auto")
        self.level_combo.currentIndexChanged.connect(self._on_config_changed)
        layout.addWidget(self.level_combo, 2, 1, 1, 2)

        # Keyframe interval
        keyframe_label = QLabel(tr("Keyframe Interval:"))
        layout.addWidget(keyframe_label, 3, 0)

        keyframe_widget = QWidget()
        keyframe_layout = QHBoxLayout(keyframe_widget)
        keyframe_layout.setContentsMargins(0, 0, 0, 0)
        keyframe_layout.setSpacing(8)

        self.gop_size_spin = QSpinBox()
        self.gop_size_spin.setRange(0, 10000)
        self.gop_size_spin.setValue(0)
        self.gop_size_spin.setSpecialValueText(tr("Auto"))
        self.gop_size_spin.valueChanged.connect(self._on_config_changed)
        keyframe_layout.addWidget(QLabel(tr("GOP:")))
        keyframe_layout.addWidget(self.gop_size_spin)

        self.keyint_spin = QSpinBox()
        self.keyint_spin.setRange(0, 1000)
        self.keyint_spin.setValue(0)
        self.keyint_spin.setSpecialValueText(tr("Auto"))
        self.keyint_spin.setSuffix(tr(" frames"))
        self.keyint_spin.valueChanged.connect(self._on_config_changed)
        keyframe_layout.addWidget(QLabel(tr("Keyint:")))
        keyframe_layout.addWidget(self.keyint_spin, 1)

        layout.addWidget(keyframe_widget, 3, 1, 1, 2)

        # B-frames
        self.bframes_label = QLabel(tr("B-Frames:"))
        layout.addWidget(self.bframes_label, 4, 0)

        self.bframes_spin = QSpinBox()
        self.bframes_spin.setRange(0, 16)
        self.bframes_spin.setValue(3)
        self.bframes_spin.valueChanged.connect(self._on_config_changed)
        layout.addWidget(self.bframes_spin, 4, 1, 1, 2)

        # Multi-pass encoding
        self.multipass_check = QCheckBox(tr("Enable Multi-Pass Encoding"))
        self.multipass_check.setToolTip(tr("Improves quality for VBR mode (slower encoding)"))
        self.multipass_check.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self.multipass_check, 5, 0, 1, 3)

        parent_layout.addWidget(self.advanced_group)

    def _create_audio_group(self, parent_layout: QVBoxLayout):
        """Create audio settings group."""
        self.audio_group = QGroupBox(tr("Audio Settings"))
        layout = QVBoxLayout(self.audio_group)
        layout.setSpacing(8)

        # Copy audio
        self.audio_copy_check = QCheckBox(tr("Copy Audio Stream"))
        self.audio_copy_check.setChecked(True)
        self.audio_copy_check.stateChanged.connect(self._on_audio_copy_changed)
        layout.addWidget(self.audio_copy_check)

        # Audio codec (when not copying)
        audio_codec_widget = QWidget()
        audio_codec_layout = QHBoxLayout(audio_codec_widget)
        audio_codec_layout.setContentsMargins(0, 0, 0, 0)

        self.audio_codec_label = QLabel(tr("Audio Codec:"))
        audio_codec_layout.addWidget(self.audio_codec_label)

        self.audio_codec_combo = QComboBox()
        self.audio_codec_combo.addItems(["aac", "opus", "mp3", "ac3", "flac"])
        self.audio_codec_combo.setEnabled(False)
        self.audio_codec_combo.currentTextChanged.connect(self._on_config_changed)
        audio_codec_layout.addWidget(self.audio_codec_combo, 1)

        layout.addWidget(audio_codec_widget)

        # Audio bitrate
        audio_bitrate_widget = QWidget()
        audio_bitrate_layout = QHBoxLayout(audio_bitrate_widget)
        audio_bitrate_layout.setContentsMargins(0, 0, 0, 0)

        self.audio_bitrate_label = QLabel(tr("Audio Bitrate:"))
        audio_bitrate_layout.addWidget(self.audio_bitrate_label)

        self.audio_bitrate_combo = QComboBox()
        self.audio_bitrate_combo.setEditable(True)
        self.audio_bitrate_combo.addItems(["128k", "192k", "256k", "320k", "640k"])
        self.audio_bitrate_combo.setCurrentText("192k")
        self.audio_bitrate_combo.setEnabled(False)
        self.audio_bitrate_combo.currentTextChanged.connect(self._on_config_changed)
        audio_bitrate_layout.addWidget(self.audio_bitrate_combo, 1)

        layout.addWidget(audio_bitrate_widget)

        parent_layout.addWidget(self.audio_group)

    def _create_custom_params_group(self, parent_layout: QVBoxLayout):
        """Create custom FFmpeg parameters group."""
        self.custom_group = QGroupBox(tr("Custom FFmpeg Parameters"))
        layout = QVBoxLayout(self.custom_group)
        layout.setSpacing(8)

        # Hint
        hint = QLabel(tr("Add custom FFmpeg encoding parameters (advanced users only)"))
        hint.setStyleSheet("color: #808080; font-size: 9pt;")
        layout.addWidget(hint)

        # Custom params input
        self.custom_params_edit = QLineEdit()
        self.custom_params_edit.setPlaceholderText(tr("e.g., -x265-params log-level=3:no-slow-firstpass=1"))
        self.custom_params_edit.textChanged.connect(self._on_config_changed)
        layout.addWidget(self.custom_params_edit)

        # Preset buttons
        preset_btn_layout = QHBoxLayout()

        fast_preset_btn = QPushButton(tr("Fast Encoding"))
        fast_preset_btn.clicked.connect(lambda: self._apply_param_preset("fast"))
        preset_btn_layout.addWidget(fast_preset_btn)

        quality_preset_btn = QPushButton(tr("High Quality"))
        quality_preset_btn.clicked.connect(lambda: self._apply_param_preset("quality"))
        preset_btn_layout.addWidget(quality_preset_btn)

        reset_btn = QPushButton(tr("Reset"))
        reset_btn.clicked.connect(self._reset_custom_params)
        preset_btn_layout.addWidget(reset_btn)

        preset_btn_layout.addStretch()
        layout.addLayout(preset_btn_layout)

        parent_layout.addWidget(self.custom_group)

    def _populate_codecs(self):
        """Populate codec dropdown from CodecManager."""
        self.codec_combo.clear()

        # Get codecs from manager
        hw_codecs = self._codec_manager.get_hardware_codecs()
        sw_codecs = self._codec_manager.get_software_codecs()

        # Add hardware codecs first
        if hw_codecs:
            self.codec_combo.addItem(tr("--- Hardware Encoders ---"), "")
            for codec_id, codec_info in hw_codecs.items():
                self.codec_combo.addItem(codec_info.name, codec_id)

        # Add software codecs
        if sw_codecs:
            self.codec_combo.addItem(tr("--- Software Encoders ---"), "")
            for codec_id, codec_info in sw_codecs.items():
                self.codec_combo.addItem(codec_info.name, codec_id)

        # Select first real codec
        for i in range(self.codec_combo.count()):
            if self.codec_combo.itemData(i) != "":
                self.codec_combo.setCurrentIndex(i)
                break

    def _populate_rate_control_modes(self):
        """Populate rate control modes."""
        self.rate_control_combo.clear()
        self.rate_control_combo.addItem(tr("Constant Quality (CQ)"), "cq")
        self.rate_control_combo.addItem(tr("Variable Bitrate (VBR)"), "vbr")
        self.rate_control_combo.addItem(tr("Constant Bitrate (CBR)"), "cbr")

    def _load_defaults(self):
        """Load default settings."""
        self._update_codec_ui()
        self._update_preset_options()

    def _on_codec_changed(self):
        """Handle codec selection change."""
        codec_id = self.codec_combo.currentData()
        if not codec_id:
            return

        self._update_codec_ui()
        self._update_preset_options()
        self._update_pixel_formats()

        self.codec_changed.emit(codec_id)
        self._on_config_changed()

    def _update_codec_ui(self):
        """Update UI based on selected codec."""
        codec_id = self.codec_combo.currentData()
        if not codec_id:
            return

        codec_info = self._codec_manager.get_codec_info(codec_id)
        if not codec_info:
            return

        # Update description
        self.codec_desc_label.setText(codec_info.description)

        # Update hardware indicator
        if codec_info.type == CodecType.HARDWARE:
            self.hw_accel_label.setText(tr("Hardware Acceleration: GPU Required"))
        else:
            self.hw_accel_label.setText(tr("Software Encoder (CPU-based)"))

        # Update quality range
        q_min, q_max = codec_info.quality_range
        self.quality_slider.setRange(q_min, q_max)
        self.quality_spin.setRange(q_min, q_max)

        # Set default quality
        self.quality_slider.setValue(codec_info.quality_default)
        self.quality_spin.setValue(codec_info.quality_default)

        # Update profile/level options based on codec
        self._update_profile_options(codec_id)
        self._update_level_options(codec_id)

    def _update_preset_options(self):
        """Update preset options based on selected codec."""
        codec_id = self.codec_combo.currentData()
        if not codec_id:
            return

        codec_info = self._codec_manager.get_codec_info(codec_id)
        if not codec_info:
            return

        self.preset_combo.clear()
        presets = codec_info.presets
        preset_names = codec_info.preset_names

        for preset in presets:
            display_name = preset_names.get(preset, preset)
            self.preset_combo.addItem(display_name, preset)

        # Set default preset
        default_preset = codec_info.default_preset
        index = self.preset_combo.findData(default_preset)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)

    def _update_pixel_formats(self):
        """Update pixel format options based on codec."""
        codec_id = self.codec_combo.currentData()
        if not codec_id:
            return

        codec_info = self._codec_manager.get_codec_info(codec_id)
        if not codec_info:
            return

        current_format = self.pixel_format_combo.currentData()
        self.pixel_format_combo.clear()
        self.pixel_format_combo.addItem(tr("Auto"), "auto")

        for fmt in codec_info.pixel_formats:
            self.pixel_format_combo.addItem(fmt, fmt)

        # Try to restore previous selection
        index = self.pixel_format_combo.findData(current_format)
        if index >= 0:
            self.pixel_format_combo.setCurrentIndex(index)

    def _update_profile_options(self, codec_id: str):
        """Update profile options based on codec."""
        self.profile_combo.clear()
        self.profile_combo.addItem(tr("Auto"), "auto")

        if codec_id in ["hevc_nvenc", "libx265"]:
            profiles = ["main", "main10", "mainstillpicture", "rext", "scc"]
            for profile in profiles:
                self.profile_combo.addItem(profile.upper(), profile)
        elif codec_id in ["libx264"]:
            profiles = ["baseline", "main", "high", "high10", "high422", "high444"]
            for profile in profiles:
                self.profile_combo.addItem(profile.capitalize(), profile)

    def _update_level_options(self, codec_id: str):
        """Update level options based on codec."""
        self.level_combo.clear()
        self.level_combo.addItem(tr("Auto"), "auto")

        if codec_id in ["hevc_nvenc", "libx265"]:
            levels = ["1", "2", "2.1", "3", "3.1", "4", "4.1", "5", "5.1", "5.2", "6", "6.1", "6.2"]
            for level in levels:
                self.level_combo.addItem(f"Level {level}", level)
        elif codec_id in ["libx264"]:
            levels = ["1", "1b", "1.1", "1.2", "1.3", "2", "2.1", "2.2", "3", "3.1", "3.2", "4", "4.1", "4.2", "5", "5.1", "5.2"]
            for level in levels:
                self.level_combo.addItem(f"Level {level}", level)

    def _on_rate_control_changed(self):
        """Handle rate control mode change."""
        mode = self.rate_control_combo.currentData()

        # Show/hide quality vs bitrate based on mode
        is_quality_mode = mode in ["cq", "crf", "cqp"]
        is_bitrate_mode = mode in ["vbr", "cbr"]

        self.quality_label.setVisible(is_quality_mode)
        self.quality_slider.parentWidget().setVisible(is_quality_mode)

        self.bitrate_label.setVisible(is_bitrate_mode)
        self.bitrate_spin.parentWidget().setVisible(is_bitrate_mode)

        self._on_config_changed()

    def _on_quality_slider_changed(self, value: int):
        """Handle quality slider change."""
        self.quality_spin.blockSignals(True)
        self.quality_spin.setValue(value)
        self.quality_spin.blockSignals(False)
        self._on_config_changed()

    def _on_quality_spin_changed(self, value: int):
        """Handle quality spin change."""
        self.quality_slider.blockSignals(True)
        self.quality_slider.setValue(value)
        self.quality_slider.blockSignals(False)
        self._on_config_changed()

    def _on_audio_copy_changed(self, state: int):
        """Handle audio copy checkbox change."""
        is_copy = state == Qt.CheckState.Checked.value
        self.audio_codec_combo.setEnabled(not is_copy)
        self.audio_bitrate_combo.setEnabled(not is_copy)
        self._on_config_changed()

    def _apply_param_preset(self, preset_type: str):
        """Apply a predefined parameter preset."""
        codec_id = self.codec_combo.currentData()

        if preset_type == "fast":
            # Fast encoding preset
            if codec_id in ["hevc_nvenc", "av1_nvenc"]:
                self.preset_combo.setCurrentIndex(0)  # P1 or fastest
            elif codec_id in ["libx265", "libx264"]:
                index = self.preset_combo.findData("veryfast")
                if index >= 0:
                    self.preset_combo.setCurrentIndex(index)
            elif codec_id == "libaom-av1":
                self.preset_combo.setCurrentIndex(5)  # Much faster
            self.quality_slider.setValue(28)  # Lower quality but faster

        elif preset_type == "quality":
            # High quality preset
            if codec_id in ["hevc_nvenc", "av1_nvenc"]:
                self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)  # P7 or slowest
            elif codec_id in ["libx265", "libx264"]:
                index = self.preset_combo.findData("slow")
                if index >= 0:
                    self.preset_combo.setCurrentIndex(index)
            elif codec_id == "libaom-av1":
                self.preset_combo.setCurrentIndex(1)  # Better quality
            self.quality_slider.setValue(18)  # Higher quality

    def _reset_custom_params(self):
        """Reset custom parameters."""
        self.custom_params_edit.clear()

    def _on_config_changed(self):
        """Handle configuration change."""
        self.config_changed.emit()

    def retranslate_ui(self):
        """Retranslate UI elements after language change."""
        # Repopulate codecs with translated strings
        self._populate_codecs()
        self._populate_rate_control_modes()
        self._update_codec_ui()

    def load_config(self, config: Dict[str, Any]):
        """Load configuration into UI."""
        # Load output mode
        output_mode = config.get("output_mode", "video")
        for i in range(self.output_mode_combo.count()):
            if self.output_mode_combo.itemData(i) == output_mode:
                self.output_mode_combo.setCurrentIndex(i)
                break

        # Load image format settings
        image_format = config.get("image_format", "png")
        for i in range(self.image_format_combo.count()):
            if self.image_format_combo.itemData(i) == image_format:
                self.image_format_combo.setCurrentIndex(i)
                break

        image_quality = config.get("image_quality", 95)
        self.image_quality_spin.setValue(image_quality)

        # Load codec
        codec = config.get("codec", "hevc_nvenc")
        for i in range(self.codec_combo.count()):
            if self.codec_combo.itemData(i) == codec:
                self.codec_combo.setCurrentIndex(i)
                break

        # Load rate control
        rate_control = config.get("rate_control", "cq")
        for i in range(self.rate_control_combo.count()):
            if self.rate_control_combo.itemData(i) == rate_control:
                self.rate_control_combo.setCurrentIndex(i)
                break

        # Load quality
        quality = config.get("quality", 22)
        self.quality_slider.setValue(quality)
        self.quality_spin.setValue(quality)

        # Load bitrate
        bitrate = config.get("bitrate", 8000)
        self.bitrate_spin.setValue(bitrate)
        max_bitrate = config.get("max_bitrate") or 10000  # Handle None
        self.max_bitrate_spin.setValue(max_bitrate)
        use_max_bitrate = config.get("use_max_bitrate", False)
        self.max_bitrate_check.setChecked(use_max_bitrate)

        # Load preset
        preset = config.get("preset", "")
        if preset:
            for i in range(self.preset_combo.count()):
                if self.preset_combo.itemData(i) == preset:
                    self.preset_combo.setCurrentIndex(i)
                    break

        # Load advanced settings
        pixel_format = config.get("pixel_format", "auto")
        for i in range(self.pixel_format_combo.count()):
            if self.pixel_format_combo.itemData(i) == pixel_format:
                self.pixel_format_combo.setCurrentIndex(i)
                break

        profile = config.get("profile", "auto")
        for i in range(self.profile_combo.count()):
            if self.profile_combo.itemData(i) == profile:
                self.profile_combo.setCurrentIndex(i)
                break

        level = config.get("level", "auto")
        for i in range(self.level_combo.count()):
            if self.level_combo.itemData(i) == level:
                self.level_combo.setCurrentIndex(i)
                break

        gop_size = config.get("gop_size", 0)
        self.gop_size_spin.setValue(gop_size)

        keyint = config.get("keyint", 0)
        self.keyint_spin.setValue(keyint)

        bframes = config.get("bframes", 3)
        self.bframes_spin.setValue(bframes)

        multipass = config.get("multipass", False)
        self.multipass_check.setChecked(multipass)

        # Load audio settings
        audio_copy = config.get("audio_copy", True)
        self.audio_copy_check.setChecked(audio_copy)

        audio_codec = config.get("audio_codec", "aac")
        index = self.audio_codec_combo.findText(audio_codec)
        if index >= 0:
            self.audio_codec_combo.setCurrentIndex(index)

        audio_bitrate = config.get("audio_bitrate", "192k")
        self.audio_bitrate_combo.setCurrentText(audio_bitrate)

        # Load custom parameters
        custom_params = config.get("custom_params", "")
        self.custom_params_edit.setText(custom_params)

        # Load output path settings
        output_dir = config.get("output_dir", "")
        self.custom_dir_check.setChecked(bool(output_dir))
        self.custom_dir_edit.setText(output_dir)

        output_subdir = config.get("output_subdir", "")
        self.subdir_edit.setText(output_subdir)

        output_filename = config.get("output_filename", "")
        self.filename_edit.setText(output_filename)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration from UI."""
        return {
            "output_mode": self.output_mode_combo.currentData() or "video",
            "image_format": self.image_format_combo.currentData() or "png",
            "image_quality": self.image_quality_spin.value(),
            "codec": self.codec_combo.currentData() or "hevc_nvenc",
            "rate_control": self.rate_control_combo.currentData() or "cq",
            "quality": self.quality_spin.value(),
            "bitrate": self.bitrate_spin.value(),
            "max_bitrate": self.max_bitrate_spin.value() if self.max_bitrate_check.isChecked() else None,
            "use_max_bitrate": self.max_bitrate_check.isChecked(),
            "preset": self.preset_combo.currentData() or "",
            "pixel_format": self.pixel_format_combo.currentData() or "auto",
            "profile": self.profile_combo.currentData() or "auto",
            "level": self.level_combo.currentData() or "auto",
            "gop_size": self.gop_size_spin.value(),
            "keyint": self.keyint_spin.value(),
            "bframes": self.bframes_spin.value(),
            "multipass": self.multipass_check.isChecked(),
            "audio_copy": self.audio_copy_check.isChecked(),
            "audio_codec": self.audio_codec_combo.currentText(),
            "audio_bitrate": self.audio_bitrate_combo.currentText(),
            "custom_params": self.custom_params_edit.text(),
            # Output path settings
            "output_dir": self.custom_dir_edit.text() if self.custom_dir_check.isChecked() else "",
            "output_subdir": self.subdir_edit.text(),
            "output_filename": self.filename_edit.text(),
        }

    def get_codec_config(self) -> CodecConfig:
        """Get current configuration as CodecConfig object."""
        return CodecConfig.from_dict(self.get_config())
