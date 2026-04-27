"""Pipeline configuration widget for RIFE, ESRGAN, and Scene Detection settings."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QTabWidget,
    QScrollArea,
    QFrame,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal

from core import tr
from core.model_selection import ModelSelectionManager
from core.device_type import DeviceType
from core.device_manager import device_manager
from ui.widgets.codec_settings import CodecSettingsWidget
from ui.widgets.benchmark_dialog import BenchmarkDialog

if TYPE_CHECKING:
    from core.config import Config


class PipelineConfigWidget(QWidget):
    """Widget for configuring the video processing pipeline."""

    config_changed = pyqtSignal()

    # Interpolation multipliers
    INTERP_MULTIPLIERS = [2, 4, 8]

    # Scene detection models
    SCENE_DETECT_MODELS = {
        0: "EfficientFormerV2-S0 (224px)",
        1: "EfficientFormerV2-S0 + RIFE46 Flow",
        2: "EfficientNetV2-B0 (256px)",
        3: "EfficientNetV2-B0 + RIFE46 Flow",
        4: "SwinV2-Small (256px)",
        5: "SwinV2-Small + RIFE46 Flow",
        6: "EfficientNetV2-B0 (48x27)",
        7: "DaViT-Small (256px) - 30k",
        8: "DaViT-Small (256px) - 40k",
        9: "MaxViTV2-Nano (256px) - 20k",
        10: "MaxViTV2-Nano (256px) - 30k",
        11: "MaxViTV2-Base (224px)",
        12: "MobileViTV2 + RIFE422 + Sobel (Recommended)",
        13: "AutoShot (5 images)",
        14: "Shift-LPIPS-Alex",
        15: "Shift-LPIPS-VGG",
        16: "DISTS",
    }

    def __init__(
        self,
        config: "Config",
        model_selection_manager: Optional["ModelSelectionManager"] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._config = config
        
        # Create or use provided model selection manager
        if model_selection_manager is None:
            self._model_selection = ModelSelectionManager(config)
        else:
            self._model_selection = model_selection_manager
        
        # Connect to model selection signals
        self._model_selection.model_type_changed.connect(self._on_external_model_type_changed)
        self._model_selection.checkpoint_changed.connect(self._on_external_checkpoint_changed)
        self._model_selection.available_models_changed.connect(self._on_available_models_changed)
        
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Initialize device list (using device_manager singleton)
        self._available_devices = []

        # Tab widget for different settings sections
        self.tab_widget = QTabWidget()

        # Interpolation tab
        interp_tab = self._create_interpolation_tab()
        self.tab_widget.addTab(interp_tab, tr("Interpolation"))

        # Upscaling tab
        upscale_tab = self._create_upscaling_tab()
        self.tab_widget.addTab(upscale_tab, tr("Upscaling"))

        # Scene Detection tab
        scene_tab = self._create_scene_detection_tab()
        self.tab_widget.addTab(scene_tab, tr("Scene Detect"))

        # Device tab
        device_tab = self._create_device_tab()
        self.tab_widget.addTab(device_tab, tr("Device"))

        # Output tab
        output_tab = self._create_output_tab()
        self.tab_widget.addTab(output_tab, tr("Output"))

        layout.addWidget(self.tab_widget)
    
    def retranslate_ui(self):
        """Retranslate UI elements after language change."""
        self.tab_widget.setTabText(0, tr("Interpolation"))
        self.tab_widget.setTabText(1, tr("Upscaling"))
        self.tab_widget.setTabText(2, tr("Scene Detect"))
        self.tab_widget.setTabText(3, tr("Device"))
        self.tab_widget.setTabText(4, tr("Output"))
        
        # Interpolation tab
        self.interp_enabled.setText(tr("Enable RIFE Interpolation"))
        self.interp_settings_group.setTitle(tr("Interpolation Settings"))
        self.interp_type_label.setText(tr("Model:"))
        self.interp_model_label.setText(tr("Model:"))
        self.interp_multi_label.setText(tr("Multiplier:"))
        self.interp_scale_label.setText(tr("Scale:"))
        self.interp_scene_change.setText(tr("Enable Scene Change Detection (sc=True)"))
        self.no_models_label.setText(tr("No models installed. Go to Models tab to download."))
        
        # Upscaling tab
        self.upscale_enabled.setText(tr("Enable Upscaling"))
        self.upscale_settings_group.setTitle(tr("Upscaling Settings"))
        self.upscale_engine_label.setText(tr("Engine:"))
        self.upscale_tile_label.setText(tr("Tile Size:"))
        self.upscale_overlap_label.setText(tr("Overlap:"))
        self.upscale_streams_label.setText(tr("GPU Streams:"))
        
        # Scene Detection tab
        self.scene_enabled.setText(tr("Enable Scene Detection"))
        self.scene_settings_group.setTitle(tr("Scene Detection Settings"))
        self.scene_model_label.setText(tr("Model:"))
        self.scene_threshold_label.setText(tr("Threshold:"))
        self.scene_fp16.setText(tr("Use FP16 Precision"))
        
        # Device tab
        self.device_label.setText(tr("Device:"))
        self.device_auto.setText(tr("Auto-select best device"))
        self.refresh_devices_btn.setText(tr("Refresh Device List"))
        self.benchmark_btn.setText(tr("Device Detection & Benchmark..."))
        
        # Output tab (codec settings widget handles its own retranslation)
        self.codec_settings.retranslate_ui()

    def _create_interpolation_tab(self) -> QWidget:
        """Create the interpolation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Enable checkbox
        self.interp_enabled = QCheckBox(tr("Enable Interpolation"))
        self.interp_enabled.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self.interp_enabled)

        # Settings group
        self.interp_settings_group = QGroupBox(tr("Interpolation Settings"))
        settings_layout = QVBoxLayout(self.interp_settings_group)

        # Model type selection
        type_layout = QHBoxLayout()
        self.interp_type_label = QLabel(tr("Model:"))
        type_layout.addWidget(self.interp_type_label)
        self.interp_type = QComboBox()
        type_layout.addWidget(self.interp_type, 1)
        self.interp_type.currentTextChanged.connect(self._on_model_type_changed)
        settings_layout.addLayout(type_layout)

        # Model version selection
        version_layout = QHBoxLayout()
        self.interp_model_label = QLabel(tr("Version:"))
        version_layout.addWidget(self.interp_model_label)
        self.interp_model = QComboBox()
        self.interp_model.currentTextChanged.connect(self._on_model_version_changed)
        version_layout.addWidget(self.interp_model, 1)
        settings_layout.addLayout(version_layout)

        # Multiplier
        multi_layout = QHBoxLayout()
        self.interp_multi_label = QLabel(tr("Multiplier:"))
        multi_layout.addWidget(self.interp_multi_label)
        self.interp_multi = QComboBox()
        self.interp_multi.addItems([str(m) for m in self.INTERP_MULTIPLIERS])
        self.interp_multi.currentTextChanged.connect(self._on_config_changed)
        multi_layout.addWidget(self.interp_multi, 1)
        settings_layout.addLayout(multi_layout)

        # Scale
        scale_layout = QHBoxLayout()
        self.interp_scale_label = QLabel(tr("Scale:"))
        scale_layout.addWidget(self.interp_scale_label)
        self.interp_scale = QDoubleSpinBox()
        self.interp_scale.setRange(0.1, 2.0)
        self.interp_scale.setSingleStep(0.1)
        self.interp_scale.setValue(1.0)
        self.interp_scale.valueChanged.connect(self._on_config_changed)
        scale_layout.addWidget(self.interp_scale, 1)
        settings_layout.addLayout(scale_layout)

        # Scene change detection
        self.interp_scene_change = QCheckBox(tr("Enable Scene Change Detection (sc=True)"))
        self.interp_scene_change.stateChanged.connect(self._on_config_changed)
        settings_layout.addWidget(self.interp_scene_change)

        # No models warning
        self.no_models_label = QLabel(tr("No models installed. Go to Models tab to download."))
        self.no_models_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.no_models_label.setVisible(False)
        settings_layout.addWidget(self.no_models_label)

        layout.addWidget(self.interp_settings_group)
        layout.addStretch()
        
        # Populate model types after all UI elements are created
        self._populate_model_types()
        self._populate_model_versions(self.interp_type.currentText())

        return widget

    def _populate_model_types(self):
        """Populate model type dropdown with types that have installed models."""
        self.interp_type.clear()
        
        model_types = self._model_selection.get_available_model_types()
        
        for type_info in sorted(model_types, key=lambda x: x.name):
            self.interp_type.addItem(
                f"{type_info.display_name}",
                type_info.name
            )
        
        if not model_types:
            self.no_models_label.setVisible(True)
            self.interp_settings_group.setEnabled(False)
        else:
            self.no_models_label.setVisible(False)
            self.interp_settings_group.setEnabled(True)
            
            # Select current type from model selection manager
            current_type = self._model_selection.get_selection().model_type
            for i in range(self.interp_type.count()):
                if self.interp_type.itemData(i) == current_type:
                    self.interp_type.setCurrentIndex(i)
                    break

    def _populate_model_versions(self, model_type: str):
        """Populate model version dropdown with installed versions.
        
        Args:
            model_type: Model type name (e.g., "rife", "film")
        """
        self.interp_model.clear()
        
        if not model_type:
            return
            
        # Get the actual model type from combo box data
        actual_type = self.interp_type.currentData() or model_type.lower()
        
        # Get available checkpoints from model selection manager
        checkpoints = self._model_selection.get_available_checkpoints(actual_type)
        
        for ckpt_info in checkpoints:
            # Use version string for display, checkpoint name for data
            version = self._model_selection._checkpoint_to_version(actual_type, ckpt_info.name)
            self.interp_model.addItem(version, ckpt_info.name)
        
        # Select current checkpoint
        current_ckpt = self._model_selection.get_selection().checkpoint_name
        for i in range(self.interp_model.count()):
            if self.interp_model.itemData(i) == current_ckpt:
                self.interp_model.setCurrentIndex(i)
                break

    def _on_model_type_changed(self):
        """Handle model type selection change."""
        model_type = self.interp_type.currentData()
        if model_type:
            self._model_selection.set_model_type(model_type)
        self._populate_model_versions(self.interp_type.currentText())
        self._on_config_changed()
    
    def _on_model_version_changed(self):
        """Handle model version selection change."""
        checkpoint_name = self.interp_model.currentData()
        if checkpoint_name:
            self._model_selection.set_checkpoint(checkpoint_name)
        self._on_config_changed()
    
    def _on_external_model_type_changed(self, model_type: str):
        """Handle model type change from external source (e.g., model panel)."""
        # Update UI to match external selection
        for i in range(self.interp_type.count()):
            if self.interp_type.itemData(i) == model_type:
                self.interp_type.setCurrentIndex(i)
                self._populate_model_versions(model_type)
                break
    
    def _on_external_checkpoint_changed(self, model_type: str, checkpoint_name: str):
        """Handle checkpoint change from external source."""
        # Update version dropdown
        for i in range(self.interp_model.count()):
            if self.interp_model.itemData(i) == checkpoint_name:
                self.interp_model.setCurrentIndex(i)
                break
    
    def _on_available_models_changed(self):
        """Handle available models changed (installed/uninstalled)."""
        self._populate_model_types()
        self._populate_model_versions(self.interp_type.currentData() or "")

    def _create_upscaling_tab(self) -> QWidget:
        """Create the upscaling settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Enable checkbox
        self.upscale_enabled = QCheckBox(tr("Enable Upscaling"))
        self.upscale_enabled.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self.upscale_enabled)

        # Settings group
        self.upscale_settings_group = QGroupBox(tr("Upscaling Settings"))
        settings_layout = QVBoxLayout(self.upscale_settings_group)

        # Engine selection
        engine_layout = QHBoxLayout()
        self.upscale_engine_label = QLabel(tr("Engine:"))
        engine_layout.addWidget(self.upscale_engine_label)
        self.upscale_engine = QComboBox()
        self.upscale_engine.setEditable(True)
        self._populate_engines()
        self.upscale_engine.currentTextChanged.connect(self._on_config_changed)
        engine_layout.addWidget(self.upscale_engine, 1)

        # Refresh button
        refresh_btn = QLabel(tr("(Refresh on model scan)"))
        refresh_btn.setStyleSheet("color: #808080; font-size: 9pt;")
        engine_layout.addWidget(refresh_btn)
        settings_layout.addLayout(engine_layout)

        # Tile size
        tile_layout = QHBoxLayout()
        self.upscale_tile_label = QLabel(tr("Tile Size:"))
        tile_layout.addWidget(self.upscale_tile_label)
        self.upscale_tile_size = QSpinBox()
        self.upscale_tile_size.setRange(0, 4096)
        self.upscale_tile_size.setSpecialValueText(tr("Auto (0)"))
        self.upscale_tile_size.setValue(0)
        self.upscale_tile_size.valueChanged.connect(self._on_config_changed)
        tile_layout.addWidget(self.upscale_tile_size, 1)

        tile_hint = QLabel(tr("(0 = Auto)"))
        tile_hint.setStyleSheet("color: #808080; font-size: 9pt;")
        tile_layout.addWidget(tile_hint)
        settings_layout.addLayout(tile_layout)

        # Overlap
        overlap_layout = QHBoxLayout()
        self.upscale_overlap_label = QLabel(tr("Overlap:"))
        overlap_layout.addWidget(self.upscale_overlap_label)
        self.upscale_overlap = QSpinBox()
        self.upscale_overlap.setRange(0, 512)
        self.upscale_overlap.setValue(0)
        self.upscale_overlap.valueChanged.connect(self._on_config_changed)
        overlap_layout.addWidget(self.upscale_overlap, 1)
        settings_layout.addLayout(overlap_layout)

        # Num streams
        streams_layout = QHBoxLayout()
        self.upscale_streams_label = QLabel(tr("GPU Streams:"))
        streams_layout.addWidget(self.upscale_streams_label)
        self.upscale_num_streams = QSpinBox()
        self.upscale_num_streams.setRange(1, 16)
        self.upscale_num_streams.setValue(3)
        self.upscale_num_streams.valueChanged.connect(self._on_config_changed)
        streams_layout.addWidget(self.upscale_num_streams, 1)

        streams_hint = QLabel(tr("(Higher = more VRAM)"))
        streams_hint.setStyleSheet("color: #808080; font-size: 9pt;")
        streams_layout.addWidget(streams_hint)
        settings_layout.addLayout(streams_layout)

        layout.addWidget(self.upscale_settings_group)
        layout.addStretch()

        return widget

    def _create_scene_detection_tab(self) -> QWidget:
        """Create the scene detection settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Enable checkbox
        self.scene_enabled = QCheckBox(tr("Enable Scene Detection"))
        self.scene_enabled.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self.scene_enabled)

        # Settings group
        self.scene_settings_group = QGroupBox(tr("Scene Detection Settings"))
        settings_layout = QVBoxLayout(self.scene_settings_group)

        # Model selection
        model_layout = QHBoxLayout()
        self.scene_model_label = QLabel(tr("Model:"))
        model_layout.addWidget(self.scene_model_label)
        self.scene_model = QComboBox()
        for model_id, model_name in self.SCENE_DETECT_MODELS.items():
            self.scene_model.addItem(model_name, model_id)
        self.scene_model.currentIndexChanged.connect(self._on_config_changed)
        model_layout.addWidget(self.scene_model, 1)
        settings_layout.addLayout(model_layout)

        # Threshold
        thresh_layout = QHBoxLayout()
        self.scene_threshold_label = QLabel(tr("Threshold:"))
        thresh_layout.addWidget(self.scene_threshold_label)
        self.scene_threshold = QDoubleSpinBox()
        self.scene_threshold.setRange(0.0, 1.0)
        self.scene_threshold.setSingleStep(0.05)
        self.scene_threshold.setValue(0.5)
        self.scene_threshold.setDecimals(2)
        self.scene_threshold.valueChanged.connect(self._on_config_changed)
        thresh_layout.addWidget(self.scene_threshold, 1)

        thresh_hint = QLabel(tr("(Higher = less sensitive)"))
        thresh_hint.setStyleSheet("color: #808080; font-size: 9pt;")
        thresh_layout.addWidget(thresh_hint)
        settings_layout.addLayout(thresh_layout)

        # FP16
        self.scene_fp16 = QCheckBox(tr("Use FP16 Precision"))
        self.scene_fp16.setChecked(True)
        self.scene_fp16.stateChanged.connect(self._on_config_changed)
        settings_layout.addWidget(self.scene_fp16)

        # Info label
        info_label = QLabel(
            tr("Note: Scene detection identifies shot boundaries to prevent\ninterpolation artifacts during scene transitions.")
        )
        info_label.setStyleSheet("color: #808080; font-size: 9pt;")
        settings_layout.addWidget(info_label)

        layout.addWidget(self.scene_settings_group)
        layout.addStretch()

        return widget

    def _create_device_tab(self) -> QWidget:
        """Create the device selection settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Device selection group
        device_group = QGroupBox(tr("Processing Device"))
        device_layout = QVBoxLayout(device_group)

        # Device selection dropdown
        device_select_layout = QHBoxLayout()
        self.device_label = QLabel(tr("Device:"))
        device_select_layout.addWidget(self.device_label)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(250)
        self.device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
        device_select_layout.addWidget(self.device_combo, 1)

        device_layout.addLayout(device_select_layout)

        # Device info label
        self.device_info_label = QLabel(tr("No device selected"))
        self.device_info_label.setStyleSheet("color: #808080; font-size: 9pt;")
        self.device_info_label.setWordWrap(True)
        device_layout.addWidget(self.device_info_label)

        # Refresh devices button
        refresh_layout = QHBoxLayout()
        self.refresh_devices_btn = QPushButton(tr("Refresh Device List"))
        self.refresh_devices_btn.clicked.connect(self._refresh_devices)
        refresh_layout.addWidget(self.refresh_devices_btn)

        # Benchmark button
        self.benchmark_btn = QPushButton(tr("Device Detection & Benchmark..."))
        self.benchmark_btn.clicked.connect(self._open_benchmark_dialog)
        refresh_layout.addWidget(self.benchmark_btn)
        refresh_layout.addStretch()

        device_layout.addLayout(refresh_layout)
        layout.addWidget(device_group)

        # Auto device option
        self.device_auto = QCheckBox(tr("Auto-select best device"))
        self.device_auto.setChecked(False)  # Default to manual selection
        self.device_auto.stateChanged.connect(self._on_auto_device_changed)
        layout.addWidget(self.device_auto)

        # Device details group
        details_group = QGroupBox(tr("Selected Device Details"))
        details_layout = QFormLayout(details_group)

        self.device_type_label = QLabel(tr("-"))
        self.device_memory_label = QLabel(tr("-"))
        self.device_compute_label = QLabel(tr("-"))

        details_layout.addRow(tr("Type:"), self.device_type_label)
        details_layout.addRow(tr("Memory:"), self.device_memory_label)
        details_layout.addRow(tr("Compute Capability:"), self.device_compute_label)

        layout.addWidget(details_group)

        # Info label
        info_label = QLabel(
            tr("Note: Select the device for video processing. "
               "Auto-select will choose the best available GPU.")
        )
        info_label.setStyleSheet("color: #808080; font-size: 9pt;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

        # Populate device list
        self._refresh_devices()
        
        # Set initial state based on auto checkbox
        self.device_combo.setEnabled(not self.device_auto.isChecked())

        return widget

    def _on_device_selection_changed(self):
        """Handle device selection change."""
        self._update_device_info()
        self._on_config_changed()

    def _on_auto_device_changed(self, state):
        """Handle auto device checkbox change."""
        self.device_combo.setEnabled(not self.device_auto.isChecked())
        self._update_device_info()
        self._on_config_changed()

    def _refresh_devices(self):
        """Refresh the list of available devices."""
        try:
            self.device_combo.clear()
            self._available_devices = device_manager.get_devices()

            # Add auto option
            self.device_combo.addItem(tr("Auto (Best Available)"), "auto")

            for device in self._available_devices:
                device_str = f"{device.display_name}"
                if device.total_memory_mb > 0:
                    device_str += f" ({device.memory_gb:.1f} GB)"

                # Create device identifier based on device type
                if device.device_type == DeviceType.CUDA:
                    device_id = f"cuda:{device.device_id}"
                elif device.device_type == DeviceType.ROCM:
                    device_id = f"rocm:{device.device_id}"
                elif device.device_type == DeviceType.XPU:
                    device_id = f"xpu:{device.device_id}"
                else:
                    device_id = "cpu"

                self.device_combo.addItem(device_str, device_id)

            # Select best device by default
            self._update_device_info()
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to refresh devices: {e}")
            # Add fallback options
            self.device_combo.addItem(tr("Auto (Best Available)"), "auto")
            self.device_combo.addItem(tr("CPU"), "cpu")
            self._update_device_info()

    def _update_device_info(self):
        """Update device info labels based on selection."""
        device_str = self.device_combo.currentData()
        is_auto = self.device_auto.isChecked()

        # Default to auto if no selection
        if device_str is None:
            device_str = "auto"

        if is_auto or device_str == "auto":
            best_device = device_manager.get_best_device()
            if best_device:
                self.device_info_label.setText(
                    tr("Will use best available device: {}")
                    .format(best_device.display_name)
                )
                self.device_type_label.setText(best_device.device_type.value.upper())
                self.device_memory_label.setText(
                    f"{best_device.memory_gb:.1f} GB" if best_device.total_memory_mb > 0 else "N/A"
                )
                self.device_compute_label.setText(
                    best_device.compute_capability or "N/A"
                )
        else:
            # Find device info
            for device in self._available_devices:
                current_id = None
                if device.device_type == DeviceType.CUDA:
                    current_id = f"cuda:{device.device_id}"
                elif device.device_type == DeviceType.ROCM:
                    current_id = f"rocm:{device.device_id}"
                elif device.device_type == DeviceType.XPU:
                    current_id = f"xpu:{device.device_id}"
                else:
                    current_id = "cpu"

                if current_id == device_str:
                    self.device_info_label.setText(device.display_name)
                    self.device_type_label.setText(device.device_type.value.upper())
                    self.device_memory_label.setText(
                        f"{device.memory_gb:.1f} GB" if device.total_memory_mb > 0 else "N/A"
                    )
                    self.device_compute_label.setText(
                        device.compute_capability or "N/A"
                    )
                    break

    def _open_benchmark_dialog(self):
        """Open the benchmark dialog."""
        dialog = BenchmarkDialog(self)
        dialog.exec()
        # Refresh devices after dialog closes
        self._refresh_devices()

    def _create_output_tab(self) -> QWidget:
        """Create the output settings tab with comprehensive codec configuration."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use the new CodecSettingsWidget
        self.codec_settings = CodecSettingsWidget(self._config)
        self.codec_settings.config_changed.connect(self._on_config_changed)
        layout.addWidget(self.codec_settings)

        return widget

    def _populate_engines(self):
        """Populate the engine dropdown with available engines."""
        self.upscale_engine.clear()

        # Add engines from model selection manager's underlying model manager
        model_manager = self._model_selection.get_model_manager()
        engines = model_manager.get_available_engines()
        for engine_path in engines:
            # Display the filename, but store the full path
            engine_name = Path(engine_path).name
            self.upscale_engine.addItem(engine_name, engine_path)

    def _on_config_changed(self):
        """Emit config changed signal."""
        self.config_changed.emit()

    def load_config(self, config: Dict[str, Any]):
        """Load configuration into UI."""
        # Interpolation
        interp = config.get("interpolation", {})
        self.interp_enabled.setChecked(interp.get("enabled", False))
        
        # Model selection is managed by ModelSelectionManager which reads from config
        # Just refresh the UI to match current selection
        self._populate_model_types()
        self._populate_model_versions(self._model_selection.get_selection().model_type)
        
        self.interp_multi.setCurrentText(str(interp.get("multi", 2)))
        self.interp_scale.setValue(interp.get("scale", 1.0))
        self.interp_scene_change.setChecked(interp.get("scene_change", False))

        # Upscaling
        upscale = config.get("upscaling", {})
        self.upscale_enabled.setChecked(upscale.get("enabled", True))
        self.upscale_engine.setCurrentText(upscale.get("engine", ""))
        self.upscale_tile_size.setValue(upscale.get("tile_size", 0))
        self.upscale_overlap.setValue(upscale.get("overlap", 0))
        self.upscale_num_streams.setValue(upscale.get("num_streams", 3))

        # Scene Detection
        scene = config.get("scene_detection", {})
        self.scene_enabled.setChecked(scene.get("enabled", False))
        model_id = scene.get("model", 12)
        index = self.scene_model.findData(model_id)
        if index >= 0:
            self.scene_model.setCurrentIndex(index)
        self.scene_threshold.setValue(scene.get("threshold", 0.5))
        self.scene_fp16.setChecked(scene.get("fp16", True))

        # Device
        device_config = config.get("device", {})
        device_str = device_config.get("device", "auto")
        self.device_auto.setChecked(device_str == "auto")
        self.device_combo.setEnabled(device_str != "auto")

        # Select device in combo
        if device_str != "auto":
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == device_str:
                    self.device_combo.setCurrentIndex(i)
                    break
        self._update_device_info()

        # Output - use codec settings widget
        output = config.get("output", {})
        self.codec_settings.load_config(output)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration from UI."""
        # Get selection from ModelSelectionManager (source of truth)
        selection = self._model_selection.get_selection()
        model_type = selection.model_type
        model_checkpoint = selection.checkpoint_name
        model_version = self._model_selection.get_version_string()
        
        # Get full checkpoint path
        checkpoint_path = self._model_selection.get_checkpoint_path()
        
        # Get full engine path (use currentData which stores full path)
        engine_path = self.upscale_engine.currentData() or self.upscale_engine.currentText()
        
        # Get output config from codec settings widget
        output_config = self.codec_settings.get_config()
        
        return {
            "interpolation": {
                "enabled": self.interp_enabled.isChecked(),
                "model_type": model_type,
                "model_version": model_version,
                "checkpoint_name": model_checkpoint,
                "checkpoint_path": checkpoint_path,
                "multi": int(self.interp_multi.currentText()),
                "scale": self.interp_scale.value(),
                "scene_change": self.interp_scene_change.isChecked(),
            },
            "upscaling": {
                "enabled": self.upscale_enabled.isChecked(),
                "engine": engine_path,
                "tile_size": self.upscale_tile_size.value(),
                "overlap": self.upscale_overlap.value(),
                "num_streams": self.upscale_num_streams.value(),
            },
            "scene_detection": {
                "enabled": self.scene_enabled.isChecked(),
                "model": self.scene_model.currentData(),
                "threshold": self.scene_threshold.value(),
                "fp16": self.scene_fp16.isChecked(),
            },
            "device": {
                "device": "auto" if self.device_auto.isChecked() else self.device_combo.currentData(),
            },
            "output": output_config,
        }
    
    def refresh_models(self):
        """Refresh the model list from model manager."""
        self._model_selection.refresh()
        self._populate_model_types()
        self._populate_engines()
