"""Settings dialog for VFI-gui.

A tabbed settings dialog with Performance, Proxy, and Dependencies tabs.
Uses ConfigFacade for configuration access and supports i18n.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QLabel,
    QGroupBox,
)
from PyQt6.QtCore import QEvent, Qt

if TYPE_CHECKING:
    from core.config import ConfigFacade


class SettingsDialog(QDialog):
    """Tabbed settings dialog for application configuration.

    Features:
    - Performance tab: Threading, precision, inference streams
    - Proxy tab: Proxy type, host, port, authentication
    - Dependencies tab: FFmpeg, VapourSynth, PyTorch status

    All text uses self.tr() for i18n support.
    """

    def __init__(self, config: "ConfigFacade", parent=None):
        super().__init__(parent)
        self._config = config
        self._setup_ui()
        self._load_settings()
        self._connect_signals()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    def retranslate_ui(self):
        """Update all user-visible text for i18n."""
        self.setWindowTitle(self.tr("Settings"))

        # Tab names
        self._tab_widget.setTabText(0, self.tr("Performance"))
        self._tab_widget.setTabText(1, self.tr("Proxy"))
        self._tab_widget.setTabText(2, self.tr("Dependencies"))

        # Performance tab
        self._perf_group.setTitle(self.tr("Inference Settings"))
        self._thread_label.setText(self.tr("Thread Count:"))
        self._precision_label.setText(self.tr("GPU Precision:"))
        self._streams_label.setText(self.tr("Inference Streams:"))
        self._compile_check.setText(self.tr("Enable torch.compile"))

        # Proxy tab
        self._proxy_group.setTitle(self.tr("Proxy Configuration"))
        self._proxy_type_label.setText(self.tr("Proxy Type:"))
        self._proxy_host_label.setText(self.tr("Host:"))
        self._proxy_port_label.setText(self.tr("Port:"))
        self._proxy_user_label.setText(self.tr("Username:"))
        self._proxy_pass_label.setText(self.tr("Password:"))

        # Dependencies tab
        self._dep_group.setTitle(self.tr("External Dependencies"))

        # Buttons
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("OK"))
        self._button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr("Cancel"))

    def _setup_ui(self):
        """Create widgets and layout."""
        self.setWindowTitle(self.tr("Settings"))
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Tab widget
        self._tab_widget = QTabWidget()
        layout.addWidget(self._tab_widget)

        # Create tabs
        self._create_performance_tab()
        self._create_proxy_tab()
        self._create_dependencies_tab()

        # Dialog buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self._button_box)

        self.retranslate_ui()

    def _create_performance_tab(self):
        """Create Performance settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self._perf_group = QGroupBox()
        form = QFormLayout(self._perf_group)

        # Thread count
        self._thread_spin = QSpinBox()
        self._thread_spin.setRange(1, 32)
        self._thread_spin.setValue(4)
        self._thread_label = QLabel()
        form.addRow(self._thread_label, self._thread_spin)

        # GPU precision
        self._precision_combo = QComboBox()
        self._precision_combo.addItems(["fp32", "fp16", "bf16"])
        self._precision_label = QLabel()
        form.addRow(self._precision_label, self._precision_combo)

        # Inference streams
        self._streams_spin = QSpinBox()
        self._streams_spin.setRange(1, 8)
        self._streams_spin.setValue(3)
        self._streams_label = QLabel()
        form.addRow(self._streams_label, self._streams_spin)

        # torch.compile
        self._compile_check = QCheckBox()
        form.addRow(self._compile_check)

        layout.addWidget(self._perf_group)
        layout.addStretch()

        self._tab_widget.addTab(tab, self.tr("Performance"))

    def _create_proxy_tab(self):
        """Create Proxy settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self._proxy_group = QGroupBox()
        form = QFormLayout(self._proxy_group)

        # Proxy type
        self._proxy_type_combo = QComboBox()
        self._proxy_type_combo.addItems(["None", "HTTP", "SOCKS5"])
        self._proxy_type_label = QLabel()
        form.addRow(self._proxy_type_label, self._proxy_type_combo)

        # Host
        self._proxy_host_edit = QLineEdit()
        self._proxy_host_edit.setPlaceholderText(self.tr("e.g., 127.0.0.1"))
        self._proxy_host_label = QLabel()
        form.addRow(self._proxy_host_label, self._proxy_host_edit)

        # Port
        self._proxy_port_spin = QSpinBox()
        self._proxy_port_spin.setRange(1, 65535)
        self._proxy_port_spin.setValue(8080)
        self._proxy_port_label = QLabel()
        form.addRow(self._proxy_port_label, self._proxy_port_spin)

        # Username
        self._proxy_user_edit = QLineEdit()
        self._proxy_user_label = QLabel()
        form.addRow(self._proxy_user_label, self._proxy_user_edit)

        # Password
        self._proxy_pass_edit = QLineEdit()
        self._proxy_pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._proxy_pass_label = QLabel()
        form.addRow(self._proxy_pass_label, self._proxy_pass_edit)

        layout.addWidget(self._proxy_group)
        layout.addStretch()

        self._tab_widget.addTab(tab, self.tr("Proxy"))

    def _create_dependencies_tab(self):
        """Create Dependencies status tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self._dep_group = QGroupBox()
        form = QFormLayout(self._dep_group)

        # FFmpeg
        self._ffmpeg_label = QLabel(self.tr("Checking..."))
        form.addRow(self.tr("FFmpeg:"), self._ffmpeg_label)

        # VapourSynth
        self._vs_label = QLabel(self.tr("Checking..."))
        form.addRow(self.tr("VapourSynth:"), self._vs_label)

        # Python
        import sys
        self._python_label = QLabel(sys.version.split()[0])
        form.addRow(self.tr("Python:"), self._python_label)

        # PyTorch
        self._pytorch_label = QLabel(self.tr("Checking..."))
        form.addRow(self.tr("PyTorch:"), self._pytorch_label)

        layout.addWidget(self._dep_group)
        layout.addStretch()

        # Populate dependency info
        self._update_dependency_info()

        self._tab_widget.addTab(tab, self.tr("Dependencies"))

    def _update_dependency_info(self):
        """Update dependency status labels."""
        import sys

        # FFmpeg
        from core.dependency_manager import FFmpegManager
        ffmpeg_info = FFmpegManager().detect()
        if ffmpeg_info.installed:
            self._ffmpeg_label.setText(f"{ffmpeg_info.version} ({ffmpeg_info.path})")
        else:
            self._ffmpeg_label.setText(self.tr("Not found"))

        # VapourSynth
        try:
            import vapoursynth as vs
            self._vs_label.setText(f"R{vs.core.version_number()}")
        except ImportError:
            self._vs_label.setText(self.tr("Not installed"))

        # PyTorch
        try:
            import torch
            cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
            self._pytorch_label.setText(f"{torch.__version__} ({cuda_status})")
        except ImportError:
            self._pytorch_label.setText(self.tr("Not installed"))

    def _load_settings(self):
        """Load current settings into widgets."""
        # Performance settings
        threads = self._config.performance.get_inference_threads()
        self._thread_spin.setValue(threads)

        # Proxy settings
        proxy_config = self._config.network.get_proxy_config()
        http_proxy = proxy_config.get("http_proxy", "")
        if http_proxy:
            # Parse proxy URL
            if http_proxy.startswith("socks5://"):
                self._proxy_type_combo.setCurrentText("SOCKS5")
            elif http_proxy.startswith("http://"):
                self._proxy_type_combo.setCurrentText("HTTP")
            else:
                self._proxy_type_combo.setCurrentText("None")

    def _connect_signals(self):
        """Connect button signals."""
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

    def accept(self):
        """Save settings and close dialog."""
        self._save_settings()
        super().accept()

    def _save_settings(self):
        """Save widget values to config."""
        # Performance
        self._config.performance.set_inference_threads(self._thread_spin.value())

        # Proxy - build proxy URL if configured
        proxy_type = self._proxy_type_combo.currentText()
        if proxy_type != "None":
            host = self._proxy_host_edit.text()
            port = self._proxy_port_spin.value()
            scheme = "http" if proxy_type == "HTTP" else "socks5"
            proxy_url = f"{scheme}://{host}:{port}"
            self._config.network.set_proxy_config(http_proxy=proxy_url, https_proxy=proxy_url)
        else:
            self._config.network.set_proxy_config()