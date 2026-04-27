"""FFmpeg Manager Widget for VFI-gui.

Provides UI for FFmpeg detection, configuration, and management.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from loguru import logger

from core.dependency_manager import FFmpegManager, DependencyInfo
from core import tr


class FFmpegManagerWidget(QWidget):
    """Widget for managing FFmpeg installation."""

    # Signals
    ffmpeg_changed = pyqtSignal(str)  # Emitted when FFmpeg path changes

    def __init__(self, config=None, parent=None):
        """Initialize FFmpeg manager widget.

        Args:
            config: Optional Config instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.config = config
        self.ffmpeg_manager: Optional[FFmpegManager] = None
        self._setup_ui()
        self._check_ffmpeg()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Status group
        status_group = QGroupBox(tr("FFmpeg Status"))
        status_layout = QGridLayout(status_group)
        status_layout.setSpacing(8)

        # Status indicator
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(20, 20)
        self.status_icon.setText("●")
        self.status_icon.setStyleSheet("color: #888; font-size: 20px;")
        status_layout.addWidget(self.status_icon, 0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel(tr("Checking..."))
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label, 0, 1, 1, 1)

        # Version
        self.version_label = QLabel(tr("Version: -"))
        status_layout.addWidget(self.version_label, 1, 0, 1, 2)

        # Path
        self.path_label = QLabel(tr("Path: -"))
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_layout.addWidget(self.path_label, 2, 0, 1, 2)

        layout.addWidget(status_group)

        # Configuration group
        config_group = QGroupBox(tr("FFmpeg Configuration"))
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(8)

        # Custom path
        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)

        path_label = QLabel(tr("Custom Path:"))
        path_layout.addWidget(path_label)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(tr("Leave empty to use system PATH"))
        self.path_edit.textChanged.connect(self._on_path_changed)
        path_layout.addWidget(self.path_edit, 1)

        self.browse_btn = QPushButton(tr("Browse..."))
        self.browse_btn.clicked.connect(self._on_browse)
        path_layout.addWidget(self.browse_btn)

        config_layout.addLayout(path_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.detect_btn = QPushButton(tr("Detect"))
        self.detect_btn.clicked.connect(self._check_ffmpeg)
        btn_layout.addWidget(self.detect_btn)

        self.apply_btn = QPushButton(tr("Apply"))
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_btn.setEnabled(False)
        btn_layout.addWidget(self.apply_btn)

        btn_layout.addStretch()

        config_layout.addLayout(btn_layout)

        layout.addWidget(config_group)

        # Hardware acceleration group
        hw_group = QGroupBox(tr("Hardware Acceleration"))
        hw_layout = QGridLayout(hw_group)
        hw_layout.setSpacing(8)

        self.hw_labels = {}
        hw_items = [
            ("nvenc", tr("NVIDIA NVENC")),
            ("qsv", tr("Intel Quick Sync")),
            ("amf", tr("AMD AMF")),
            ("cuda", tr("CUDA")),
            ("vaapi", tr("VAAPI")),
            ("videotoolbox", tr("VideoToolbox")),
        ]

        for i, (key, name) in enumerate(hw_items):
            row = i // 2
            col = (i % 2) * 2

            label = QLabel(f"{name}:")
            hw_layout.addWidget(label, row, col)

            value_label = QLabel("-")
            self.hw_labels[key] = value_label
            hw_layout.addWidget(value_label, row, col + 1)

        layout.addWidget(hw_group)

        # Load saved path
        if self.config:
            saved_path = self.config.get("paths.ffmpeg_dir", "")
            if saved_path:
                self.path_edit.setText(saved_path)

        layout.addStretch()

    def _check_ffmpeg(self):
        """Check FFmpeg installation."""
        custom_path = self.path_edit.text().strip() or None
        self.ffmpeg_manager = FFmpegManager(custom_path)
        info = self.ffmpeg_manager.detect()

        self._update_status(info)

        if info.installed:
            self._check_hardware_acceleration()

    def _update_status(self, info: DependencyInfo):
        """Update status display.

        Args:
            info: DependencyInfo from detection.
        """
        if info.installed:
            self.status_icon.setStyleSheet("color: #4CAF50; font-size: 20px;")
            self.status_label.setText(tr("Installed"))
            self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")

            if info.version:
                self.version_label.setText(tr("Version: {}").format(info.version))
            else:
                self.version_label.setText(tr("Version: Unknown"))

            if info.path:
                self.path_label.setText(tr("Path: {}").format(info.path))
            else:
                self.path_label.setText(tr("Path: -"))
        else:
            self.status_icon.setStyleSheet("color: #F44336; font-size: 20px;")
            self.status_label.setText(tr("Not Found"))
            self.status_label.setStyleSheet("font-weight: bold; color: #F44336;")
            self.version_label.setText(tr("Version: -"))
            self.path_label.setText(tr("Path: -"))

            if info.error:
                self.path_label.setText(tr("Error: {}").format(info.error))

    def _check_hardware_acceleration(self):
        """Check and display hardware acceleration support."""
        if not self.ffmpeg_manager:
            return

        hw_accels = self.ffmpeg_manager.check_hardware_acceleration()

        for key, label in self.hw_labels.items():
            available = hw_accels.get(key, False)
            if available:
                label.setText(tr("Available"))
                label.setStyleSheet("color: #4CAF50;")
            else:
                label.setText(tr("Not Available"))
                label.setStyleSheet("color: #888;")

    def _on_path_changed(self, text: str):
        """Handle path text change."""
        self.apply_btn.setEnabled(True)

    def _on_browse(self):
        """Browse for FFmpeg directory or executable."""
        current_path = self.path_edit.text().strip()

        # Try to open file dialog for executable first
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select FFmpeg Executable"),
            current_path,
            tr("Executables (*.exe);;All Files (*)") if hasattr(Qt, 'Win') else tr("All Files (*)")
        )

        if file_path:
            self.path_edit.setText(str(Path(file_path).parent))
            self.apply_btn.setEnabled(True)
        else:
            # Try directory dialog
            dir_path = QFileDialog.getExistingDirectory(
                self,
                tr("Select FFmpeg Directory"),
                current_path
            )
            if dir_path:
                self.path_edit.setText(dir_path)
                self.apply_btn.setEnabled(True)

    def _on_apply(self):
        """Apply custom FFmpeg path."""
        custom_path = self.path_edit.text().strip()

        if custom_path:
            self.ffmpeg_manager = FFmpegManager(custom_path)
            info = self.ffmpeg_manager.detect()

            if not info.installed:
                QMessageBox.warning(
                    self,
                    tr("FFmpeg Not Found"),
                    tr("FFmpeg was not found at the specified path.\n\n{}").format(
                        info.error or tr("Please check the path and try again.")
                    )
                )
                return

            # Save to config
            if self.config:
                self.config.set("paths.ffmpeg_dir", custom_path)
                self.config.save()

            self._update_status(info)
            self._check_hardware_acceleration()
            self.ffmpeg_changed.emit(custom_path)
        else:
            # Clear custom path, use system PATH
            if self.config:
                self.config.set("paths.ffmpeg_dir", "")
                self.config.save()

            self._check_ffmpeg()
            self.ffmpeg_changed.emit("")

        self.apply_btn.setEnabled(False)
        QMessageBox.information(
            self,
            tr("Settings Saved"),
            tr("FFmpeg path has been updated.")
        )

    def retranslate_ui(self):
        """Retranslate UI elements."""
        # Update group box titles
        self.findChild(QGroupBox, tr("FFmpeg Status")).setTitle(tr("FFmpeg Status"))
        self.findChild(QGroupBox, tr("FFmpeg Configuration")).setTitle(tr("FFmpeg Configuration"))
        self.findChild(QGroupBox, tr("Hardware Acceleration")).setTitle(tr("Hardware Acceleration"))

        # Re-check to update labels
        self._check_ffmpeg()
