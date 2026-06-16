"""About dialog for VFI-gui.

A simple dialog displaying application information, version, license, and links.
"""

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
)
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl


class AboutDialog(QDialog):
    """Dialog showing application information.

    Features:
    - Application name and version
    - Description text
    - License information
    - Links to GitHub and related projects
    """

    # Application metadata
    APP_NAME = "VFI-gui"
    APP_VERSION = "0.1.0"
    APP_LICENSE = "MIT"

    # URLs
    GITHUB_URL = "https://github.com/o5-null/VFI-gui"
    RIFE_URL = "https://github.com/hzwer/RIFE"
    VSGAN_URL = "https://github.com/styler00dollar/VSGAN-tensorrt-docker"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    def retranslate_ui(self):
        """Update all user-visible text for i18n."""
        self.setWindowTitle(self.tr("About %1").replace("%1", self.APP_NAME))

        self._title_label.setText(f"{self.APP_NAME}")
        self._version_label.setText(self.tr("Version: %1").replace("%1", self.APP_VERSION))

        self._desc_label.setText(
            self.tr("A PyQt6 desktop application for AI-powered video frame interpolation "
                    "with multi-backend support (PyTorch, TensorRT, VapourSynth).")
        )

        self._license_label.setText(self.tr("License: %1").replace("%1", self.APP_LICENSE))

        self._links_label.setText(self.tr("Links:"))
        self._github_button.setText(self.tr("GitHub"))
        self._rife_button.setText(self.tr("RIFE"))
        self._vsgan_button.setText(self.tr("VSGAN-tensorrt"))

        self._close_button.button(QDialogButtonBox.StandardButton.Close).setText(self.tr("Close"))

    def _setup_ui(self):
        """Create widgets and layout."""
        self.setWindowTitle(self.tr("About %1").replace("%1", self.APP_NAME))
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        self._title_label = QLabel(self.APP_NAME)
        self._title_label.setStyleSheet("font-size: 24pt; font-weight: bold;")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title_label)

        # Version
        self._version_label = QLabel()
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._version_label)

        # Description
        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._desc_label)

        # License
        self._license_label = QLabel()
        self._license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._license_label)

        # Links section
        self._links_label = QLabel()
        layout.addWidget(self._links_label)

        link_layout = QHBoxLayout()
        self._github_button = QPushButton()
        self._rife_button = QPushButton()
        self._vsgan_button = QPushButton()
        link_layout.addWidget(self._github_button)
        link_layout.addWidget(self._rife_button)
        link_layout.addWidget(self._vsgan_button)
        layout.addLayout(link_layout)

        # Close button
        self._close_button = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        layout.addWidget(self._close_button)

        self.retranslate_ui()

    def _connect_signals(self):
        """Connect button signals."""
        self._github_button.clicked.connect(lambda: self._open_url(self.GITHUB_URL))
        self._rife_button.clicked.connect(lambda: self._open_url(self.RIFE_URL))
        self._vsgan_button.clicked.connect(lambda: self._open_url(self.VSGAN_URL))
        self._close_button.rejected.connect(self.reject)

    def _open_url(self, url: str):
        """Open URL in default browser."""
        QDesktopServices.openUrl(QUrl(url))