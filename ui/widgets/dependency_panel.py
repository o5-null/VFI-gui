"""Dependency Management Panel for VFI-gui.

Provides a panel with sidebar navigation for managing various dependencies.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QLabel,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from loguru import logger

from core import tr
from ui.widgets.ffmpeg_manager import FFmpegManagerWidget


class DependencyPanel(QWidget):
    """Panel for managing external dependencies with sidebar navigation."""

    def __init__(self, config=None, parent=None):
        """Initialize dependency panel.

        Args:
            config: Optional Config instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("dependencySidebar")
        sidebar.setFixedWidth(180)
        sidebar.setStyleSheet("""
            #dependencySidebar {
                background-color: #2d2d2d;
                border-right: 1px solid #3d3d3d;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(4)

        # Sidebar title
        self.title_label = QLabel(tr("Dependencies"))
        self.title_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            color: #ffffff;
            padding: 8px 4px;
        """)
        sidebar_layout.addWidget(self.title_label)

        # Navigation list
        self.nav_list = QListWidget()
        self.nav_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                color: #cccccc;
                padding: 10px 12px;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #3d3d3d;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        sidebar_layout.addWidget(self.nav_list)

        sidebar_layout.addStretch()

        layout.addWidget(sidebar)

        # Content stack
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background-color: #1e1e1e;")

        # Create pages
        self._create_pages()

        layout.addWidget(self.content_stack, 1)

        # Select first item
        if self.nav_list.count() > 0:
            self.nav_list.setCurrentRow(0)

    def _create_pages(self):
        """Create dependency management pages."""
        # FFmpeg page
        ffmpeg_widget = FFmpegManagerWidget(self.config)
        self.content_stack.addWidget(ffmpeg_widget)

        ffmpeg_item = QListWidgetItem(tr("FFmpeg"))
        ffmpeg_item.setData(Qt.ItemDataRole.UserRole, "ffmpeg")
        self.nav_list.addItem(ffmpeg_item)

        # Placeholder for future dependencies
        # Python/VapourSynth page (placeholder)
        # python_widget = self._create_placeholder_page(tr("Python/VapourSynth"))
        # self.content_stack.addWidget(python_widget)
        # python_item = QListWidgetItem(tr("Python/VapourSynth"))
        # python_item.setData(Qt.ItemDataRole.UserRole, "python")
        # self.nav_list.addItem(python_item)

    def _create_placeholder_page(self, name: str) -> QWidget:
        """Create a placeholder page for future dependencies.

        Args:
            name: Name of the dependency.

        Returns:
            Placeholder widget.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel(tr("{} management will be added in a future update.").format(name))
        label.setStyleSheet("color: #888; font-size: 14px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        return widget

    def _on_nav_changed(self, row: int):
        """Handle navigation selection change.

        Args:
            row: Selected row index.
        """
        self.content_stack.setCurrentIndex(row)

    def retranslate_ui(self):
        """Retranslate UI elements."""
        # Update sidebar title
        self.title_label.setText(tr("Dependencies"))

        # Update navigation items
        for i in range(self.nav_list.count()):
            item = self.nav_list.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            if key == "ffmpeg":
                item.setText(tr("FFmpeg"))
            elif key == "python":
                item.setText(tr("Python/VapourSynth"))

        # Retranslate current page
        current_widget = self.content_stack.currentWidget()
        if hasattr(current_widget, "retranslate_ui"):
            current_widget.retranslate_ui()
