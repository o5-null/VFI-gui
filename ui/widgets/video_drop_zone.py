"""VideoDropZone — video drag-and-drop zone with file browsing.

A styled QFrame for video file selection via drag-and-drop or file browser.
Shows placeholder when empty, video info when loaded.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt6.QtCore import pyqtSignal, QEvent, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QDragLeaveEvent

from ui.styles.icons import IconManager
from ui.styles.theme import Theme

if TYPE_CHECKING:
    from ui.viewmodels.pipeline_viewmodel import PipelineViewModel


class VideoDropZone(QFrame):
    """Video drag-and-drop zone with file browsing support.

    Signals:
        video_selected: Emitted when a video file is selected (str path)

    Two display modes:
        - Empty: Large drop zone with icon + "Drop video here or click to browse"
        - Loaded: Compact info panel showing filename, size, and Change button
    """

    video_selected = pyqtSignal(str)

    # Supported video extensions
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}

    def __init__(self, vm: "PipelineViewModel", parent=None):
        """Initialize VideoDropZone.

        Args:
            vm: PipelineViewModel for state binding
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        self.setAcceptDrops(True)
        self.setObjectName("videoDropZone")
        self._setup_ui()
        self._bind_viewmodel()
        self.retranslate_ui()

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change for i18n."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(a0)

    def retranslate_ui(self) -> None:
        """Update all user-visible text for i18n."""
        # Only retranslate if we have the widgets
        if hasattr(self, "_placeholder_label"):
            self._placeholder_label.setText(
                self.tr("Drop video here or click to browse")
            )
            self._browse_btn.setText(self.tr("Browse"))
            self._change_btn.setText(self.tr("Change"))
            self._clear_btn.setText(self.tr("Clear"))

    def _setup_ui(self) -> None:
        """Create widget structure."""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(
            Theme.PADDING_LG, Theme.PADDING_LG,
            Theme.PADDING_LG, Theme.PADDING_LG
        )
        self._layout.setSpacing(Theme.SPACING_MD)

        # Placeholder mode widgets (shown when no video)
        self._icon_label = QLabel()
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_icon = IconManager.get("video", Theme.TEXT_SECONDARY)
        self._icon_label.setPixmap(video_icon.pixmap(48, 48))

        self._placeholder_label = QLabel()
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setObjectName("dropZonePlaceholder")

        self._browse_btn = QPushButton()
        self._browse_btn.clicked.connect(self._open_file_dialog)

        # Placeholder container
        self._placeholder_container = QVBoxLayout()
        self._placeholder_container.addWidget(self._icon_label)
        self._placeholder_container.addWidget(self._placeholder_label)
        self._placeholder_container.addWidget(self._browse_btn)
        self._placeholder_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Loaded mode widgets (shown when video selected)
        self._filename_label = QLabel()
        self._filename_label.setObjectName("dropZoneFilename")

        self._filesize_label = QLabel()
        self._filesize_label.setObjectName("dropZoneFileSize")

        self._change_btn = QPushButton()
        self._change_btn.clicked.connect(self._open_file_dialog)

        self._clear_btn = QPushButton()
        self._clear_btn.clicked.connect(self._clear_video)

        # Loaded container
        self._loaded_container = QVBoxLayout()
        self._loaded_container.addWidget(self._filename_label)
        self._loaded_container.addWidget(self._filesize_label)
        btn_row = QVBoxLayout()
        btn_row.addWidget(self._change_btn)
        btn_row.addWidget(self._clear_btn)
        self._loaded_container.addLayout(btn_row)
        self._loaded_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Start in placeholder mode
        self._show_placeholder_mode()

    def _bind_viewmodel(self) -> None:
        """Connect to PipelineViewModel signals."""
        self._vm.video_path_changed.connect(self._on_video_path_changed)
        # Initialize display from current state
        self._update_display(self._vm.video_path)

    def _on_video_path_changed(self, path: str) -> None:
        """Handle video path change from ViewModel."""
        self._update_display(path)

    def _update_display(self, path: str) -> None:
        """Update display based on video path."""
        if path and Path(path).exists():
            self._show_loaded_mode(path)
        else:
            self._show_placeholder_mode()

    def _show_placeholder_mode(self) -> None:
        """Show empty drop zone with placeholder."""
        # Hide loaded widgets
        self._filename_label.hide()
        self._filesize_label.hide()
        self._change_btn.hide()
        self._clear_btn.hide()

        # Show placeholder widgets
        self._icon_label.show()
        self._placeholder_label.show()
        self._browse_btn.show()

        # Resize to larger size for drop zone
        self.setMinimumHeight(120)

    def _show_loaded_mode(self, path: str) -> None:
        """Show video info panel."""
        # Hide placeholder widgets
        self._icon_label.hide()
        self._placeholder_label.hide()
        self._browse_btn.hide()

        # Show loaded widgets
        self._filename_label.show()
        self._filesize_label.show()
        self._change_btn.show()
        self._clear_btn.show()

        # Update info
        file_path = Path(path)
        self._filename_label.setText(file_path.name)

        # Calculate file size
        try:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            if size_mb >= 1024:
                size_gb = size_mb / 1024
                self._filesize_label.setText(f"{size_gb:.2f} GB")
            else:
                self._filesize_label.setText(f"{size_mb:.2f} MB")
        except OSError:
            self._filesize_label.setText(self.tr("Size unknown"))

        # Resize to compact size
        self.setMinimumHeight(80)

    def _open_file_dialog(self) -> None:
        """Open file browser for video selection."""
        filter_str = self.tr("Video files (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv)")
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Video"),
            "",
            filter_str,
        )
        if path:
            self._select_video(path)

    def _select_video(self, path: str) -> None:
        """Handle video selection."""
        self._vm.set_video_path(path)
        self.video_selected.emit(path)

    def _clear_video(self) -> None:
        """Clear current video selection."""
        self._vm.set_video_path("")
        self.video_selected.emit("")

    # ====================
    # Drag and Drop Events
    # ====================

    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:
        """Accept drag if it contains video file URLs."""
        if a0 is None:
            return
        mime_data = a0.mimeData()
        if mime_data is not None and mime_data.hasUrls():
            urls = mime_data.urls()
            for url in urls:
                path = url.toLocalFile()
                ext = Path(path).suffix.lower()
                if ext in self.VIDEO_EXTENSIONS:
                    a0.acceptProposedAction()
                    return
        a0.ignore()

    def dragLeaveEvent(self, a0: QDragLeaveEvent | None) -> None:
        """Visual feedback when drag leaves."""
        # Reset to normal state - stylesheet handles visual feedback
        super().dragLeaveEvent(a0)

    def dropEvent(self, a0: QDropEvent | None) -> None:
        """Handle drop — extract first valid video URL."""
        if a0 is None:
            return
        mime_data = a0.mimeData()
        if mime_data is None:
            a0.ignore()
            return
        urls = mime_data.urls()
        for url in urls:
            path = url.toLocalFile()
            ext = Path(path).suffix.lower()
            if ext in self.VIDEO_EXTENSIONS:
                self._select_video(path)
                a0.acceptProposedAction()
                return
        a0.ignore()


__all__ = ["VideoDropZone"]