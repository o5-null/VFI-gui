"""InputSourceWidget - Unified input selector for video files and image sequences.

Replaces VideoDropZone with auto-detection of input type:
- Video file → input_type="video"
- Image file or folder of images → input_type="image_sequence"

Supports drag-and-drop of:
- Video files (.mp4, .mkv, .avi, etc.)
- Image files (.png, .jpg, .tiff, etc.)
- Folders containing image sequences

Auto-detects input type and updates PipelineViewModel accordingly.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt6.QtCore import pyqtSignal, QEvent, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QDragLeaveEvent

from ui.styles.icons import IconManager
from ui.styles.theme import Theme

from core.utils.file_utils import (
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    is_image_file, is_video_file,
    get_image_sequence_files, detect_image_sequence_pattern,
)

if TYPE_CHECKING:
    from ui.viewmodels.pipeline_viewmodel import PipelineViewModel


class InputSourceWidget(QFrame):
    """Unified input selector with auto-detection of video vs image sequence.

    Signals:
        source_selected(str): Emitted when a source is selected (path string)

    Three display modes:
        - Empty: Large drop zone with icon + hint text
        - Video loaded: Shows filename, size, type badge "视频"
        - Image sequence loaded: Shows folder name, frame count, type badge "图片序列"
    """

    source_selected = pyqtSignal(str)

    def __init__(self, vm: "PipelineViewModel", parent=None):
        super().__init__(parent)
        self._vm = vm
        self._image_frames: List[str] = []
        self.setAcceptDrops(True)
        self.setObjectName("inputSourceWidget")
        self._setup_ui()
        self._bind_viewmodel()
        self._retranslate_ui()

    # ====================
    # Properties
    # ====================

    @property
    def image_frames(self) -> List[str]:
        """Get image sequence frame paths (empty if video input)."""
        return self._image_frames.copy()

    # ====================
    # UI Setup
    # ====================

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_LG, Theme.PADDING_LG,
            Theme.PADDING_LG, Theme.PADDING_LG,
        )
        layout.setSpacing(Theme.SPACING_MD)

        # --- Placeholder mode ---
        self._icon_label = QLabel()
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_icon = IconManager.get("video", Theme.TEXT_SECONDARY)
        self._icon_label.setPixmap(video_icon.pixmap(48, 48))

        self._placeholder_label = QLabel()
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setObjectName("inputSourcePlaceholder")
        self._placeholder_label.setWordWrap(True)

        # Browse buttons row
        self._browse_row = QVBoxLayout()
        self._browse_row.setSpacing(Theme.SPACING_SM)
        self._browse_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._browse_video_btn = QPushButton()
        self._browse_video_btn.setStyleSheet(self._btn_style(secondary=True))
        self._browse_video_btn.clicked.connect(self._open_video_dialog)

        self._browse_folder_btn = QPushButton()
        self._browse_folder_btn.setStyleSheet(self._btn_style(secondary=True))
        self._browse_folder_btn.clicked.connect(self._open_folder_dialog)

        self._browse_row.addWidget(self._browse_video_btn)
        self._browse_row.addWidget(self._browse_folder_btn)

        # --- Loaded mode ---
        self._type_badge = QLabel()
        self._type_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._type_badge.setFixedHeight(24)
        self._type_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {Theme.ACCENT};
                color: white;
                border-radius: 12px;
                padding: 2px 12px;
                font-size: {Theme.FONT_SIZE_SM};
                font-weight: bold;
            }}
        """)

        self._filename_label = QLabel()
        self._filename_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {Theme.FONT_SIZE_LG};
                font-weight: bold;
            }}
        """)
        self._filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._detail_label = QLabel()
        self._detail_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
                font-size: {Theme.FONT_SIZE_MD};
            }}
        """)
        self._detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Loaded buttons
        self._change_btn = QPushButton()
        self._change_btn.setStyleSheet(self._btn_style(secondary=True))
        self._change_btn.clicked.connect(self._open_video_dialog)

        self._clear_btn = QPushButton()
        self._clear_btn.setStyleSheet(self._btn_style(secondary=True))
        self._clear_btn.clicked.connect(self._clear_source)

        self._loaded_btn_row = QVBoxLayout()
        self._loaded_btn_row.setSpacing(Theme.SPACING_SM)
        self._loaded_btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loaded_btn_row.addWidget(self._change_btn)
        self._loaded_btn_row.addWidget(self._clear_btn)

        # Start in placeholder mode
        self._show_placeholder_mode()

        # Frame style
        self.setStyleSheet(f"""
            QFrame#inputSourceWidget {{
                background-color: {Theme.DROP_ZONE_BG};
                border: 2px dashed {Theme.DROP_ZONE_BORDER};
                border-radius: {Theme.RADIUS_LG}px;
            }}
            QFrame#inputSourceWidget[loaded="true"] {{
                background-color: {Theme.BG_TERTIARY};
                border: 1px solid {Theme.BORDER_LIGHT};
            }}
        """)

    @staticmethod
    def _btn_style(secondary: bool = False) -> str:
        bg = Theme.BG_TERTIARY if secondary else Theme.ACCENT
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_SM}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
                min-width: 140px;
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                border-color: {Theme.BORDER_LIGHT};
            }}
        """

    def _bind_viewmodel(self):
        self._vm.video_path_changed.connect(self._on_video_path_changed)
        self._vm.input_type_changed.connect(self._on_input_type_changed)
        self._update_display(self._vm.video_path, self._vm.input_type)

    # ====================
    # Display Modes
    # ====================

    def _show_placeholder_mode(self):
        """Show empty drop zone."""
        # Clear layout
        while self.layout().count():
            item = self.layout().takeAt(0)
            w = item.widget()
            if w:
                w.hide()

        self._icon_label.show()
        self._placeholder_label.show()
        self._browse_video_btn.show()
        self._browse_folder_btn.show()

        self.layout().addWidget(self._icon_label)
        self.layout().addWidget(self._placeholder_label)
        self.layout().addLayout(self._browse_row)
        self.layout().addStretch()

        self.setMinimumHeight(160)
        self.setProperty("loaded", "false")
        self.setStyle(self.style())

    def _show_loaded_mode(self, path: str, input_type: str):
        """Show loaded source info."""
        while self.layout().count():
            item = self.layout().takeAt(0)
            w = item.widget()
            if w:
                w.hide()

        file_path = Path(path)

        # Type badge
        if input_type == "image_sequence":
            self._type_badge.setText(self.tr("图片序列"))
            self._type_badge.setStyleSheet(f"""
                QLabel {{
                    background-color: {Theme.WARNING};
                    color: #1a1a1a;
                    border-radius: 12px;
                    padding: 2px 12px;
                    font-size: {Theme.FONT_SIZE_SM};
                    font-weight: bold;
                }}
            """)
        else:
            self._type_badge.setText(self.tr("视频"))
            self._type_badge.setStyleSheet(f"""
                QLabel {{
                    background-color: {Theme.ACCENT};
                    color: white;
                    border-radius: 12px;
                    padding: 2px 12px;
                    font-size: {Theme.FONT_SIZE_SM};
                    font-weight: bold;
                }}
            """)

        # Filename / folder name
        frame_count = len(self._image_frames)
        if input_type == "image_sequence" and file_path.is_dir():
            self._filename_label.setText(file_path.name)
            self._detail_label.setText(self.tr("{} 帧").format(frame_count))
        elif input_type == "image_sequence" and file_path.is_file():
            parent = file_path.parent
            self._filename_label.setText(parent.name)
            self._detail_label.setText(
                self.tr("{} 帧 · {}").format(frame_count, file_path.name)
            )
        else:
            self._filename_label.setText(file_path.name)
            try:
                size_bytes = file_path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                if size_mb >= 1024:
                    self._detail_label.setText(f"{size_mb / 1024:.2f} GB")
                else:
                    self._detail_label.setText(f"{size_mb:.2f} MB")
            except OSError:
                self._detail_label.setText("")

        self._type_badge.show()
        self._filename_label.show()
        self._detail_label.show()
        self._change_btn.show()
        self._clear_btn.show()

        self.layout().addWidget(self._type_badge)
        self.layout().addWidget(self._filename_label)
        self.layout().addWidget(self._detail_label)
        self.layout().addLayout(self._loaded_btn_row)
        self.layout().addStretch()

        self.setMinimumHeight(120)
        self.setProperty("loaded", "true")
        self.setStyle(self.style())

    # ====================
    # Auto-detection Logic
    # ====================

    def _process_path(self, path: str):
        """Process a path: auto-detect video vs image sequence, update ViewModel."""
        p = Path(path)
        if not p.exists():
            return

        ext = p.suffix.lower()

        # Case 1: Video file
        if p.is_file() and ext in VIDEO_EXTENSIONS:
            self._image_frames = []
            self._vm.set_video_path(path)
            self.source_selected.emit(path)
            return

        # Case 2: Single image file → detect as image sequence from its folder
        if p.is_file() and ext in IMAGE_EXTENSIONS:
            self._scan_image_sequence(p.parent)
            return

        # Case 3: Directory → scan for image sequences
        if p.is_dir():
            self._scan_image_sequence(p)
            return

    def _scan_image_sequence(self, folder: Path):
        """Scan a folder for image sequences and update ViewModel."""
        try:
            files = get_image_sequence_files(folder, sort=True)
        except ValueError:
            return

        if not files:
            return

        self._image_frames = [str(f) for f in files]

        # Validate sequence pattern
        pattern, prefix, padding, sep = detect_image_sequence_pattern(files)

        # Set the folder path as video_path (backend uses this to find frames)
        self._vm.set_video_path(str(folder))
        self.source_selected.emit(str(folder))

    # ====================
    # File Dialogs
    # ====================

    def _open_video_dialog(self):
        """Open file dialog for video or image selection."""
        video_ext = " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))
        image_ext = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS))
        filter_str = (
            f"{self.tr('媒体文件')} ({video_ext} {image_ext})"
            f";;{self.tr('视频文件')} ({video_ext})"
            f";;{self.tr('图片文件')} ({image_ext})"
            ";;All Files (*)"
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("选择视频或图片"),
            "",
            filter_str,
        )
        if path:
            self._process_path(path)

    def _open_folder_dialog(self):
        """Open folder dialog for image sequence selection."""
        folder = QFileDialog.getExistingDirectory(
            self,
            self.tr("选择图片序列文件夹"),
        )
        if folder:
            self._process_path(folder)

    def _clear_source(self):
        """Clear current source selection."""
        self._image_frames = []
        self._vm.set_video_path("")
        self.source_selected.emit("")

    # ====================
    # ViewModel Handlers
    # ====================

    def _on_video_path_changed(self, path: str):
        self._update_display(path, self._vm.input_type)

    def _on_input_type_changed(self, input_type: str):
        self._update_display(self._vm.video_path, input_type)

    def _update_display(self, path: str, input_type: str):
        if path and Path(path).exists():
            self._show_loaded_mode(path, input_type)
        else:
            self._image_frames = []
            self._show_placeholder_mode()

    # ====================
    # Drag and Drop
    # ====================

    def dragEnterEvent(self, a0: QDragEnterEvent | None):
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is None:
            return

        if mime.hasUrls():
            for url in mime.urls():
                path = Path(url.toLocalFile())
                # Accept: video files, image files, directories
                if path.is_dir():
                    a0.acceptProposedAction()
                    return
                ext = path.suffix.lower()
                if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                    a0.acceptProposedAction()
                    return
        a0.ignore()

    def dragLeaveEvent(self, a0: QDragLeaveEvent | None):
        super().dragLeaveEvent(a0)

    def dropEvent(self, a0: QDropEvent | None):
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is None:
            a0.ignore()
            return

        for url in mime.urls():
            path = url.toLocalFile()
            if path:
                self._process_path(path)
                a0.acceptProposedAction()
                return
        a0.ignore()

    # ====================
    # i18n
    # ====================

    def changeEvent(self, a0: QEvent | None):
        if a0 and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)

    def _retranslate_ui(self):
        if hasattr(self, "_placeholder_label"):
            self._placeholder_label.setText(
                self.tr("拖放视频、图片或图片文件夹到此处")
            )
            self._browse_video_btn.setText(self.tr("选择文件..."))
            self._browse_folder_btn.setText(self.tr("选择图片文件夹..."))
            self._change_btn.setText(self.tr("更改..."))
            self._clear_btn.setText(self.tr("清除"))


__all__ = ["InputSourceWidget"]
