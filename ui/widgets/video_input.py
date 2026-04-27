"""Video file input widget with drag & drop support."""

import os
import re
from pathlib import Path
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QFrame,
    QGroupBox,
    QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QAction

from core import tr


class VideoInputWidget(QWidget):
    """Widget for video file selection with drag & drop support."""

    video_selected = pyqtSignal(str)
    input_type_changed = pyqtSignal(bool)  # True if image sequence, False if video file

    # Supported video extensions
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv", ".ts", ".m2ts"}
    
    # Supported image extensions for image sequences
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".exr", ".dpx"}

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._video_path: str = ""
        self._is_image_sequence: bool = False
        self._image_sequence_pattern: str = ""  # e.g., "image.%04d.png"
        self._image_sequence_frames: List[str] = []  # List of frame files
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Group box
        self.group = QGroupBox(tr("Video Input"))
        group_layout = QVBoxLayout(self.group)

        # File path input row
        path_layout = QHBoxLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(tr("Select a video file, image sequence, or drag & drop here..."))
        self.path_edit.setReadOnly(True)
        path_layout.addWidget(self.path_edit, 1)

        self.browse_btn = QPushButton(tr("Browse..."))
        self.browse_btn.setFixedWidth(100)
        self.browse_btn.clicked.connect(self._on_browse_menu)
        path_layout.addWidget(self.browse_btn)

        group_layout.addLayout(path_layout)

        # Drop zone
        self.drop_frame = QFrame()
        self.drop_frame.setObjectName("dropZone")
        self.drop_frame.setFixedHeight(80)
        self.drop_frame.setAcceptDrops(True)

        drop_layout = QVBoxLayout(self.drop_frame)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.drop_label = QLabel(tr("Drag & Drop Video Files, Folders, or Image Sequences Here"))
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("color: #808080; font-size: 12pt;")
        drop_layout.addWidget(self.drop_label)

        group_layout.addWidget(self.drop_frame)

        # Video info
        self.info_label = QLabel(tr("No video selected"))
        self.info_label.setStyleSheet("color: #808080;")
        group_layout.addWidget(self.info_label)

        layout.addWidget(self.group)

        # Enable drag & drop on the whole widget
        self.setAcceptDrops(True)
    
    def retranslate_ui(self):
        """Retranslate UI elements after language change."""
        self.group.setTitle(tr("Video Input"))
        self.path_edit.setPlaceholderText(tr("Select a video file, image sequence, or drag & drop here..."))
        self.browse_btn.setText(tr("Browse..."))
        self.drop_label.setText(tr("Drag & Drop Video Files, Folders, or Image Sequences Here"))
        if not self._video_path:
            self.info_label.setText(tr("No video selected"))

    def _on_browse_menu(self):
        """Show browse menu with options."""
        menu = QMenu(self)
        
        video_action = QAction(tr("Video File..."), self)
        video_action.triggered.connect(self._on_browse_video)
        menu.addAction(video_action)
        
        sequence_action = QAction(tr("Image Sequence..."), self)
        sequence_action.triggered.connect(self._on_browse_sequence)
        menu.addAction(sequence_action)
        
        menu.exec(self.browse_btn.mapToGlobal(self.browse_btn.rect().bottomLeft()))

    def _on_browse_video(self):
        """Open file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select Video File"),
            "",
            tr("Video Files (*.mp4 *.mkv *.avi *.webm *.mov *.flv *.wmv *.ts *.m2ts);;All Files (*)"),
        )
        if file_path:
            self.set_video_path(file_path)

    def _on_browse_sequence(self):
        """Open file dialog to select an image sequence."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select First Image in Sequence"),
            "",
            tr("Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.exr *.dpx);;All Files (*)"),
        )
        if file_path:
            self._detect_and_set_image_sequence(file_path)

    def _detect_and_set_image_sequence(self, first_image_path: str):
        """Detect image sequence from first image and set it as input.
        
        Uses core.utils.file_utils for robust sequence detection.
        
        Args:
            first_image_path: Path to the first image in the sequence.
        """
        from core.utils.file_utils import (
            get_image_sequence_files,
            detect_image_sequence_pattern,
            parse_frame_number,
            sort_files_naturally,
        )
        
        first_path = Path(first_image_path)
        directory = first_path.parent
        extension = first_path.suffix.lower()
        
        # Get all image files in directory with same extension
        try:
            all_images = get_image_sequence_files(
                directory,
                extensions=[extension],
                sort=True,
            )
        except ValueError:
            self.set_video_path(first_image_path)
            return
        
        if len(all_images) < 2:
            # Not a sequence, treat as single image
            self.set_video_path(first_image_path)
            return
        
        # Detect sequence pattern
        pattern, prefix, padding, separator = detect_image_sequence_pattern(all_images)
        
        if pattern is None:
            # No pattern detected, treat as single image
            self.set_video_path(first_image_path)
            return
        
        # Build pattern string for FFmpeg/OpenCV
        # Format: /path/to/prefix%04d.png (use forward slash for FFmpeg compatibility)
        directory_str = str(directory).replace("\\", "/")
        if prefix:
            # Include separator in the pattern
            pattern_str = f"{directory_str}/{prefix}{separator}%0{padding}d{extension}"
        else:
            pattern_str = f"{directory_str}/%0{padding}d{extension}"
        
        frame_files = [str(f) for f in all_images]
        
        self._is_image_sequence = True
        self._image_sequence_pattern = pattern_str
        self._image_sequence_frames = frame_files
        self._video_path = pattern_str
        
        self.path_edit.setText(pattern_str)
        self._update_sequence_info(frame_files)
        self.video_selected.emit(pattern_str)
        self.input_type_changed.emit(True)  # Image sequence

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            # Check if any of the URLs are video files, images, or folders
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                path_obj = Path(path)
                
                # Accept folders
                if path_obj.is_dir():
                    event.acceptProposedAction()
                    self.drop_frame.setStyleSheet(
                        "QFrame#dropZone { border-color: #0078d4; background-color: #2d2d2d; }"
                    )
                    return
                
                # Accept video files and images
                suffix = path_obj.suffix.lower()
                if suffix in self.VIDEO_EXTENSIONS or suffix in self.IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    self.drop_frame.setStyleSheet(
                        "QFrame#dropZone { border-color: #0078d4; background-color: #2d2d2d; }"
                    )
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.drop_frame.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        self.drop_frame.setStyleSheet("")

        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        
        # Process dropped items
        for url in urls:
            path = url.toLocalFile()
            path_obj = Path(path)
            
            # Handle folder
            if path_obj.is_dir():
                if self._handle_dropped_folder(path):
                    event.acceptProposedAction()
                    return
                continue
            
            # Handle single file
            suffix = path_obj.suffix.lower()
            if suffix in self.VIDEO_EXTENSIONS:
                self.set_video_path(path)
                event.acceptProposedAction()
                return
            elif suffix in self.IMAGE_EXTENSIONS:
                self._detect_and_set_image_sequence(path)
                event.acceptProposedAction()
                return
        
        # Check for image sequence (multiple images dropped)
        image_files = []
        for url in urls:
            path = url.toLocalFile()
            if Path(path).is_file() and Path(path).suffix.lower() in self.IMAGE_EXTENSIONS:
                image_files.append(path)
        
        if len(image_files) > 1:
            image_files.sort()
            self._detect_and_set_image_sequence(image_files[0])
            event.acceptProposedAction()
            return
        
        event.ignore()
    
    def _handle_dropped_folder(self, folder_path: str) -> bool:
        """Handle a dropped folder.
        
        Scans the folder for video files or image sequences.
        Returns True if a valid input was found and set.
        
        Args:
            folder_path: Path to the dropped folder.
            
        Returns:
            True if successfully handled, False otherwise.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            return False
        
        # Find all video files in folder (non-recursive)
        video_files = []
        for ext in self.VIDEO_EXTENSIONS:
            video_files.extend(folder.glob(f"*{ext}"))
        video_files = sorted(set(video_files))
        
        # Find all image files in folder (non-recursive)
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(folder.glob(f"*{ext}"))
        image_files = sorted(set(image_files))
        
        # Priority 1: Image sequence (if multiple images found)
        if len(image_files) >= 2:
            self._detect_and_set_image_sequence(str(image_files[0]))
            return True
        
        # Priority 2: Video files
        if video_files:
            self.set_video_path(str(video_files[0]))
            # Show info if multiple videos found
            if len(video_files) > 1:
                self.info_label.setText(
                    tr("Found {} videos, selected: {}").format(len(video_files), video_files[0].name)
                )
                self.info_label.setStyleSheet("color: #FFA500;")
            return True
        
        # Priority 3: Single image
        if len(image_files) == 1:
            self._detect_and_set_image_sequence(str(image_files[0]))
            return True
        
        # No supported files found
        self.info_label.setText(tr("No supported files found in folder"))
        self.info_label.setStyleSheet("color: #d42a2a;")
        return False

    def _on_browse(self):
        """Open file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select Video File"),
            "",
            tr("Video Files (*.mp4 *.mkv *.avi *.webm *.mov *.flv *.wmv *.ts *.m2ts);;All Files (*)"),
        )
        if file_path:
            self.set_video_path(file_path)

    def set_video_path(self, path: str):
        """Set the video file path."""
        self._video_path = path
        self._is_image_sequence = False
        self._image_sequence_pattern = ""
        self._image_sequence_frames = []
        self.path_edit.setText(path)
        self._update_video_info(path)
        self.video_selected.emit(path)
        self.input_type_changed.emit(False)  # Video file

    def get_video_path(self) -> str:
        """Get the current video file path."""
        return self._video_path
    
    def is_image_sequence(self) -> bool:
        """Check if current input is an image sequence."""
        return self._is_image_sequence
    
    def get_image_sequence_info(self) -> Tuple[str, List[str]]:
        """Get image sequence pattern and frame list.
        
        Returns:
            Tuple of (pattern, frame_list) or ("", []) if not a sequence.
        """
        return self._image_sequence_pattern, self._image_sequence_frames

    def _update_sequence_info(self, frame_files: List[str]):
        """Update info label for image sequence."""
        num_frames = len(frame_files)
        if frame_files:
            first = Path(frame_files[0])
            last = Path(frame_files[-1])
            self.info_label.setText(
                tr("Image Sequence: {} frames ({} to {})")
                .format(num_frames, first.name, last.name)
            )
            self.info_label.setStyleSheet("color: #4CAF50;")

    def _update_video_info(self, path: str):
        """Update the video info label."""
        try:
            video_path = Path(path)
            if video_path.exists():
                size_mb = video_path.stat().st_size / (1024 * 1024)
                self.info_label.setText(
                    tr("File: {} | Size: {:.1f} MB").format(video_path.name, size_mb)
                )
                self.info_label.setStyleSheet("color: #e0e0e0;")
            else:
                self.info_label.setText(tr("File not found"))
                self.info_label.setStyleSheet("color: #d42a2a;")
        except Exception:
            self.info_label.setText(tr("Error reading file info"))
            self.info_label.setStyleSheet("color: #d42a2a;")
