"""Progress monitoring panel with progress bar, FPS, ETA, and log viewer."""

import time
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, QDateTime
from PyQt6.QtGui import QTextCursor

from core import tr


class ProgressPanel(QWidget):
    """Widget for displaying processing progress and logs."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._start_time: Optional[float] = None
        self._current_frame: int = 0
        self._total_frames: int = 0
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Progress group
        self.progress_group = QGroupBox(tr("Progress"))
        progress_layout = QVBoxLayout(self.progress_group)

        # Stage label
        stage_layout = QHBoxLayout()
        stage_layout.addWidget(QLabel(tr("Stage:")))
        self.stage_label = QLabel(tr("Idle"))
        self.stage_label.setStyleSheet("color: #0078d4; font-weight: bold;")
        stage_layout.addWidget(self.stage_label, 1)
        progress_layout.addLayout(stage_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v / %m frames)")
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Stats row
        stats_layout = QHBoxLayout()

        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel(tr("FPS:")))
        self.fps_label = QLabel("--")
        self.fps_label.setStyleSheet("font-family: monospace;")
        fps_layout.addWidget(self.fps_label)
        stats_layout.addLayout(fps_layout)

        # ETA
        eta_layout = QHBoxLayout()
        eta_layout.addWidget(QLabel(tr("ETA:")))
        self.eta_label = QLabel("--:--:--")
        self.eta_label.setStyleSheet("font-family: monospace;")
        eta_layout.addWidget(self.eta_label)
        stats_layout.addLayout(eta_layout)

        # Elapsed
        elapsed_layout = QHBoxLayout()
        elapsed_layout.addWidget(QLabel(tr("Elapsed:")))
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet("font-family: monospace;")
        elapsed_layout.addWidget(self.elapsed_label)
        stats_layout.addLayout(elapsed_layout)

        stats_layout.addStretch()

        # Cancel button
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        stats_layout.addWidget(self.cancel_btn)

        progress_layout.addLayout(stats_layout)
        layout.addWidget(self.progress_group)

        # Log viewer group
        self.log_group = QGroupBox(tr("Log Output"))
        log_layout = QVBoxLayout(self.log_group)

        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setFontFamily("Consolas, Courier New, monospace")
        self.log_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        log_layout.addWidget(self.log_viewer)

        # Clear log button
        clear_layout = QHBoxLayout()
        clear_layout.addStretch()
        self.clear_log_btn = QPushButton(tr("Clear Log"))
        self.clear_log_btn.setFixedWidth(100)
        self.clear_log_btn.clicked.connect(self.clear_log)
        clear_layout.addWidget(self.clear_log_btn)
        log_layout.addLayout(clear_layout)

        layout.addWidget(self.log_group)
    
    def retranslate_ui(self):
        """Retranslate UI elements after language change."""
        self.progress_group.setTitle(tr("Progress"))
        self.log_group.setTitle(tr("Log Output"))
        self.cancel_btn.setText(tr("Cancel"))
        self.clear_log_btn.setText(tr("Clear Log"))

    def update_progress(self, current: int, total: int, fps: float):
        """Update progress display."""
        if self._start_time is None:
            self._start_time = time.time()

        self._current_frame = current
        self._total_frames = total

        # Update progress bar
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

        # Update FPS
        self.fps_label.setText(f"{fps:.2f}")

        # Update elapsed time
        elapsed = time.time() - self._start_time
        self.elapsed_label.setText(self._format_time(elapsed))

        # Update ETA
        if fps > 0 and current > 0:
            remaining_frames = total - current
            eta_seconds = remaining_frames / fps
            self.eta_label.setText(self._format_time(eta_seconds))

    def set_stage(self, stage: str):
        """Set the current processing stage."""
        self.stage_label.setText(stage)

        if stage == tr("Initializing..."):
            self._start_time = None
            self.progress_bar.setValue(0)
            self.fps_label.setText("--")
            self.eta_label.setText("--:--:--")
            self.elapsed_label.setText("00:00:00")

    def append_log(self, message: str):
        """Append a message to the log viewer."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted = f"[{timestamp}] {message}"

        self.log_viewer.append(formatted)
        # Auto-scroll to bottom
        cursor = self.log_viewer.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_viewer.setTextCursor(cursor)

    def clear_log(self):
        """Clear the log viewer."""
        self.log_viewer.clear()

    def reset(self):
        """Reset all progress indicators."""
        self._start_time = None
        self._current_frame = 0
        self._total_frames = 0
        self.stage_label.setText(tr("Idle"))
        self.progress_bar.setValue(0)
        self.fps_label.setText("--")
        self.eta_label.setText("--:--:--")
        self.elapsed_label.setText("00:00:00")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
