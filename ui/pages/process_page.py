"""ProcessPage - Processing view for VFI-gui.

Displayed when a task is running. Contains:
- Action buttons (Pause, Stop, Back)
- Current task info (video name)
- ProgressBar
- GPUMonitor
- QueueList
- TaskLog

Layout uses QSplitter for middle section.
"""

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QPushButton,
)

from ui.styles import Theme
from ui.widgets.progress.gpu_monitor import HwMonitorPlaceholder
from ui.widgets.progress.progress_bar import ProgressBar
from ui.widgets.progress.task_log import TaskLog
from ui.widgets.queue.queue_list import QueueList

if TYPE_CHECKING:
    from ui.viewmodels import ViewModelContainer
    from ui.controllers import ControllerContainer


class ProcessPage(QWidget):
    """Processing view — displayed when task is running.
    
    Layout:
        ┌──────────────────────────────────────────────────────────────┐
        │  [⏸ Pause]  [⏹ Stop]  [◀ Back to Config]                    │
        ├──────────────────────────────────────────────────────────────┤
        │  📊 Current Task: video2.mp4                                 │
        │  ProgressBar (progress + FPS + ETA)                          │
        ├──────────────────────────┬───────────────────────────────────┤
        │  HwMonitorPlaceholder    │  QueueList                        │
        │  (future: GPU monitor)   │                                   │
        ├──────────────────────────┴───────────────────────────────────┤
        │  TaskLog                                                   │
        └──────────────────────────────────────────────────────────────┘
    
    Uses QSplitter for the middle section.
    All user-visible text uses self.tr() for i18n.
    """
    
    def __init__(
        self,
        vms: "ViewModelContainer",
        ctrls: "ControllerContainer",
        parent=None,
    ):
        """Initialize ProcessPage.
        
        Args:
            vms: ViewModelContainer with pipeline, task, queue, device, codec
            ctrls: ControllerContainer with processing, queue, settings
            parent: Parent widget
        """
        super().__init__(parent)
        self._vms = vms
        self._ctrls = ctrls
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Setup UI layout."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
        )
        main_layout.setSpacing(Theme.SPACING_MD)
        
        # Top action buttons
        self._setup_action_buttons(main_layout)
        
        # Task info section
        self._setup_task_info(main_layout)
        
        # Progress bar widget (replaces basic QProgressBar)
        self._progress_widget = ProgressBar(self._vms.task, self)
        main_layout.addWidget(self._progress_widget)
        
        # Middle splitter section
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        
        # HwMonitor (real GPU monitor widget)
        hw_monitor = HwMonitorPlaceholder(self._vms.task, self)
        splitter.addWidget(hw_monitor)
        
        # QueueList (real queue widget)
        queue_list = QueueList(self._vms.queue, self._ctrls.queue, self)
        splitter.addWidget(queue_list)
        
        # Set stretch factors
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter, 1)
        
        # TaskLog (real log widget)
        task_log = TaskLog(self._vms.task, self)
        main_layout.addWidget(task_log)
        
    def _setup_action_buttons(self, layout: QVBoxLayout) -> None:
        """Setup action button row at top."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(Theme.SPACING_MD)
        
        # Pause button
        self._pause_btn = QPushButton()
        self._pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                border-color: {Theme.BORDER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_PRESSED};
            }}
        """)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        button_layout.addWidget(self._pause_btn)
        
        # Stop button
        self._stop_btn = QPushButton()
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.ERROR};
                color: {Theme.TEXT_PRIMARY};
                border: none;
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #b82020;
            }}
            QPushButton:pressed {{
                background-color: #9a1818;
            }}
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self._stop_btn)
        
        # Back to Config button
        self._back_btn = QPushButton()
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                border-color: {Theme.BORDER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_PRESSED};
            }}
        """)
        self._back_btn.clicked.connect(self._on_back_clicked)
        button_layout.addWidget(self._back_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def _setup_task_info(self, layout: QVBoxLayout) -> None:
        """Setup task info section."""
        info_layout = QHBoxLayout()
        info_layout.setSpacing(Theme.SPACING_SM)
        
        # Task icon
        icon_label = QLabel("📊")
        icon_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {Theme.FONT_SIZE_LG};
            }}
        """)
        info_layout.addWidget(icon_label)
        
        # Task name label
        self._task_name_label = QLabel()
        self._task_name_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {Theme.FONT_SIZE_LG};
                font-weight: bold;
            }}
        """)
        info_layout.addWidget(self._task_name_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
    def _connect_signals(self) -> None:
        """Connect ViewModel signals."""
        # Task name and state signals (ProgressBar binds to TaskVM internally)
        self._vms.task.video_name_changed.connect(self._on_video_name_changed)
        self._vms.task.state_changed.connect(self._on_state_changed)
        
    def _on_video_name_changed(self, name: str) -> None:
        """Handle video name change."""
        self._task_name_label.setText(
            self.tr("Current Task: {0}").format(name) if name else self.tr("No task running")
        )
        
    def _on_state_changed(self, state: str) -> None:
        """Handle task state change.
        
        Updates button states based on current task state.
        """
        # Update pause button
        if state == "paused":
            self._pause_btn.setText(self.tr("▶ Resume"))
        elif state in ("loading", "processing"):
            self._pause_btn.setText(self.tr("⏸ Pause"))
            self._pause_btn.setEnabled(True)
        else:
            self._pause_btn.setEnabled(False)
            
        # Update stop button
        self._stop_btn.setEnabled(state in ("loading", "processing", "paused"))
        
        # Update back button
        self._back_btn.setEnabled(state in ("completed", "failed", "cancelled", "idle"))
        
    # ====================
    # Button Handlers
    # ====================
    
    def _on_pause_clicked(self) -> None:
        """Handle Pause/Resume button click."""
        state = self._vms.task.state
        
        if state == "paused":
            self._ctrls.processing.resume_task()
        elif state in ("loading", "processing"):
            self._ctrls.processing.pause_task()
            
    def _on_stop_clicked(self) -> None:
        """Handle Stop button click."""
        self._ctrls.processing.cancel_task()
        
    def _on_back_clicked(self) -> None:
        """Handle Back to Config button click.
        
        Returns to ConfigPage and resets task state.
        """
        # Cancel current task if still running
        if self._vms.task.state in ("loading", "processing", "paused"):
            self._ctrls.processing.cancel_task()
        
        # Reset task viewmodel
        self._vms.task.reset()
        
    # ====================
    # i18n Support
    # ====================
    
    def changeEvent(self, a0: QEvent | None) -> None:  # noqa: ARG002
        """Handle change events (language change)."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)
        
    def _retranslate_ui(self) -> None:
        """Retranslate UI text for current locale."""
        self._pause_btn.setText(self.tr("⏸ Pause"))
        self._stop_btn.setText(self.tr("⏹ Stop"))
        self._back_btn.setText(self.tr("◀ Back to Config"))
        
        # Update task name label
        video_name = self._vms.task.video_name
        if video_name:
            self._task_name_label.setText(self.tr("Current Task: {0}").format(video_name))
        else:
            self._task_name_label.setText(self.tr("No task running"))


__all__ = ["ProcessPage"]