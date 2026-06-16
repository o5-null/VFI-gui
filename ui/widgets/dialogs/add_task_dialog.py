"""AddTaskDialog - Modal dialog for adding a new task to the queue.

Combines InputSourceWidget + InterpolationGroup + SceneDetectGroup + OutputGroup
into a single dialog. User configures settings and clicks OK to add the task.

InputSourceWidget auto-detects video files vs image sequences.

Layout:
    ┌────────────────────────────────────────────────────────────┐
    │  添加任务                                                   │
    ├────────────────────────────────────────────────────────────┤
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │  InputSourceWidget (视频/图片序列 自动检测)            │  │
    │  ├──────────────────────────────────────────────────────┤  │
    │  │  InterpolationGroup                                  │  │
    │  ├──────────────────────────────────────────────────────┤  │
    │  │  SceneDetectGroup                                    │  │
    │  ├──────────────────────────────────────────────────────┤  │
    │  │  OutputGroup                                         │  │
    │  └──────────────────────────────────────────────────────┘  │
    ├────────────────────────────────────────────────────────────┤
    │                    [添加到队列]  [开始处理]  [取消]         │
    └────────────────────────────────────────────────────────────┘
"""

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
    QScrollArea, QFrame, QMessageBox, QSizePolicy,
)

from ui.styles.theme import Theme
from ui.styles.icons import IconManager
from ui.widgets.input_source_widget import InputSourceWidget
from ui.widgets.pipeline.interpolation_group import InterpolationGroup
from ui.widgets.pipeline.scene_detect_group import SceneDetectGroup
from ui.widgets.pipeline.output_group import OutputGroup

if TYPE_CHECKING:
    from ui.viewmodels import ViewModelContainer
    from ui.controllers import ControllerContainer


class AddTaskDialog(QDialog):
    """Modal dialog for adding a new task.

    Uses the shared PipelineViewModel and CodecViewModel for settings.
    On accept: persists config and either adds to queue or starts processing.

    Signals (via return code):
        Accepted with "queue": Add to queue
        Accepted with "start": Start processing immediately
    """

    def __init__(
        self,
        vms: "ViewModelContainer",
        ctrls: "ControllerContainer",
        parent=None,
    ):
        super().__init__(parent)
        self._vms = vms
        self._ctrls = ctrls
        self._action = "queue"  # "queue" or "start"

        self._setup_ui()
        self._retranslate_ui()

    @property
    def action(self) -> str:
        """Get the user's chosen action: 'queue' or 'start'."""
        return self._action

    def _setup_ui(self):
        """Build the dialog layout."""
        self.setMinimumSize(520, 620)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {Theme.BG_PRIMARY};
            }}
        """)

        # Content widget
        content = QWidget()
        content.setStyleSheet(f"background-color: {Theme.BG_PRIMARY};")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(
            Theme.PADDING_XL, Theme.PADDING_LG,
            Theme.PADDING_XL, Theme.PADDING_LG,
        )
        content_layout.setSpacing(Theme.SPACING_LG)

        # InputSourceWidget (auto-detects video / image sequence)
        self._input_source = InputSourceWidget(self._vms.pipeline, self)
        content_layout.addWidget(self._input_source)

        # InterpolationGroup
        self._interp_group = InterpolationGroup(self._vms.pipeline, self)
        content_layout.addWidget(self._interp_group)

        # SceneDetectGroup
        self._scene_detect_group = SceneDetectGroup(self._vms.pipeline, self)
        content_layout.addWidget(self._scene_detect_group)

        # OutputGroup
        self._output_group = OutputGroup(self._vms.pipeline, self._vms.codec, self)
        content_layout.addWidget(self._output_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)

        # Bottom button bar
        btn_bar = QFrame()
        btn_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_SECONDARY};
                border-top: 1px solid {Theme.BORDER};
            }}
        """)
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(
            Theme.PADDING_LG, Theme.PADDING_MD,
            Theme.PADDING_LG, Theme.PADDING_MD,
        )
        btn_layout.setSpacing(Theme.SPACING_MD)

        # Add to Queue button
        self._add_queue_btn = QPushButton()
        self._add_queue_btn.setObjectName("addTaskQueueBtn")
        self._add_queue_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                border-color: {Theme.BORDER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_PRESSED};
            }}
        """)
        self._add_queue_btn.clicked.connect(self._on_add_queue)
        btn_layout.addWidget(self._add_queue_btn)

        # Start Processing button
        self._start_btn = QPushButton()
        self._start_btn.setObjectName("addTaskStartBtn")
        self._start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.ACCENT};
                color: {Theme.TEXT_PRIMARY};
                border: none;
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {Theme.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {Theme.ACCENT_PRESSED};
            }}
        """)
        self._start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self._start_btn)

        btn_layout.addStretch()

        # Cancel button
        self._cancel_btn = QPushButton()
        self._cancel_btn.setObjectName("addTaskCancelBtn")
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                color: {Theme.TEXT_PRIMARY};
            }}
        """)
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)

        main_layout.addWidget(btn_bar)

        # Dialog-level stylesheet
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Theme.BG_PRIMARY};
            }}
        """)

    def _validate(self) -> bool:
        """Validate current configuration. Shows warning if invalid."""
        errors = self._vms.pipeline.validate()
        if errors:
            QMessageBox.warning(
                self,
                self.tr("验证错误"),
                "\n".join(errors),
            )
            return False

        video_path = self._vms.pipeline.video_path
        if not video_path:
            QMessageBox.warning(
                self,
                self.tr("未选择视频"),
                self.tr("请先选择一个视频文件。"),
            )
            return False

        return True

    def _on_add_queue(self):
        """Add to queue and close."""
        if not self._validate():
            return

        self._vms.pipeline.persist()
        self._action = "queue"
        self.accept()

    def _on_start(self):
        """Start processing and close."""
        if not self._validate():
            return

        self._vms.pipeline.persist()
        self._action = "start"
        self.accept()

    def _retranslate_ui(self):
        """Update UI text for i18n."""
        self.setWindowTitle(self.tr("添加任务"))
        if hasattr(self, "_add_queue_btn"):
            self._add_queue_btn.setText(self.tr("添加到队列"))
            self._start_btn.setText(self.tr("开始处理"))
            self._cancel_btn.setText(self.tr("取消"))

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)


__all__ = ["AddTaskDialog"]
