"""TaskDetailsTabs - Bottom panel with tabbed details (qBittorrent-style).

Contains tabs for:
- General: task summary info (video, model, settings)
- Progress: progress bar + FPS + ETA + frame info
- Log: real-time log viewer
- GPU: hardware monitor
"""

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel,
    QGridLayout, QGroupBox,
)

from ui.styles.theme import Theme
from ui.widgets.progress.task_log import TaskLog
from ui.widgets.progress.gpu_monitor import HwMonitorPlaceholder
from ui.widgets.progress.progress_bar import ProgressBar

if TYPE_CHECKING:
    from ui.viewmodels import ViewModelContainer
    from ui.viewmodels.queue_viewmodel import QueueItemVO


class GeneralTab(QWidget):
    """General info tab — shows task summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Theme.PADDING_MD, Theme.PADDING_MD,
                                  Theme.PADDING_MD, Theme.PADDING_MD)
        layout.setSpacing(Theme.SPACING_MD)

        grid = QGridLayout()
        grid.setSpacing(Theme.SPACING_MD)

        labels = [
            ("video_label", self.tr("视频:")),
            ("model_label", self.tr("模型:")),
            ("multi_label", self.tr("倍率:")),
            ("scene_label", self.tr("场景检测:")),
            ("output_label", self.tr("输出:")),
            ("codec_label", self.tr("编码器:")),
        ]

        self._value_labels: dict[str, QLabel] = {}

        for row, (key, text) in enumerate(labels):
            label = QLabel(text)
            label.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.TEXT_SECONDARY};
                    font-size: {Theme.FONT_SIZE_MD};
                }}
            """)
            grid.addWidget(label, row, 0)

            value = QLabel("-")
            value.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.TEXT_PRIMARY};
                    font-size: {Theme.FONT_SIZE_MD};
                }}
            """)
            grid.addWidget(value, row, 1)
            self._value_labels[key] = value

        layout.addLayout(grid)
        layout.addStretch()

    def update_info(self, item: "QueueItemVO | None"):
        """Update info display from queue item."""
        if item is None:
            for label in self._value_labels.values():
                label.setText("-")
            return

        self._value_labels["video_label"].setText(item.video_name)
        self._value_labels["model_label"].setText("-")
        self._value_labels["multi_label"].setText("-")
        self._value_labels["scene_label"].setText("-")
        self._value_labels["output_label"].setText("-")
        self._value_labels["codec_label"].setText("-")


class ProgressTab(QWidget):
    """Progress tab — shows detailed progress info."""

    def __init__(self, vms: "ViewModelContainer", parent=None):
        super().__init__(parent)
        self._vms = vms
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Theme.PADDING_MD, Theme.PADDING_MD,
                                  Theme.PADDING_MD, Theme.PADDING_MD)
        layout.setSpacing(Theme.SPACING_MD)

        self._progress_widget = ProgressBar(self._vms.task, self)
        layout.addWidget(self._progress_widget)

        grid = QGridLayout()
        grid.setSpacing(Theme.SPACING_MD)

        detail_items = [
            ("frame_label", self.tr("帧:")),
            ("fps_label", self.tr("速度:")),
            ("eta_label", self.tr("剩余:")),
            ("scene_cuts_label", self.tr("场景切换:")),
            ("skipped_label", self.tr("跳过帧:")),
        ]

        self._detail_values: dict[str, QLabel] = {}

        for row, (key, text) in enumerate(detail_items):
            label = QLabel(text)
            label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: {Theme.FONT_SIZE_MD};")
            grid.addWidget(label, row, 0)

            value = QLabel("-")
            value.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-size: {Theme.FONT_SIZE_MD};")
            grid.addWidget(value, row, 1)
            self._detail_values[key] = value

        layout.addLayout(grid)
        layout.addStretch()

        self._vms.task.current_frame_changed.connect(self._on_frame_changed)
        self._vms.task.total_frames_changed.connect(self._on_frame_changed)
        self._vms.task.fps_changed.connect(self._on_fps_changed)
        self._vms.task.eta_changed.connect(self._on_eta_changed)
        self._vms.task.scene_cuts_changed.connect(self._on_scene_cuts_changed)
        self._vms.task.skipped_frames_changed.connect(self._on_skipped_changed)

    def _on_frame_changed(self, _=None):
        current = self._vms.task.current_frame
        total = self._vms.task.total_frames
        self._detail_values["frame_label"].setText(f"{current} / {total}")

    def _on_fps_changed(self, fps: float):
        self._detail_values["fps_label"].setText(f"{fps:.1f} fps")

    def _on_eta_changed(self, eta: str):
        self._detail_values["eta_label"].setText(eta if eta else "-")

    def _on_scene_cuts_changed(self, cuts: int):
        self._detail_values["scene_cuts_label"].setText(str(cuts))

    def _on_skipped_changed(self, skipped: int):
        self._detail_values["skipped_label"].setText(str(skipped))


class TaskDetailsTabs(QWidget):
    """Bottom panel with tabbed task details (qBittorrent-style).

    Layout:
        ┌──────────────────────────────────────────┐
        │ [通用] [进度] [日志] [GPU]               │
        ├──────────────────────────────────────────┤
        │  Tab content                             │
        └──────────────────────────────────────────┘
    """

    def __init__(self, vms: "ViewModelContainer", parent=None):
        super().__init__(parent)
        self._vms = vms
        self._setup_ui()
        self._retranslate_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {Theme.BORDER};
                background-color: {Theme.BG_PRIMARY};
                border-radius: {Theme.RADIUS_SM}px;
            }}
            QTabBar::tab {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_SECONDARY};
                padding: 6px 16px;
                border: 1px solid {Theme.BORDER};
                border-bottom: none;
                border-top-left-radius: {Theme.RADIUS_MD}px;
                border-top-right-radius: {Theme.RADIUS_MD}px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {Theme.BG_PRIMARY};
                color: {Theme.TEXT_PRIMARY};
                border-bottom: 2px solid {Theme.ACCENT};
            }}
            QTabBar::tab:hover {{
                background-color: {Theme.BG_HOVER};
                color: {Theme.TEXT_PRIMARY};
            }}
        """)

        self._general_tab = GeneralTab(self)
        self._tabs.addTab(self._general_tab, self.tr("通用"))

        self._progress_tab = ProgressTab(self._vms, self)
        self._tabs.addTab(self._progress_tab, self.tr("进度"))

        self._log_tab = TaskLog(self._vms.task, self)
        self._tabs.addTab(self._log_tab, self.tr("日志"))

        self._gpu_tab = HwMonitorPlaceholder(self._vms.task, self)
        self._tabs.addTab(self._gpu_tab, self.tr("GPU"))

        layout.addWidget(self._tabs)

    def update_item(self, item: "QueueItemVO | None"):
        """Update details for selected item."""
        self._general_tab.update_info(item)

    def _retranslate_ui(self):
        """Update UI text for i18n."""
        if hasattr(self, "_tabs"):
            self._tabs.setTabText(0, self.tr("通用"))
            self._tabs.setTabText(1, self.tr("进度"))
            self._tabs.setTabText(2, self.tr("日志"))
            self._tabs.setTabText(3, self.tr("GPU"))

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)


__all__ = ["TaskDetailsTabs", "GeneralTab", "ProgressTab"]
