"""ProgressBar — main progress display with FPS, ETA, and frame info.

A composite widget showing processing progress with stats.
Reads TaskViewModel for all progress data.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel
from PyQt6.QtCore import QEvent, Qt

from ui.styles.theme import Theme

if TYPE_CHECKING:
    from ui.viewmodels.task_viewmodel import TaskViewModel


class ProgressBar(QWidget):
    """Main progress display — progress bar + FPS + ETA + frame info.
    
    Reads TaskViewModel for all progress data.
    
    Features:
        - QProgressBar (range 0-100), updated via vm.progress_changed
        - Frame counter: "Frame X / Y"
        - FPS display: "FPS: XX.X"
        - ETA display: "ETA: M:SS"
        - Scene cuts count
        - Skipped frames count
    """

    def __init__(self, vm: "TaskViewModel", parent=None):
        """Initialize ProgressBar.
        
        Args:
            vm: TaskViewModel for progress data binding
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        self.setObjectName("progressBarWidget")
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
        if hasattr(self, "_fps_label"):
            self._fps_label.setText(self.tr("FPS: 0.0"))
            self._eta_label.setText(self.tr("ETA: --"))
            self._scene_cuts_label.setText(self.tr("Scene cuts: 0"))
            self._skipped_label.setText(self.tr("Skipped: 0"))

    def _setup_ui(self) -> None:
        """Create widget structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_MD, Theme.PADDING_MD,
            Theme.PADDING_MD, Theme.PADDING_MD
        )
        layout.setSpacing(Theme.SPACING_MD)

        # Progress bar (0-100 range)
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("progressBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        layout.addWidget(self._progress_bar)

        # Frame info row
        frame_row = QHBoxLayout()
        frame_row.setSpacing(Theme.SPACING_MD)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setObjectName("progressFrameLabel")
        frame_row.addWidget(self._frame_label)

        frame_row.addStretch()

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(Theme.SPACING_MD)

        self._fps_label = QLabel()
        self._fps_label.setObjectName("progressFpsLabel")
        stats_row.addWidget(self._fps_label)

        self._eta_label = QLabel()
        self._eta_label.setObjectName("progressEtaLabel")
        stats_row.addWidget(self._eta_label)

        stats_row.addStretch()

        # Additional stats row
        extra_stats_row = QHBoxLayout()
        extra_stats_row.setSpacing(Theme.SPACING_MD)

        self._scene_cuts_label = QLabel()
        self._scene_cuts_label.setObjectName("progressSceneCutsLabel")
        extra_stats_row.addWidget(self._scene_cuts_label)

        self._skipped_label = QLabel()
        self._skipped_label.setObjectName("progressSkippedLabel")
        extra_stats_row.addWidget(self._skipped_label)

        extra_stats_row.addStretch()

        # Add all rows to main layout
        layout.addLayout(frame_row)
        layout.addLayout(stats_row)
        layout.addLayout(extra_stats_row)

    def _bind_viewmodel(self) -> None:
        """Connect to TaskViewModel signals."""
        # Progress
        self._vm.progress_changed.connect(self._on_progress_changed)
        self._vm.current_frame_changed.connect(self._on_current_frame_changed)
        self._vm.total_frames_changed.connect(self._on_total_frames_changed)

        # Stats
        self._vm.fps_changed.connect(self._on_fps_changed)
        self._vm.eta_changed.connect(self._on_eta_changed)
        self._vm.scene_cuts_changed.connect(self._on_scene_cuts_changed)
        self._vm.skipped_frames_changed.connect(self._on_skipped_changed)

        # Initialize display from current state
        self._update_progress_display()

    def _on_progress_changed(self, progress: float) -> None:
        """Handle progress change."""
        self._progress_bar.setValue(int(progress * 100))

    def _on_current_frame_changed(self, frame: int) -> None:
        """Handle current frame change."""
        self._frame_label.setText(f"{frame} / {self._vm.total_frames}")

    def _on_total_frames_changed(self, total: int) -> None:
        """Handle total frames change."""
        self._frame_label.setText(f"{self._vm.current_frame} / {total}")

    def _on_fps_changed(self, fps: float) -> None:
        """Handle FPS change."""
        self._fps_label.setText(self.tr("FPS: {fps:.1f}").format(fps=fps))

    def _on_eta_changed(self, eta: str) -> None:
        """Handle ETA change."""
        if eta:
            self._eta_label.setText(self.tr("ETA: {eta}").format(eta=eta))
        else:
            self._eta_label.setText(self.tr("ETA: --"))

    def _on_scene_cuts_changed(self, cuts: int) -> None:
        """Handle scene cuts change."""
        self._scene_cuts_label.setText(self.tr("Scene cuts: {cuts}").format(cuts=cuts))

    def _on_skipped_changed(self, skipped: int) -> None:
        """Handle skipped frames change."""
        self._skipped_label.setText(self.tr("Skipped: {skipped}").format(skipped=skipped))

    def _update_progress_display(self) -> None:
        """Initialize display from current ViewModel state."""
        self._progress_bar.setValue(int(self._vm.progress * 100))
        self._frame_label.setText(f"{self._vm.current_frame} / {self._vm.total_frames}")
        self._fps_label.setText(self.tr("FPS: {fps:.1f}").format(fps=self._vm.fps))
        self._eta_label.setText(
            self.tr("ETA: {eta}").format(eta=self._vm.eta) if self._vm.eta
            else self.tr("ETA: --")
        )
        self._scene_cuts_label.setText(self.tr("Scene cuts: {cuts}").format(cuts=self._vm.scene_cuts))
        self._skipped_label.setText(self.tr("Skipped: {skipped}").format(skipped=self._vm.skipped_frames))


__all__ = ["ProgressBar"]