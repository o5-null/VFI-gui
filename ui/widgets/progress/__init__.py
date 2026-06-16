"""Progress widgets package — ProcessPage progress monitoring widgets.

This package contains widgets for progress display and monitoring:
- ProgressBar: Progress bar with FPS, ETA, frame info
- TaskLog: Real-time log viewer with auto-scroll
- HwMonitorPlaceholder: GPU monitoring placeholder
"""

from ui.widgets.progress.progress_bar import ProgressBar
from ui.widgets.progress.task_log import TaskLog
from ui.widgets.progress.gpu_monitor import HwMonitorPlaceholder

__all__ = ["ProgressBar", "TaskLog", "HwMonitorPlaceholder"]