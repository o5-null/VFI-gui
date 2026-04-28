"""Processing Worker for VFI-gui.

.. deprecated::
    Use :class:`TaskOrchestrator <core.task_orchestrator.TaskOrchestrator>` instead.
    The ProcessingWorker class is kept for backward compatibility only.

Background worker for video processing operations.
"""

import warnings
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal

from core import Processor


class ProcessingWorker(QThread):
    """Worker thread for video processing.

    .. deprecated:: Use TaskOrchestrator instead.

    Provides progress updates and cancellation support.
    """

    # Signals
    progress = pyqtSignal(int, int, float)  # frame, total, fps
    finished = pyqtSignal(bool, str)  # success, message
    error = pyqtSignal(str)  # error message

    def __init__(self, processor: Optional[Processor], video_path: str, output_path: str):
        warnings.warn(
            "ProcessingWorker is deprecated. Use TaskOrchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self._processor = processor
        self._video_path = video_path
        self._output_path = output_path
        self._is_cancelled = False

    def run(self):
        """Execute processing."""
        try:
            # This would call the actual processing
            # For now, just a placeholder structure
            self.finished.emit(True, "Processing completed")
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False, str(e))

    def cancel(self):
        """Request cancellation."""
        self._is_cancelled = True
        if self._processor:
            self._processor.cancel()