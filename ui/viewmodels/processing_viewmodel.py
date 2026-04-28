"""Processing ViewModel for VFI-gui.

.. deprecated::
    Use :class:`ProcessingController <ui.controllers.processing_controller.ProcessingController>`
    instead. The ProcessingViewModel class is kept for backward compatibility only.

Acts as intermediary between processing UI and core processing logic,
providing a clean interface for video processing operations.
"""

import warnings
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal

from core import Processor, ProcessingConfig, BackendType
from core.workers import ProcessingWorker


class ProcessingViewModel(QObject):
    """ViewModel for video processing operations.

    .. deprecated:: Use ProcessingController instead.

    Decouples processing UI from core Processor implementation.
    Manages processing state and worker thread.
    """
    
    # Signals for UI updates
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(bool, str)  # success, message
    processing_cancelled = pyqtSignal()
    progress_updated = pyqtSignal(int, int, float)  # frame, total, fps
    error_occurred = pyqtSignal(str)
    state_changed = pyqtSignal(str)  # new state
    
    # States
    STATE_IDLE = "idle"
    STATE_PROCESSING = "processing"
    STATE_CANCELLING = "cancelling"
    STATE_ERROR = "error"
    
    def __init__(self, parent=None):
        warnings.warn(
            "ProcessingViewModel is deprecated. Use ProcessingController instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(parent)
        self._processor: Optional[Processor] = None
        self._worker: Optional[ProcessingWorker] = None
        self._state = self.STATE_IDLE
        self._current_video: Optional[str] = None
        self._current_output: Optional[str] = None
    
    def get_state(self) -> str:
        """Get current processing state."""
        return self._state
    
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._state == self.STATE_PROCESSING
    
    def can_start(self) -> bool:
        """Check if processing can be started."""
        return self._state == self.STATE_IDLE
    
    def can_cancel(self) -> bool:
        """Check if processing can be cancelled."""
        return self._state == self.STATE_PROCESSING
    
    def _set_state(self, state: str) -> None:
        """Update state and notify."""
        self._state = state
        self.state_changed.emit(state)
    
    def start_processing(
        self,
        video_path: str,
        output_path: str,
        config: ProcessingConfig,
        backend_type: BackendType = BackendType.TORCH,
    ) -> bool:
        """Start video processing.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            config: Processing configuration
            backend_type: Backend to use
            
        Returns:
            True if started successfully
        """
        if not self.can_start():
            return False
        
        try:
            # Create processor with config
            self._processor = Processor(
                config=config,
                backend_type=backend_type,
            )
            
            self._current_video = video_path
            self._current_output = output_path
            
            # Create and start worker thread
            self._worker = ProcessingWorker(
                self._processor,
                video_path,
                output_path,
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.finished.connect(self._on_finished)
            self._worker.error.connect(self._on_error)
            
            self._set_state(self.STATE_PROCESSING)
            self.processing_started.emit()
            self._worker.start()
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            self._set_state(self.STATE_ERROR)
            return False
    
    def cancel_processing(self) -> bool:
        """Request processing cancellation.
        
        Returns:
            True if cancellation was requested
        """
        if not self.can_cancel():
            return False
        
        self._set_state(self.STATE_CANCELLING)
        
        if self._worker:
            self._worker.cancel()
        
        self.processing_cancelled.emit()
        return True
    
    def _on_progress(self, frame: int, total: int, fps: float):
        """Handle progress update from worker."""
        self.progress_updated.emit(frame, total, fps)
    
    def _on_finished(self, success: bool, message: str):
        """Handle processing completion."""
        self._set_state(self.STATE_IDLE)
        self.processing_finished.emit(success, message)
        self._cleanup()
    
    def _on_error(self, error_msg: str):
        """Handle processing error."""
        self._set_state(self.STATE_ERROR)
        self.error_occurred.emit(error_msg)
    
    def _cleanup(self):
        """Clean up resources."""
        self._worker = None
        self._processor = None
