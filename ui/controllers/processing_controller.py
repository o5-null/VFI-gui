"""Processing Controller for VFI-gui.

Handles video processing operations, separating processing logic from MainWindow.
"""

from typing import Optional, Callable
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger

from core import Processor, ProcessingConfig, BackendType
from ui.viewmodels.processing_viewmodel import ProcessingViewModel


class ProcessingController(QObject):
    """Controller for video processing operations.
    
    Encapsulates processing logic that was previously in MainWindow,
    providing a clean interface for starting, monitoring, and cancelling
    video processing tasks.
    """
    
    # Signals forwarded from ViewModel
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(bool, str)  # success, message
    processing_cancelled = pyqtSignal()
    progress_updated = pyqtSignal(int, int, float)  # frame, total, fps
    error_occurred = pyqtSignal(str)
    state_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewmodel = ProcessingViewModel(self)
        self._setup_connections()
        
        # Callbacks for UI updates
        self._on_progress_callback: Optional[Callable] = None
        self._on_finished_callback: Optional[Callable] = None
    
    def _setup_connections(self):
        """Connect ViewModel signals to controller signals."""
        self._viewmodel.processing_started.connect(self.processing_started)
        self._viewmodel.processing_finished.connect(self.processing_finished)
        self._viewmodel.processing_cancelled.connect(self.processing_cancelled)
        self._viewmodel.progress_updated.connect(self.progress_updated)
        self._viewmodel.error_occurred.connect(self.error_occurred)
        self._viewmodel.state_changed.connect(self.state_changed)
    
    def get_state(self) -> str:
        """Get current processing state."""
        return self._viewmodel.get_state()
    
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._viewmodel.is_processing()
    
    def can_start(self) -> bool:
        """Check if processing can be started."""
        return self._viewmodel.can_start()
    
    def can_cancel(self) -> bool:
        """Check if processing can be cancelled."""
        return self._viewmodel.can_cancel()
    
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
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            self.error_occurred.emit(f"Video file not found: {video_path}")
            return False
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting processing: {video_path} -> {output_path}")
        
        return self._viewmodel.start_processing(
            video_path=video_path,
            output_path=output_path,
            config=config,
            backend_type=backend_type,
        )
    
    def cancel_processing(self) -> bool:
        """Request processing cancellation."""
        if not self.can_cancel():
            logger.warning("Cannot cancel - not currently processing")
            return False
        
        logger.info("Cancelling processing")
        return self._viewmodel.cancel_processing()
    
    def get_progress_percentage(self, current: int, total: int) -> int:
        """Calculate progress percentage.
        
        Args:
            current: Current frame
            total: Total frames
            
        Returns:
            Percentage (0-100)
        """
        if total <= 0:
            return 0
        return min(100, int((current / total) * 100))
    
    def format_time_remaining(self, current: int, total: int, fps: float) -> str:
        """Format estimated time remaining.
        
        Args:
            current: Current frame
            total: Total frames
            fps: Current processing FPS
            
        Returns:
            Formatted time string (HH:MM:SS)
        """
        if fps <= 0 or total <= 0:
            return "--:--:--"
        
        remaining_frames = total - current
        remaining_seconds = remaining_frames / fps
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        seconds = int(remaining_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
