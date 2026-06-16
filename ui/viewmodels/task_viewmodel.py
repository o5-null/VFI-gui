"""TaskViewModel for task progress and state tracking.

Tracks processing state, progress, and GPU metrics for UI display.
GPU signals are placeholders - no data source, signals won't fire, UI shows "N/A".
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


class TaskViewModel(QObject):
    """ViewModel for task progress and state tracking.
    
    Signals:
        state_changed: Task state (idle, loading, processing, completed, failed, cancelled)
        video_path_changed: Input video path
        video_name_changed: Video filename for display
        progress_changed: Progress percentage (0.0 to 1.0)
        current_frame_changed: Current frame index
        total_frames_changed: Total frames count
        fps_changed: Current processing speed
        eta_changed: Estimated time remaining (formatted string)
        scene_cuts_changed: Number of detected scene cuts
        skipped_frames_changed: Number of skipped frames
        vram_used_changed: VRAM used in GB (placeholder, may not fire)
        vram_total_changed: Total VRAM in GB (placeholder, may not fire)
        gpu_util_changed: GPU utilization percentage (placeholder, may not fire)
        gpu_temp_changed: GPU temperature in Celsius (placeholder, may not fire)
        log_entry_added: Log entry (level, message)
        error_occurred: Error message
    
    Properties:
        All fields as properties with getters
    
    Setters:
        All fields have setters that emit corresponding signals
    
    Note:
        GPU signals are placeholders - no data source connected.
        UI components should handle missing data gracefully (show "N/A").
    """
    
    # State signals
    state_changed = pyqtSignal(str)
    video_path_changed = pyqtSignal(str)
    video_name_changed = pyqtSignal(str)
    
    # Progress signals
    progress_changed = pyqtSignal(float)
    current_frame_changed = pyqtSignal(int)
    total_frames_changed = pyqtSignal(int)
    fps_changed = pyqtSignal(float)
    eta_changed = pyqtSignal(str)
    
    # Statistics signals
    scene_cuts_changed = pyqtSignal(int)
    skipped_frames_changed = pyqtSignal(int)
    
    # GPU signals (placeholders - no data source)
    vram_used_changed = pyqtSignal(float)
    vram_total_changed = pyqtSignal(float)
    gpu_util_changed = pyqtSignal(float)
    gpu_temp_changed = pyqtSignal(int)
    
    # Log signals
    log_entry_added = pyqtSignal(str, str)  # level, message
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize TaskViewModel with default state."""
        super().__init__(parent)
        
        # Private fields with defaults
        self._state: str = "idle"
        self._video_path: str = ""
        self._video_name: str = ""
        self._progress: float = 0.0
        self._current_frame: int = 0
        self._total_frames: int = 0
        self._fps: float = 0.0
        self._eta: str = ""
        self._scene_cuts: int = 0
        self._skipped_frames: int = 0
        
        # GPU fields (placeholders)
        self._vram_used: float = 0.0
        self._vram_total: float = 0.0
        self._gpu_util: float = 0.0
        self._gpu_temp: int = 0
    
    # ====================
    # Properties - State
    # ====================
    
    @property
    def state(self) -> str:
        """Get current task state."""
        return self._state
    
    @property
    def video_path(self) -> str:
        """Get input video path."""
        return self._video_path
    
    @property
    def video_name(self) -> str:
        """Get video filename for display."""
        return self._video_name
    
    # ====================
    # Properties - Progress
    # ====================
    
    @property
    def progress(self) -> float:
        """Get progress percentage (0.0 to 1.0)."""
        return self._progress
    
    @property
    def current_frame(self) -> int:
        """Get current frame index."""
        return self._current_frame
    
    @property
    def total_frames(self) -> int:
        """Get total frames count."""
        return self._total_frames
    
    @property
    def fps(self) -> float:
        """Get current processing speed."""
        return self._fps
    
    @property
    def eta(self) -> str:
        """Get estimated time remaining."""
        return self._eta
    
    # ====================
    # Properties - Statistics
    # ====================
    
    @property
    def scene_cuts(self) -> int:
        """Get number of detected scene cuts."""
        return self._scene_cuts
    
    @property
    def skipped_frames(self) -> int:
        """Get number of skipped frames."""
        return self._skipped_frames
    
    # ====================
    # Properties - GPU (placeholders)
    # ====================
    
    @property
    def vram_used(self) -> float:
        """Get VRAM used in GB."""
        return self._vram_used
    
    @property
    def vram_total(self) -> float:
        """Get total VRAM in GB."""
        return self._vram_total
    
    @property
    def gpu_util(self) -> float:
        """Get GPU utilization percentage."""
        return self._gpu_util
    
    @property
    def gpu_temp(self) -> int:
        """Get GPU temperature in Celsius."""
        return self._gpu_temp
    
    # ====================
    # Setters (emit signals)
    # ====================
    
    def set_state(self, state: str) -> None:
        """Set task state."""
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)
    
    def set_video_path(self, path: str) -> None:
        """Set video path and update name."""
        if path != self._video_path:
            self._video_path = path
            self._video_name = Path(path).name if path else ""
            self.video_path_changed.emit(path)
            self.video_name_changed.emit(self._video_name)
    
    def set_video_name(self, name: str) -> None:
        """Set video name directly."""
        if name != self._video_name:
            self._video_name = name
            self.video_name_changed.emit(name)
    
    def set_progress(self, progress: float) -> None:
        """Set progress percentage."""
        if progress != self._progress:
            self._progress = progress
            self.progress_changed.emit(progress)
    
    def set_current_frame(self, frame: int) -> None:
        """Set current frame index."""
        if frame != self._current_frame:
            self._current_frame = frame
            self.current_frame_changed.emit(frame)
    
    def set_total_frames(self, total: int) -> None:
        """Set total frames count."""
        if total != self._total_frames:
            self._total_frames = total
            self.total_frames_changed.emit(total)
    
    def set_fps(self, fps: float) -> None:
        """Set processing speed."""
        if fps != self._fps:
            self._fps = fps
            self.fps_changed.emit(fps)
    
    def set_eta(self, eta: str) -> None:
        """Set estimated time remaining."""
        if eta != self._eta:
            self._eta = eta
            self.eta_changed.emit(eta)
    
    def set_scene_cuts(self, cuts: int) -> None:
        """Set scene cuts count."""
        if cuts != self._scene_cuts:
            self._scene_cuts = cuts
            self.scene_cuts_changed.emit(cuts)
    
    def set_skipped_frames(self, skipped: int) -> None:
        """Set skipped frames count."""
        if skipped != self._skipped_frames:
            self._skipped_frames = skipped
            self.skipped_frames_changed.emit(skipped)
    
    def set_vram_used(self, vram: float) -> None:
        """Set VRAM used (placeholder)."""
        if vram != self._vram_used:
            self._vram_used = vram
            self.vram_used_changed.emit(vram)
    
    def set_vram_total(self, vram: float) -> None:
        """Set total VRAM (placeholder)."""
        if vram != self._vram_total:
            self._vram_total = vram
            self.vram_total_changed.emit(vram)
    
    def set_gpu_util(self, util: float) -> None:
        """Set GPU utilization (placeholder)."""
        if util != self._gpu_util:
            self._gpu_util = util
            self.gpu_util_changed.emit(util)
    
    def set_gpu_temp(self, temp: int) -> None:
        """Set GPU temperature (placeholder)."""
        if temp != self._gpu_temp:
            self._gpu_temp = temp
            self.gpu_temp_changed.emit(temp)
    
    # ====================
    # Utility Methods
    # ====================
    
    def add_log(self, level: str, message: str) -> None:
        """Add a log entry.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
        """
        self.log_entry_added.emit(level, message)
    
    def report_error(self, error: str) -> None:
        """Report an error.
        
        Args:
            error: Error message
        """
        self._state = "failed"
        self.state_changed.emit(self._state)
        self.error_occurred.emit(error)
    
    def reset(self) -> None:
        """Reset all state to defaults."""
        self._state = "idle"
        self._video_path = ""
        self._video_name = ""
        self._progress = 0.0
        self._current_frame = 0
        self._total_frames = 0
        self._fps = 0.0
        self._eta = ""
        self._scene_cuts = 0
        self._skipped_frames = 0
        
        # Emit all signals to reset UI
        self.state_changed.emit(self._state)
        self.video_path_changed.emit(self._video_path)
        self.video_name_changed.emit(self._video_name)
        self.progress_changed.emit(self._progress)
        self.current_frame_changed.emit(self._current_frame)
        self.total_frames_changed.emit(self._total_frames)
        self.fps_changed.emit(self._fps)
        self.eta_changed.emit(self._eta)
        self.scene_cuts_changed.emit(self._scene_cuts)
        self.skipped_frames_changed.emit(self._skipped_frames)
    
    def update_progress(
        self,
        current_frame: int,
        total_frames: int,
        fps: float,
        elapsed_seconds: float,
    ) -> None:
        """Update progress information in batch.
        
        Args:
            current_frame: Current frame index
            total_frames: Total frames
            fps: Processing speed
            elapsed_seconds: Time elapsed
        """
        self.set_current_frame(current_frame)
        self.set_total_frames(total_frames)
        self.set_fps(fps)
        
        # Calculate progress
        if total_frames > 0:
            progress = current_frame / total_frames
            self.set_progress(progress)
        
        # Calculate ETA
        if fps > 0 and total_frames > current_frame:
            remaining_frames = total_frames - current_frame
            remaining_seconds = remaining_frames / fps
            eta_str = self._format_eta(remaining_seconds)
            self.set_eta(eta_str)
        else:
            self.set_eta("")
    
    def _format_eta(self, seconds: float) -> str:
        """Format ETA seconds to human-readable string.
        
        Args:
            seconds: Remaining seconds
            
        Returns:
            Formatted ETA string (e.g., "5m 30s")
        """
        if seconds < 0:
            return ""
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


__all__ = ["TaskViewModel"]