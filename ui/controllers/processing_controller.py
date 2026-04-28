"""Processing Controller for VFI-gui.

Bridge between Qt UI and TaskOrchestrator. Handles UI-agnostic processing
operations while connecting to Qt signals for UI updates.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger

from core.task_orchestrator import TaskOrchestrator, TaskProgress
from core.workers import ProcessingWorker


class ProcessingController(QObject):
    """Bridge between Qt UI and TaskOrchestrator.

    Connects Qt signals from MainWindow to TaskOrchestrator's Blinker events.
    Handles UI-specific logic (confirm dialogs, status messages).
    """

    # Qt signals for UI
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(bool, str)  # success, message
    processing_cancelled = pyqtSignal()
    progress_updated = pyqtSignal(int, int, float)  # frame, total, fps
    error_occurred = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(self, orchestrator: TaskOrchestrator, parent: Optional[QObject] = None):
        """Initialize controller with TaskOrchestrator.

        Args:
            orchestrator: TaskOrchestrator instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._connect_events()

        # Callbacks for UI updates
        self._on_progress_callback = None
        self._on_finished_callback = None

    def _connect_events(self):
        """Bridge Blinker events -> Qt signals."""
        from core.events import (
            task_started,
            task_progress,
            task_finished,
            task_failed,
            task_cancelled,
            orchestrator_state_changed,
        )

        task_started.connect(self._on_task_started)
        task_progress.connect(self._on_task_progress)
        task_finished.connect(self._on_task_finished)
        task_failed.connect(self._on_task_failed)
        task_cancelled.connect(self._on_task_cancelled)
        orchestrator_state_changed.connect(self._on_orchestrator_state_changed)

    # ====================
    # Signal handlers (Blinker -> Qt)
    # ====================

    def _on_task_started(self, sender, **kwargs):
        """Handle task started event."""
        video_path = kwargs.get("video_path", "")
        logger.info(f"Processing started: {video_path}")
        self.processing_started.emit()

    def _on_task_progress(self, sender, **kwargs):
        """Handle task progress event."""
        progress = kwargs.get("progress")
        if progress is not None and hasattr(progress, "current_frame"):
            self.progress_updated.emit(
                progress.current_frame,
                progress.total_frames,
                progress.fps,
            )

    def _on_task_finished(self, sender, **kwargs):
        """Handle task finished event."""
        task_id = kwargs.get("task_id", "unknown")
        output_path = kwargs.get("output_path", "")
        error = kwargs.get("error")

        if error:
            logger.error(f"Task {task_id} failed: {error}")
            self.error_occurred.emit(error)
            self.processing_finished.emit(False, error)
        else:
            logger.info(f"Task {task_id} completed: {output_path}")
            self.processing_finished.emit(True, output_path)

    def _on_task_failed(self, sender, **kwargs):
        """Handle task failed event."""
        task_id = kwargs.get("task_id", "unknown")
        error = kwargs.get("error", "Unknown error")
        logger.error(f"Task {task_id} failed: {error}")
        self.error_occurred.emit(error)

    def _on_task_cancelled(self, sender, **kwargs):
        """Handle task cancelled event."""
        task_id = kwargs.get("task_id", "unknown")
        logger.info(f"Task {task_id} cancelled")
        self.processing_cancelled.emit()

    def _on_orchestrator_state_changed(self, sender, **kwargs):
        """Handle orchestrator state change."""
        state = kwargs.get("state", "idle")
        self.state_changed.emit(state)

    # ====================
    # Public API (delegate to orchestrator)
    # ====================

    def start_processing(self, video_path: str, pipeline_config: Dict[str, Any]) -> str:
        """Start video processing via TaskOrchestrator.

        Args:
            video_path: Path to input video
            pipeline_config: Pipeline configuration dict

        Returns:
            task_id: Task identifier
        """
        from pathlib import Path

        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            self.error_occurred.emit(f"Video file not found: {video_path}")
            return ""

        logger.info(f"Starting processing: {video_path}")

        # Submit task to orchestrator
        task_id = self._orchestrator.submit_task(video_path, pipeline_config)

        # Start orchestrator if not already running
        if not self._orchestrator.has_pending_tasks():
            self._orchestrator.start()

        return task_id

    def cancel_processing(self) -> bool:
        """Request processing cancellation."""
        logger.info("Cancelling processing")
        self._orchestrator.cancel_current()
        return True

    def pause_processing(self) -> None:
        """Pause current processing."""
        self._orchestrator.pause()

    def resume_processing(self) -> None:
        """Resume paused processing."""
        self._orchestrator.resume()

    def get_state(self) -> str:
        """Get current orchestrator state."""
        return self._orchestrator.get_state().value

    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._orchestrator.get_state().value in ("running", "paused")

    def can_start(self) -> bool:
        """Check if processing can be started."""
        return not self.is_processing()

    def can_cancel(self) -> bool:
        """Check if processing can be cancelled."""
        return self.is_processing()

    # Legacy compatibility - these methods existed in the old controller
    def start_processing_legacy(
        self,
        video_path: str,
        output_path: str,
        config: Any,
        backend_type: Any,
    ) -> bool:
        """Legacy method for backward compatibility.

        Deprecated: Use start_processing() instead.
        """
        import warnings

        warnings.warn(
            "start_processing_legacy() is deprecated. Use start_processing() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Build pipeline config from legacy parameters
        pipeline_config = {
            "backend": backend_type.value if hasattr(backend_type, "value") else str(backend_type),
            "interpolation": config.interpolation if hasattr(config, "interpolation") else {},
            "upscaling": config.upscaling if hasattr(config, "upscaling") else {},
            "scene_detection": config.scene_detection if hasattr(config, "scene_detection") else {},
            "output": {
                "output_dir": str(Path(output_path).parent),
                "output_filename": Path(output_path).stem,
            },
        }

        return bool(self.start_processing(video_path, pipeline_config))

    def cancel_legacy(self) -> bool:
        """Legacy method for backward compatibility.

        Deprecated: Use cancel_processing() instead.
        """
        import warnings

        warnings.warn(
            "cancel_legacy() is deprecated. Use cancel_processing() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.cancel_processing()


__all__ = [
    "ProcessingController",
]