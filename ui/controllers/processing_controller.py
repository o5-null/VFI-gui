"""ProcessingController - bridges Blinker signals to TaskViewModel.

Stateless controller that:
- Subscribes to core.events Blinker signals
- Translates TaskProgress to TaskViewModel signal emissions
- Provides action methods for UI commands

Controllers do NOT hold state - they only bridge signals.
"""

from typing import Any, Dict, TYPE_CHECKING

from loguru import logger

from core.events import (
    task_started,
    task_progress,
    task_finished,
    task_failed,
    task_cancelled,
    orchestrator_state_changed,
)
from core.task_orchestrator import TaskOrchestrator, OrchestratorState, TaskProgress

if TYPE_CHECKING:
    from ui.viewmodels.task_viewmodel import TaskViewModel
    from ui.viewmodels.pipeline_viewmodel import PipelineViewModel


class ProcessingController:
    """Stateless controller bridging Blinker events to TaskViewModel.
    
    This controller:
    - Connects core.events Blinker signals to TaskViewModel Qt signals
    - Translates TaskProgress data to TaskViewModel state updates
    - Provides action methods for UI commands
    
    Controllers do NOT hold state - they are pure signal bridges.
    
    Methods:
        start_task(video_path, pipeline_config): Start processing
        pause_task(): Pause current task
        cancel_task(): Cancel current task
        resume_task(): Resume paused task
    """
    
    def __init__(
        self,
        orchestrator: TaskOrchestrator,
        task_vm: "TaskViewModel",
        pipeline_vm: "PipelineViewModel",
    ):
        """Initialize ProcessingController.
        
        Args:
            orchestrator: TaskOrchestrator instance
            task_vm: TaskViewModel instance
            pipeline_vm: PipelineViewModel instance
        """
        self._orchestrator = orchestrator
        self._task_vm = task_vm
        self._pipeline_vm = pipeline_vm
        
        # Subscribe to Blinker signals
        self._subscribe_blinker()
    
    def _subscribe_blinker(self) -> None:
        """Subscribe to core.events Blinker signals."""
        task_started.connect(self._on_task_started)
        task_progress.connect(self._on_task_progress)
        task_finished.connect(self._on_task_finished)
        task_failed.connect(self._on_task_failed)
        task_cancelled.connect(self._on_task_cancelled)
        orchestrator_state_changed.connect(self._on_orchestrator_state)
    
    # ====================
    # Blinker Signal Handlers
    # ====================
    
    def _on_task_started(self, sender, **kwargs) -> None:
        """Handle task_started signal.
        
        Args:
            sender: Signal sender (TaskOrchestrator)
            kwargs: task_id, video_path
        """
        task_id = kwargs.get("task_id", "")
        video_path = kwargs.get("video_path", "")
        
        logger.debug(f"Task started: {task_id}")
        
        self._task_vm.set_state("loading")
        self._task_vm.set_video_path(video_path)
        self._task_vm.set_progress(0.0)
        self._task_vm.add_log("info", f"Started processing: {video_path}")
    
    def _on_task_progress(self, sender, **kwargs) -> None:
        """Handle task_progress signal.
        
        Args:
            sender: Signal sender (TaskOrchestrator)
            kwargs: task_id, progress (TaskProgress)
        """
        task_id = kwargs.get("task_id", "")
        progress = kwargs.get("progress")
        
        if progress is None:
            return
        
        # TaskProgress has: current_frame, total_frames, fps, stage, elapsed_seconds
        self._task_vm.update_progress(
            current_frame=progress.current_frame,
            total_frames=progress.total_frames,
            fps=progress.fps,
            elapsed_seconds=progress.elapsed_seconds,
        )
        
        # Update state based on stage
        if progress.stage:
            self._task_vm.set_state("processing")
        
        # Log progress periodically
        if progress.current_frame % 100 == 0:
            self._task_vm.add_log("debug", f"Frame {progress.current_frame}/{progress.total_frames}")
    
    def _on_task_finished(self, sender, **kwargs) -> None:
        """Handle task_finished signal.
        
        Args:
            sender: Signal sender (TaskOrchestrator)
            kwargs: task_id, output_path, error
        """
        task_id = kwargs.get("task_id", "")
        output_path = kwargs.get("output_path", "")
        error = kwargs.get("error")
        
        if error:
            # Task failed with error
            self._task_vm.set_state("failed")
            self._task_vm.report_error(error)
            self._task_vm.add_log("error", f"Task failed: {error}")
        else:
            # Task completed successfully
            self._task_vm.set_state("completed")
            self._task_vm.set_progress(1.0)
            self._task_vm.add_log("info", f"Task completed: {output_path}")
        
        logger.debug(f"Task finished: {task_id}")
    
    def _on_task_failed(self, sender, **kwargs) -> None:
        """Handle task_failed signal.
        
        Args:
            sender: Signal sender (TaskOrchestrator)
            kwargs: task_id, error
        """
        task_id = kwargs.get("task_id", "")
        error = kwargs.get("error", "Unknown error")
        
        self._task_vm.set_state("failed")
        self._task_vm.report_error(error)
        self._task_vm.add_log("error", f"Task failed: {error}")
        
        logger.error(f"Task failed: {task_id} - {error}")
    
    def _on_task_cancelled(self, sender, **kwargs) -> None:
        """Handle task_cancelled signal.
        
        Args:
            sender: Signal sender (TaskOrchestrator)
            kwargs: task_id
        """
        task_id = kwargs.get("task_id", "")
        
        self._task_vm.set_state("cancelled")
        self._task_vm.add_log("warning", f"Task cancelled: {task_id}")
        
        logger.debug(f"Task cancelled: {task_id}")
    
    def _on_orchestrator_state(self, sender, **kwargs) -> None:
        """Handle orchestrator_state_changed signal.
        
        Args:
            sender: Signal sender (TaskOrchestrator)
            kwargs: state (str)
        """
        state = kwargs.get("state", "idle")
        
        # Map orchestrator state to task_vm state
        state_map = {
            "idle": "idle",
            "running": "processing",
            "paused": "paused",
            "cancelling": "cancelling",
            "shutting_down": "idle",
        }
        
        task_state = state_map.get(state, "idle")
        
        # Only update if not already in terminal state
        current_state = self._task_vm.state
        terminal_states = {"completed", "failed", "cancelled"}
        
        if current_state not in terminal_states:
            self._task_vm.set_state(task_state)
        
        logger.debug(f"Orchestrator state changed: {state}")
    
    # ====================
    # Action Methods
    # ====================
    
    def start_task(self, video_path: str, pipeline_config: Dict[str, Any]) -> str:
        """Start processing a video.
        
        Args:
            video_path: Path to input video
            pipeline_config: Pipeline configuration dict
            
        Returns:
            Task ID
        """
        logger.info(f"Starting task: {video_path}")
        
        # Reset task state
        self._task_vm.reset()
        self._task_vm.set_video_path(video_path)
        self._task_vm.set_state("loading")
        
        # Submit task to orchestrator
        task_id = self._orchestrator.submit_task(video_path, pipeline_config)
        
        # Start processing
        self._orchestrator.start()
        
        return task_id
    
    def pause_task(self) -> None:
        """Pause current processing."""
        logger.info("Pausing task")
        self._orchestrator.pause()
        self._task_vm.set_state("paused")
        self._task_vm.add_log("warning", "Processing paused")
    
    def cancel_task(self) -> None:
        """Cancel current processing."""
        logger.info("Cancelling task")
        self._orchestrator.cancel_current()
        self._task_vm.set_state("cancelling")
        self._task_vm.add_log("warning", "Cancelling processing...")
    
    def resume_task(self) -> None:
        """Resume paused processing."""
        logger.info("Resuming task")
        self._orchestrator.resume()
        self._task_vm.set_state("processing")
        self._task_vm.add_log("info", "Processing resumed")
    
    def cancel_all(self) -> None:
        """Cancel all pending and running tasks."""
        logger.info("Cancelling all tasks")
        self._orchestrator.cancel_all()
        self._task_vm.add_log("warning", "Cancelling all tasks...")
    
    def shutdown(self) -> None:
        """Shutdown orchestrator."""
        logger.info("Shutting down orchestrator")
        self._orchestrator.shutdown()
    
    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        state = self._orchestrator.get_state()
        return state == OrchestratorState.RUNNING
    
    def is_paused(self) -> bool:
        """Check if orchestrator is paused."""
        state = self._orchestrator.get_state()
        return state == OrchestratorState.PAUSED
    
    def get_current_task(self) -> Any:
        """Get current task context."""
        return self._orchestrator.get_current_task()


__all__ = ["ProcessingController"]