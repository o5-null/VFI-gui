"""Task orchestration for video processing pipeline.

This module provides TaskOrchestrator which controls the entire task workflow:
- Reads settings from ConfigFacade
- Schedules and coordinates tasks
- Manages backend and IO components
- Supports streaming pipeline: process frame -> write frame -> release memory

Architecture:
    MainWindow -> ProcessingController -> TaskOrchestrator -> Backend + IO

Usage:
    from core.task_orchestrator import TaskOrchestrator
    from core.config import get_config

    orchestrator = TaskOrchestrator(get_config())
    task_id = orchestrator.submit_task(video_path, pipeline_config)
    orchestrator.start()
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from loguru import logger

from core.task_parser import TaskParser
from core.types import TaskDescriptor, TaskCheckpoint, TaskState
from core.checkpoint_manager import CheckpointManager

if TYPE_CHECKING:
    from core.config.config_facade import ConfigFacade
    from core.backends.base_backend import BaseBackend
    from core.io.frame_writer import FrameWriter


class OrchestratorState(Enum):
    """Orchestrator lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    SHUTTING_DOWN = "shutting_down"


class TaskProgress:
    """Task progress information.

    Attributes:
        current_frame: Current frame being processed (0-indexed)
        total_frames: Total number of frames to process
        fps: Current processing speed (frames per second)
        stage: Current processing stage name
        elapsed_seconds: Total time elapsed since task start
    """

    def __init__(
        self,
        current_frame: int = 0,
        total_frames: int = 0,
        fps: float = 0.0,
        stage: str = "",
        elapsed_seconds: float = 0.0,
    ):
        self.current_frame = current_frame
        self.total_frames = total_frames
        self.fps = fps
        self.stage = stage
        self.elapsed_seconds = elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "stage": self.stage,
            "elapsed_seconds": self.elapsed_seconds,
        }


class TaskContext:
    """Context for a single processing task.

    Attributes:
        task_id: Unique task identifier
        video_path: Path to input video or image sequence pattern (e.g., /path/to/%04d.png)
        pipeline_config: Pipeline configuration dict
        image_sequence_frames: List of image file paths for image sequences (optional)
        state: Current task state
        progress: Progress information
        output_path: Path to output file (set on completion)
        error: Error message (set on failure)
        created_at: Task creation timestamp
        started_at: Task start timestamp
        finished_at: Task completion timestamp
    """

    def __init__(
        self,
        task_id: str,
        video_path: str,
        pipeline_config: Dict[str, Any],
        image_sequence_frames: Optional[List[str]] = None,
        state: TaskState = TaskState.PENDING,
        progress: Optional[TaskProgress] = None,
        output_path: Optional[str] = None,
        error: Optional[str] = None,
        created_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ):
        self.task_id = task_id
        self.video_path = video_path
        self.pipeline_config = pipeline_config
        self.image_sequence_frames = image_sequence_frames or []
        self.state = state
        self.progress = progress or TaskProgress()
        self.output_path = output_path
        self.error = error
        self.created_at = created_at or datetime.now()
        self.started_at = started_at
        self.finished_at = finished_at


class TaskOrchestrator:
    """Orchestrates video processing tasks.

    Reads settings -> schedules tasks -> coordinates backend + IO.
    Supports streaming pipeline: process frame -> write frame -> release memory.

    NOT a QThread. Runs processing in an internal worker thread.
    Communicates via Blinker events (core/events.py) for Qt-independence.

    Attributes:
        config: ConfigFacade instance for reading settings

    Usage:
        orchestrator = TaskOrchestrator(config)
        task_id = orchestrator.submit_task(video_path, pipeline_config)
        orchestrator.start()
    """

    def __init__(self, config: "ConfigFacade"):
        """Initialize TaskOrchestrator.

        Args:
            config: ConfigFacade instance (via core.get_config())
        """
        self._config = config
        self._task_parser: "TaskParser" = TaskParser(config)
        self._current_task: Optional[TaskContext] = None
        self._task_queue: List[TaskContext] = []
        self._state: OrchestratorState = OrchestratorState.IDLE
        self._worker_thread: Optional[Thread] = None
        self._worker_thread_running = False
        self._lock_guard = None  # For future: threading.Lock()

    # ====================
    # Task Submission
    # ====================

    def submit_task(
        self,
        video_path: str,
        pipeline_config: Dict[str, Any],
        image_sequence_frames: Optional[List[str]] = None,
    ) -> str:
        """Submit a single task for processing.

        Args:
            video_path: Path to input video or image sequence pattern
            pipeline_config: Pipeline configuration dict
            image_sequence_frames: List of image file paths for image sequences (optional)

        Returns:
            task_id: Unique task identifier
        """
        import time
        _debug_start = time.time()
        logger.debug(f"[DEBUG] submit_task() called: {video_path}")
        
        task_id = str(uuid.uuid4())[:8]
        task = TaskContext(
            task_id=task_id,
            video_path=video_path,
            pipeline_config=pipeline_config,
            image_sequence_frames=image_sequence_frames,
        )
        self._task_queue.append(task)
        logger.info(f"Task {task_id} submitted: {video_path}")
        logger.debug(f"[DEBUG] submit_task() complete, elapsed={time.time()-_debug_start:.3f}s")
        return task_id

    def submit_batch(self, items: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """Submit multiple tasks as a batch.

        Args:
            items: List of (video_path, pipeline_config) tuples

        Returns:
            List of task_ids
        """
        task_ids = []
        for video_path, pipeline_config in items:
            task_id = self.submit_task(video_path, pipeline_config)
            task_ids.append(task_id)
        logger.info(f"Batch submitted: {len(task_ids)} tasks")
        return task_ids

    # ====================
    # Lifecycle Control
    # ====================

    def start(self) -> None:
        """Start the orchestrator worker thread.

        If already running, this is a no-op.
        """
        import time
        _debug_start = time.time()
        logger.debug(f"[DEBUG] orchestrator.start() called, state={self._state}")
        
        if self._state in (OrchestratorState.RUNNING, OrchestratorState.PAUSED):
            logger.warning("Orchestrator already running")
            logger.debug(f"[DEBUG] orchestrator.start() early return (already running)")
            return

        if not self._task_queue:
            logger.warning("No tasks in queue")
            logger.debug(f"[DEBUG] orchestrator.start() early return (no tasks)")
            return

        self._state = OrchestratorState.RUNNING
        self._worker_thread_running = True
        logger.debug(f"[DEBUG] creating worker thread, elapsed={time.time()-_debug_start:.3f}s")
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        logger.debug(f"[DEBUG] starting worker thread, elapsed={time.time()-_debug_start:.3f}s")
        self._worker_thread.start()
        logger.info("Orchestrator started")
        logger.debug(f"[DEBUG] orchestrator.start() complete, elapsed={time.time()-_debug_start:.3f}s")

    def pause(self) -> None:
        """Pause the current processing task.

        Note: Pause is best-effort - the backend may need to complete
        the current frame before actually pausing.
        """
        if self._state == OrchestratorState.RUNNING:
            self._state = OrchestratorState.PAUSED
            logger.info("Orchestrator paused")

    def resume(self) -> None:
        """Resume a paused processing task."""
        if self._state == OrchestratorState.PAUSED:
            self._state = OrchestratorState.RUNNING
            logger.info("Orchestrator resumed")

    def cancel_current(self) -> None:
        """Cancel the currently running task."""
        if self._current_task is not None:
            self._state = OrchestratorState.CANCELLING
            logger.info(f"Cancelling task {self._current_task.task_id}")

    def cancel_all(self) -> None:
        """Cancel all pending and running tasks."""
        self._state = OrchestratorState.CANCELLING
        self._task_queue.clear()
        if self._current_task:
            logger.info(f"Cancelling all tasks (current: {self._current_task.task_id})")

    def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        self._state = OrchestratorState.SHUTTING_DOWN
        self._worker_thread_running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        self._state = OrchestratorState.IDLE
        logger.info("Orchestrator shutdown complete")

    # ====================
    # Query Methods
    # ====================

    def get_state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return self._state

    def get_current_task(self) -> Optional[TaskContext]:
        """Get the currently running task."""
        return self._current_task

    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get progress for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            TaskProgress if task exists, None otherwise
        """
        if self._current_task and self._current_task.task_id == task_id:
            return self._current_task.progress

        for task in self._task_queue:
            if task.task_id == task_id:
                return task.progress

        return None

    def has_pending_tasks(self) -> bool:
        """Check if there are pending tasks in the queue."""
        return len(self._task_queue) > 0

    # ====================
    # Internal Methods
    # ====================

    def _worker_loop(self) -> None:
        """Main worker loop - processes tasks from queue."""
        from core.events import (
            task_started,
            task_progress,
            task_finished,
            task_failed,
            task_cancelled,
            orchestrator_state_changed,
        )

        while self._worker_thread_running:
            # Check for shutdown
            if self._state == OrchestratorState.SHUTTING_DOWN:
                break

            # Get next task
            if not self._task_queue:
                break

            task = self._task_queue.pop(0)
            self._current_task = task

            # Emit task started event
            task.state = TaskState.LOADING
            task.started_at = datetime.now()
            task_started.send(self, task_id=task.task_id, video_path=task.video_path)
            orchestrator_state_changed.send(self, state=self._state.value)

            try:
                # Execute the task
                self._run_task(task)

                # Emit completion event
                if task.state == TaskState.COMPLETED:
                    task_finished.send(
                        self,
                        task_id=task.task_id,
                        output_path=task.output_path,
                    )
                elif task.state == TaskState.CANCELLED:
                    task_cancelled.send(self, task_id=task.task_id)

            except Exception as e:
                logger.exception(f"Task {task.task_id} failed: {e}")
                task.state = TaskState.FAILED
                task.error = str(e)
                task_finished.send(self, task_id=task.task_id, error=task.error)

            finally:
                self._current_task = None
                orchestrator_state_changed.send(self, state=self._state.value)

        # Worker loop ended
        self._state = OrchestratorState.IDLE
        orchestrator_state_changed.send(self, state=self._state.value)

    def _run_task(self, task: TaskContext) -> None:
        """Execute a single task with streaming pipeline.

        Uses StreamingFramePairReader + PreprocessPipeline + Backend.infer()
        + OrderedResultBuffer for constant-memory streaming processing.

        Args:
            task: Task to execute
        """
        self._run_task_v2(task)

    def _run_task_v2(self, task: TaskContext) -> None:
        """Execute a task using the new streaming pipeline.

        Uses StreamingFramePairReader + PreprocessPipeline + Backend.infer()
        + OrderedResultBuffer for constant-memory streaming processing.

        Args:
            task: Task to execute
        """
        from core.io.streaming_reader import StreamingFramePairReader
        from core.io.frame_writer import FrameWriterFactory
        from core.backends.base_backend import BackendFactory
        from core.io.frame_data import VideoMetadata as LegacyVideoMetadata

        import time as _time
        import torch

        _debug_start = _time.time()
        logger.debug(f"[DEBUG] _run_task_v2() started for task {task.task_id}")

        # 1. Create TaskDescriptor and parse to TaskDefinition
        descriptor = TaskDescriptor(
            video_path=task.video_path,
            pipeline_config=task.pipeline_config,
            image_sequence_frames=task.image_sequence_frames or None,
        )
        task_def = self._task_parser.parse(descriptor)
        backend_config = task_def.backend_config
        processing_config = task_def.processing_config

        # 2. Create backend
        backend = BackendFactory.create(backend_config.backend_type, backend_config)
        if not backend.initialize():
            raise RuntimeError("Backend initialization failed")

        # Load model if backend supports it
        _backend_any: Any = backend
        if hasattr(_backend_any, "load_model"):
            _backend_any.load_model(processing_config.interpolation)

        # Checkpoint resume support
        checkpoint_manager = CheckpointManager(backend_config.temp_dir)
        checkpoint = checkpoint_manager.load(task.task_id)
        if checkpoint:
            logger.info(
                f"Found checkpoint for task {task.task_id}: "
                f"frame {checkpoint.last_completed_frame}/{checkpoint.total_frames}"
            )

        try:
            # 3. Open streaming reader
            reader = StreamingFramePairReader(
                task.video_path,
                device=backend_config.get_device(),
            )

            # 4. Create writer (use output_path from TaskDefinition)
            output_path = task_def.output_path
            writer = FrameWriterFactory.create_writer(
                output_path=output_path,
                codec_config=task.pipeline_config.get("output", {}),
            )
            # Convert to legacy metadata for FrameWriter compat
            meta = reader.metadata
            legacy_meta = LegacyVideoMetadata(
                width=meta.width,
                height=meta.height,
                fps=meta.fps,
                total_frames=meta.total_frames,
            )
            writer.open(output_path, legacy_meta)

            # Update checkpoint's output_path if checkpoint exists
            if checkpoint:
                checkpoint.output_path = str(output_path)

            # 5. Create preprocessing pipeline
            from core.preprocess.pipeline import PreprocessPipeline
            pipeline = PreprocessPipeline(processing_config, backend_config.backend_type)

            # 6. Run streaming loop
            from core.task_scheduler import ParallelStreamingLoop

            loop = ParallelStreamingLoop()
            multiplier = processing_config.interpolation.get("multi", 2)
            start_time = datetime.now()

            def progress_callback(current: int, total: int, fps: float) -> None:
                task.progress.current_frame = current
                task.progress.total_frames = total
                task.progress.fps = fps
                elapsed = (datetime.now() - start_time).total_seconds()
                task.progress.elapsed_seconds = elapsed

                from core.events import task_progress
                task_progress.send(
                    self,
                    task_id=task.task_id,
                    progress=task.progress,
                )

                # Check cancellation
                if self._state in (OrchestratorState.CANCELLING, OrchestratorState.SHUTTING_DOWN):
                    loop.cancel()

            loop.run(
                reader=reader,
                pipeline=pipeline,
                backend=backend,
                writer=writer,
                multiplier=multiplier,
                progress_callback=progress_callback,
                checkpoint=checkpoint,
                checkpoint_manager=checkpoint_manager,
                task_id=task.task_id,
            )

            # 7. Finalize
            writer.close()

            if self._state in (OrchestratorState.CANCELLING, OrchestratorState.SHUTTING_DOWN):
                task.state = TaskState.CANCELLED
            else:
                task.state = TaskState.COMPLETED
                task.output_path = str(output_path)

            logger.debug(
                f"[DEBUG] _run_task_v2() complete, elapsed="
                f"{_time.time()-_debug_start:.3f}s"
            )

        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            raise

        finally:
            backend.cleanup()

    def _make_progress_callback(self, task: TaskContext) -> Callable[[int, int, float], None]:
        """Create progress callback for backend.

        Args:
            task: Task context

        Returns:
            Callback function
        """
        def callback(current: int, total: int, fps: float) -> None:
            task.progress.current_frame = current
            task.progress.total_frames = total
            task.progress.fps = fps

        return callback

    def _make_stage_callback(self, task: TaskContext) -> Callable[[str], None]:
        """Create stage callback for backend.

        Args:
            task: Task context

        Returns:
            Callback function
        """
        def callback(stage: str) -> None:
            task.progress.stage = stage

        return callback


__all__ = [
    "OrchestratorState",
    "TaskProgress",
    "TaskContext",
    "TaskOrchestrator",
]