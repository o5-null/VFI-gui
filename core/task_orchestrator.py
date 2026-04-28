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
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from core.config.config_facade import ConfigFacade
    from core.backends import BackendConfig, ProcessingConfig, BaseBackend
    from core.io.frame_writer import FrameWriter
    from core.io.frame_data import VideoMetadata


class OrchestratorState(Enum):
    """Orchestrator lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    SHUTTING_DOWN = "shutting_down"


class TaskState(Enum):
    """Individual task states."""
    PENDING = "pending"
    LOADING = "loading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
        video_path: Path to input video
        pipeline_config: Pipeline configuration dict
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
        self._current_task: Optional[TaskContext] = None
        self._task_queue: List[TaskContext] = []
        self._state: OrchestratorState = OrchestratorState.IDLE
        self._worker_thread: Optional[Thread] = None
        self._worker_thread_running = False
        self._lock_guard = None  # For future: threading.Lock()

    # ====================
    # Task Submission
    # ====================

    def submit_task(self, video_path: str, pipeline_config: Dict[str, Any]) -> str:
        """Submit a single task for processing.

        Args:
            video_path: Path to input video
            pipeline_config: Pipeline configuration dict

        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())[:8]
        task = TaskContext(
            task_id=task_id,
            video_path=video_path,
            pipeline_config=pipeline_config,
        )
        self._task_queue.append(task)
        logger.info(f"Task {task_id} submitted: {video_path}")
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
        if self._state in (OrchestratorState.RUNNING, OrchestratorState.PAUSED):
            logger.warning("Orchestrator already running")
            return

        if not self._task_queue:
            logger.warning("No tasks in queue")
            return

        self._state = OrchestratorState.RUNNING
        self._worker_thread_running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Orchestrator started")

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

        Args:
            task: Task to execute
        """
        from core.backends import BackendFactory, BackendType
        from core.io.frame_writer import FrameWriterFactory

        # 1. Resolve configs from pipeline_config
        backend_config = self._resolve_backend_config(task.pipeline_config)
        processing_config = self._resolve_processing_config(task.pipeline_config)

        # 2. Create and initialize backend
        backend = BackendFactory.create(backend_config.backend_type, backend_config)
        if not backend.initialize():
            raise RuntimeError("Backend initialization failed")

        try:
            # 3. Streaming pipeline: process frame -> write frame -> release
            first_yield = True
            writer: Optional[FrameWriter] = None
            output_path: Optional[Path] = None
            start_time = datetime.now()

            for frame_idx, frame_data, metadata in backend.process_frames(
                video_path=task.video_path,
                processing_config=processing_config,
                progress_callback=self._make_progress_callback(task),
                stage_callback=self._make_stage_callback(task),
                log_callback=logger.info,
            ):
                # Check for cancellation
                if self._state in (OrchestratorState.CANCELLING, OrchestratorState.SHUTTING_DOWN):
                    task.state = TaskState.CANCELLED
                    break

                # Wait while paused
                while self._state == OrchestratorState.PAUSED and self._worker_thread_running:
                    import time
                    time.sleep(0.1)

                # Lazy-open writer on first frame (need metadata from first yield)
                if first_yield:
                    output_path = self._resolve_output_path(task, metadata)
                    writer = FrameWriterFactory.create_writer(
                        output_path=output_path,
                        codec_config=task.pipeline_config.get("output", {}),
                    )
                    video_meta = VideoMetadata(
                        width=metadata.get("width", 1920),
                        height=metadata.get("height", 1080),
                        fps=metadata.get("fps", 30.0),
                        total_frames=metadata.get("total_frames", 0),
                    )
                    writer.open(output_path, video_meta)
                    first_yield = False

                # Write frame immediately - no accumulation
                from core.io.frame_data import ProcessedFrameData

                # Assert writer is initialized (guaranteed after first frame)
                assert writer is not None, "Writer should be initialized after first frame"
                writer.write_frame(ProcessedFrameData(
                    data=frame_data,
                    source_frame_idx=frame_idx,
                ))

                # Update progress
                task.progress.current_frame = frame_idx
                task.progress.total_frames = metadata.get("total_frames", 0)
                elapsed = (datetime.now() - start_time).total_seconds()
                task.progress.elapsed_seconds = elapsed
                if elapsed > 0:
                    task.progress.fps = frame_idx / elapsed

                # Emit progress event
                from core.events import task_progress
                task_progress.send(
                    self,
                    task_id=task.task_id,
                    progress=task.progress,
                )

                # frame_data can now be garbage collected

            # Finalize
            if writer:
                writer.close()

            if task.state != TaskState.CANCELLED:
                task.state = TaskState.COMPLETED
                task.output_path = str(output_path) if not first_yield else None

        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            raise

        finally:
            backend.cleanup()

    def _resolve_backend_config(self, pipeline_config: Dict[str, Any]) -> "BackendConfig":
        """Resolve BackendConfig from pipeline configuration.

        Args:
            pipeline_config: Pipeline configuration dict

        Returns:
            BackendConfig instance
        """
        from core.backends import BackendConfig, BackendType

        # Determine backend type from pipeline config
        backend_type_str = pipeline_config.get("backend", "torch")
        try:
            backend_type = BackendType(backend_type_str)
        except ValueError:
            backend_type = BackendType.TORCH

        # Get settings from ConfigFacade sub-configs
        runtime_settings = self._config.runtime.get_all()

        return BackendConfig(
            backend_type=backend_type,
            models_dir=pipeline_config.get("models_dir", "models"),
            temp_dir=pipeline_config.get("temp_dir", "temp"),
            output_dir=pipeline_config.get("output_dir", "output"),
            num_threads=runtime_settings.get("num_threads", 4),
            device=runtime_settings.get("device", "auto"),
            fp16=runtime_settings.get("fp16", True),
            extra=pipeline_config.get("backend_extra", {}),
        )

    def _resolve_processing_config(self, pipeline_config: Dict[str, Any]) -> "ProcessingConfig":
        """Resolve ProcessingConfig from pipeline configuration.

        Args:
            pipeline_config: Pipeline configuration dict

        Returns:
            ProcessingConfig instance
        """
        from core.backends import ProcessingConfig

        return ProcessingConfig(
            interpolation=pipeline_config.get("interpolation", {}),
            upscaling=pipeline_config.get("upscaling", {}),
            scene_detection=pipeline_config.get("scene_detection", {}),
            output=pipeline_config.get("output", {}),
        )

    def _resolve_output_path(self, task: TaskContext, metadata: Dict[str, Any]) -> Path:
        """Resolve output path for the processed video.

        Args:
            task: Task context
            metadata: Video metadata from first processed frame

        Returns:
            Output path
        """
        from core.codec_manager import get_codec_manager, CodecConfig

        # Get output config from pipeline config and global config
        output_config = task.pipeline_config.get("output", {})

        # Get settings from ConfigFacade sub-configs
        # Output settings: codec, quality, etc.
        output_dir = output_config.get("output_dir", "")
        if not output_dir:
            # Get from paths config
            output_dir = self._config.paths.get_output_dir()

        output_subdir = output_config.get("output_subdir", "")
        output_filename = output_config.get("output_filename", "")

        # Generate output filename if not specified
        if not output_filename:
            input_path = Path(task.video_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_path.stem}_processed_{timestamp}"

        # Get extension from codec manager
        codec_manager = get_codec_manager()
        codec_config = CodecConfig.from_dict(output_config)
        codec_manager.set_config(codec_config)

        # Determine extension based on codec
        codec = codec_config.codec or "hevc_nvenc"
        if codec in ("hevc_nvenc", "h265_nvenc"):
            extension = ".mkv"
        elif codec in ("h264_nvenc", "avc_nvenc"):
            extension = ".mp4"
        elif codec in ("vp9", "av1"):
            extension = ".mkv"
        elif codec == "gif":
            extension = ".gif"
        else:
            extension = ".mkv"  # Default

        output_path = Path(output_dir) / f"{output_filename}{extension}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

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
    "TaskState",
    "TaskProgress",
    "TaskContext",
    "TaskOrchestrator",
]