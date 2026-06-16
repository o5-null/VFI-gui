"""Parallel streaming interpolation loop and task scheduler.

This module provides the core scheduling logic for VFI-gui's streaming
processing pipeline:

    StreamingFramePairReader → frame pairs (constant memory)
    → PreprocessPipeline → INTERPOLATE / SCENE_CUT / DUPLICATE / LAST_FRAME
    → ParallelStreamingLoop → parallel inference + ordered write
    → OrderedResultBuffer → frames written in sequence order
    → FrameLifecycle → write-once, release when consumers complete

Replaces the single-threaded sequential inference in TaskOrchestrator
with a streaming architecture that maintains constant memory usage.

Architecture constraint:
    - TaskScheduler does NOT directly read/write files (uses FrameWriter)
    - Backend does NOT write files (returns InferenceResult)
    - Progress communicated via Blinker events (no UI references)
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from loguru import logger

from core.types import (
    BackendConfig,
    FramePairAction,
    InferenceRequest,
    InferenceResult,
    ParallelConfig,
    ProcessingConfig,
    ProcessedFrameData,
    TaskCheckpoint,
    TaskDefinition,
)
from core.checkpoint_manager import CheckpointManager
from core.io.streaming_reader import StreamingFramePairReader
from core.io.ordered_buffer import OrderedResultBuffer
from core.io.frame_lifecycle import FrameLifecycle
from core.io.frame_writer import FrameWriter, FrameWriterFactory
from core.preprocess.pipeline import PreprocessPipeline
from core.backends.base_backend import BaseBackend


# ====================
# Scheduler State
# ====================


class SchedulerState(Enum):
    """Task scheduler lifecycle states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    SHUTTING_DOWN = "shutting_down"


# ====================
# Parallel Streaming Loop
# ====================


class ParallelStreamingLoop:
    """Parallel streaming interpolation loop.

    Flow:
    1. StreamingFramePairReader → read frame pairs one at a time (constant memory)
    2. PreprocessPipeline → preprocessing decision per pair
    3. INTERPOLATE → submit inference to backend
    4. SCENE_CUT / DUPLICATE → skip Backend, write original frame directly
    5. OrderedResultBuffer → parallel results written in sequence order
    6. FrameLifecycle → each original frame written exactly once, released
       when all consumers complete

    Memory usage is constant: ≈ 3 frames (frame0 + frame1 + interp_frame).
    """

    CHECKPOINT_INTERVAL = 50  # Progress update interval
    CACHE_CLEANUP_INTERVAL = 10  # torch.cuda.empty_cache() interval

    def __init__(
        self,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> None:
        """Initialize the streaming loop.

        Args:
            parallel_config: Parallel processing configuration.
                If None, uses defaults (2 workers, 4 prefetch).
        """
        self._parallel_config = parallel_config or ParallelConfig()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the current loop."""
        self._cancelled = True

    def run(
        self,
        reader: StreamingFramePairReader,
        pipeline: PreprocessPipeline,
        backend: BaseBackend,
        writer: FrameWriter,
        multiplier: int,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        checkpoint: Optional[TaskCheckpoint] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        task_id: str = "",
    ) -> None:
        """Execute the streaming interpolation loop.

        Memory usage is constant: ≈ 3 frames (frame0 + frame1 + interp_frame).

        Args:
            reader: Streaming frame pair reader (source of frame pairs).
            pipeline: Preprocessing pipeline for frame pair decisions.
            backend: Inference backend (must implement infer()).
            writer: Frame writer for output.
            multiplier: Frame multiplier (e.g., 2 for 2x interpolation).
            progress_callback: Optional callback(current_frame, total_frames, fps).
            checkpoint: Optional checkpoint for resume support.
            checkpoint_manager: Optional checkpoint manager for saving progress.
            task_id: Task identifier for checkpoint tracking.
        """
        buffer = OrderedResultBuffer(writer)
        lifecycle = FrameLifecycle()
        self._cancelled = False

        output_frame_idx = 0  # Global output frame index
        start_time = time.perf_counter()

        # Calculate start_frame from checkpoint for resume support
        start_frame = 0
        if checkpoint and checkpoint_manager:
            if checkpoint_manager.validate_checkpoint(checkpoint):
                start_frame = checkpoint.last_completed_frame + 1
                logger.info(
                    f"Resuming from checkpoint: task_id={checkpoint.task_id}, "
                    f"frame={start_frame}/{checkpoint.total_frames}"
                )
            else:
                logger.warning(
                    f"Checkpoint validation failed for task_id={checkpoint.task_id}, "
                    f"starting from beginning"
                )

        for pair in reader:
            # Check cancellation
            if self._cancelled:
                break

            # Skip frames already processed (checkpoint resume)
            if pair.index < start_frame:
                continue

            # Convert tensors to numpy for preprocessing (scene/dup detection)
            frame0_np = pair.frame0.cpu().permute(1, 2, 0).numpy()
            frame1_np = (
                pair.frame1.cpu().permute(1, 2, 0).numpy()
                if pair.frame1 is not None
                else None
            )

            # Preprocessing decision
            decision = pipeline.decide(frame0_np, frame1_np, pair.index)

            if decision.action == FramePairAction.INTERPOLATE:
                # Write original frame (once only)
                if lifecycle.can_write(pair.index):
                    buffer.submit(
                        output_frame_idx,
                        ProcessedFrameData(
                            data=pair.frame0.permute(1, 2, 0).cpu(),
                            source_frame_idx=pair.index,
                        ),
                    )
                    lifecycle.mark_written(pair.index)
                    output_frame_idx += 1

                # Interpolate at each timestep
                for j in range(1, multiplier):
                    subtask_id = f"st_{pair.index}_{j}"
                    timestep = j / multiplier

                    # Register consumers for both frames
                    lifecycle.register(pair.index, subtask_id)
                    lifecycle.register(pair.index + 1, subtask_id)

                    # Submit inference
                    result = backend.infer(
                        InferenceRequest(
                            frame0=pair.frame0,
                            frame1=pair.frame1,
                            timestep=timestep,
                            model_config={},
                        )
                    )

                    if result.success:
                        buffer.submit(
                            output_frame_idx,
                            ProcessedFrameData(
                                data=result.output_frame.permute(1, 2, 0).cpu(),
                                source_frame_idx=pair.index,
                                interpolated=True,
                                interpolation_ratio=timestep,
                            ),
                        )
                        output_frame_idx += 1

                        # Check if frames can be released
                        lifecycle.can_release(pair.index, subtask_id)
                        lifecycle.can_release(pair.index + 1, subtask_id)

                        del result  # Release GPU tensor immediately
                    else:
                        logger.warning(
                            f"Inference failed for pair {pair.index} "
                            f"timestep {timestep:.2f}: {result.error}"
                        )

            elif decision.action in (
                FramePairAction.SCENE_CUT,
                FramePairAction.DUPLICATE,
            ):
                # Scene cut / duplicate: write original frame directly, no Backend
                if lifecycle.can_write(pair.index):
                    buffer.submit(
                        output_frame_idx,
                        ProcessedFrameData(
                            data=pair.frame0.permute(1, 2, 0).cpu(),
                            source_frame_idx=pair.index,
                        ),
                    )
                    lifecycle.mark_written(pair.index)
                    output_frame_idx += 1

            elif decision.action == FramePairAction.LAST_FRAME:
                buffer.submit(
                    output_frame_idx,
                    ProcessedFrameData(
                        data=pair.frame0.permute(1, 2, 0).cpu(),
                        source_frame_idx=pair.index,
                    ),
                )
                output_frame_idx += 1

            # Progress callback + checkpoint save
            if (pair.index + 1) % self.CHECKPOINT_INTERVAL == 0:
                elapsed = time.perf_counter() - start_time
                fps = (pair.index + 1) / elapsed if elapsed > 0 else 0.0

                # Progress callback
                if progress_callback:
                    progress_callback(pair.index + 1, reader.total_frames, fps)

                # Save checkpoint (independent of progress_callback)
                if checkpoint_manager and task_id:
                    checkpoint_manager.save(TaskCheckpoint(
                        task_id=task_id,
                        video_path=checkpoint.video_path if checkpoint else "",
                        output_path=checkpoint.output_path if checkpoint else "",
                        last_completed_frame=pair.index,
                        total_frames=reader.total_frames,
                        multiplier=multiplier,
                    ))

            # Periodic GPU cache cleanup
            if (pair.index + 1) % self.CACHE_CLEANUP_INTERVAL == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Flush remaining buffered frames
        buffer.flush_all()

        # Delete checkpoint on successful completion
        if checkpoint_manager and task_id:
            checkpoint_manager.delete(task_id)
            logger.info(f"Checkpoint deleted for completed task: {task_id}")


# ====================
# Task Scheduler
# ====================


class TaskScheduler:
    """Task scheduler: replaces TaskOrchestrator with streaming architecture.

    Responsibilities: task submission → parse → stream processing →
    result writing → event emission.

    State machine: IDLE → RUNNING → IDLE (or CANCELLING → IDLE)

    Unlike TaskOrchestrator, this scheduler:
    - Uses Backend.infer() instead of Backend.process_frames()
    - Uses StreamingFramePairReader for constant-memory input
    - Uses OrderedResultBuffer for out-of-order parallel results
    - Uses FrameLifecycle for write-once semantics
    - Does NOT load all frames into memory
    """

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        """Initialize the task scheduler.

        Args:
            config: Optional default backend configuration.
        """
        self._default_config = config
        self._state = SchedulerState.IDLE
        self._current_task: Optional[TaskDefinition] = None
        self._backend: Optional[BaseBackend] = None
        self._loop = ParallelStreamingLoop()

    def submit_task(self, task_def: TaskDefinition) -> str:
        """Submit a task for processing.

        Args:
            task_def: Parsed and resolved task definition.

        Returns:
            task_id: Unique task identifier.

        Raises:
            RuntimeError: If scheduler is not in IDLE state.
        """
        if self._state != SchedulerState.IDLE:
            raise RuntimeError(
                f"Cannot submit task while state={self._state.value}"
            )

        self._current_task = task_def
        self._state = SchedulerState.RUNNING
        self._emit_state_changed()

        try:
            self._run_task(task_def)
        except Exception as e:
            self._state = SchedulerState.IDLE
            self._emit_event("task_failed", task_id=task_def.task_id, error=str(e))
            self._emit_state_changed()
            raise

        return task_def.task_id

    def cancel_current(self) -> None:
        """Cancel the currently running task."""
        if self._state != SchedulerState.RUNNING:
            return

        self._state = SchedulerState.CANCELLING
        self._loop.cancel()

        if self._backend is not None:
            self._backend.cancel()

        self._state = SchedulerState.IDLE

        if self._current_task is not None:
            self._emit_event("task_cancelled", task_id=self._current_task.task_id)

        self._emit_state_changed()

    def get_state(self) -> SchedulerState:
        """Return the current scheduler state."""
        return self._state

    def _run_task(self, task_def: TaskDefinition) -> None:
        """Execute a single task using the streaming pipeline.

        Steps:
        1. Create Backend and load model
        2. Open input video with StreamingFramePairReader
        3. Create output writer via FrameWriterFactory
        4. Create PreprocessPipeline for frame pair decisions
        5. Run ParallelStreamingLoop
        6. Close writer, cleanup backend and reader
        7. Emit completion event

        Args:
            task_def: Fully resolved task definition.
        """
        from core.backends.base_backend import BackendFactory
        from core.types import BackendType

        # 1. Create and initialize backend
        self._backend = BackendFactory.create(
            task_def.backend_type, task_def.backend_config
        )
        if not self._backend.initialize():
            raise RuntimeError("Backend initialization failed")

        # Load model (if backend supports load_model, use it; otherwise
        # initialize() handles model loading for legacy backends)
        _backend_any: Any = self._backend
        if hasattr(_backend_any, "load_model"):
            _backend_any.load_model(task_def.processing_config.interpolation)

        # 2. Open input video
        reader = StreamingFramePairReader(
            task_def.video_path,
            device=task_def.backend_config.get_device(),
        )

        # 3. Create output writer
        output_config = task_def.processing_config.output
        writer = FrameWriterFactory.create_writer(
            output_path=task_def.output_path,
            codec_config=output_config,
        )
        # Convert core.types.VideoMetadata to frame_data.VideoMetadata
        # for FrameWriter compatibility (migration in progress)
        from core.io.frame_data import VideoMetadata as LegacyVideoMetadata

        meta = reader.metadata
        legacy_meta = LegacyVideoMetadata(
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            total_frames=meta.total_frames,
        )
        writer.open(task_def.output_path, legacy_meta)

        # 4. Create preprocessing pipeline
        pipeline = PreprocessPipeline(
            task_def.processing_config, task_def.backend_type
        )

        # 5. Emit start event
        self._emit_event(
            "task_started",
            task_id=task_def.task_id,
            video_path=task_def.video_path,
        )

        # 6. Run streaming loop
        multiplier = task_def.processing_config.interpolation.get("multi", 2)

        def progress_callback(current: int, total: int, fps: float) -> None:
            self._emit_event(
                "task_progress",
                task_id=task_def.task_id,
                progress={
                    "current_frame": current,
                    "total_frames": total,
                    "fps": fps,
                },
            )

        try:
            self._loop.run(
                reader=reader,
                pipeline=pipeline,
                backend=self._backend,
                writer=writer,
                multiplier=multiplier,
                progress_callback=progress_callback,
            )
        finally:
            # 7. Close writer
            writer.close()

            # 8. Cleanup backend
            self._backend.cleanup()

        # 9. Emit completion event
        self._state = SchedulerState.IDLE
        self._emit_event(
            "task_finished",
            task_id=task_def.task_id,
            output_path=str(task_def.output_path),
            error=None,
        )
        self._emit_state_changed()

    def _emit_event(self, event_name: str, **kwargs: Any) -> None:
        """Emit a Blinker event by name.

        Args:
            event_name: Event signal name (matches core/events.py signal names).
            **kwargs: Event payload keyword arguments.
        """
        from core.events import events

        signal = events.signal(event_name)
        signal.send(self, **kwargs)

    def _emit_state_changed(self) -> None:
        """Emit the scheduler state changed event."""
        from core.events import scheduler_state_changed

        scheduler_state_changed.send(self, state=self._state.value)


# ====================
# CLI Entry Point
# ====================


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="TaskScheduler CLI")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="", help="Output video path")
    parser.add_argument(
        "--model", default="rife", choices=["rife", "film", "amt", "ifrnet"]
    )
    parser.add_argument("--version", default="4.22", help="Model version")
    parser.add_argument("--multi", type=int, default=2, help="Frame multiplier")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/xpu/cpu)")
    parser.add_argument("--precision", default="fp16", help="Precision (fp16/fp32)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    try:
        from core.types import (
            BackendType,
            BackendConfig as BC,
            ProcessingConfig as PC,
            SubTaskPlan,
            TaskDefinition as TD,
        )

        # Resolve output path
        if not args.output:
            input_path = Path(args.video)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = str(
                Path("output") / f"{input_path.stem}_interp_{timestamp}.mkv"
            )

        # Build task definition
        backend_config = BC(
            backend_type=BackendType.TORCH,
            device=args.device,
            precision=args.precision,
        )
        processing_config = PC(
            interpolation={
                "model_type": args.model,
                "model_version": args.version,
                "multi": args.multi,
                "scale": 1.0,
                "scene_change": False,
            }
        )

        # Estimate subtask plan
        try:
            import av

            container = av.open(args.video)
            stream = container.streams.video[0]
            total_frames = stream.frames or 0
            container.close()
        except Exception:
            total_frames = 0

        task_def = TD(
            task_id=str(uuid.uuid4())[:8],
            video_path=args.video,
            backend_type=BackendType.TORCH,
            backend_config=backend_config,
            processing_config=processing_config,
            subtask_plan=SubTaskPlan(
                total_subtasks=total_frames * (args.multi - 1),
                input_frame_count=total_frames,
                output_frame_count=total_frames * args.multi,
                multiplier=args.multi,
                batch_size=1,
                requires_scene_detect=False,
            ),
            output_path=Path(args.output),
        )

        # Run
        scheduler = TaskScheduler(backend_config)
        start_time = time.perf_counter()
        scheduler.submit_task(task_def)
        elapsed = time.perf_counter() - start_time

        result: Dict[str, Any] = {
            "success": True,
            "output": args.output,
            "model": args.model,
            "version": args.version,
            "multiplier": args.multi,
            "elapsed_seconds": round(elapsed, 3),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Output: {result['output']}")
            print(f"Time: {result['elapsed_seconds']}s")

    except Exception as e:
        result = {"success": False, "error": str(e)}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {e}")
        raise
