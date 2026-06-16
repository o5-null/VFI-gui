# TaskOrchestrator Refactoring Plan

## Goal

Create a TaskOrchestrator that:
1. Controls the entire task workflow (read settings → schedule → coordinate backend + IO)
2. Only reads settings and performs task scheduling
3. Sends commands to backend and IO for loading/processing
4. Implements process pool management (future: async)
5. Streaming output: processed frames saved immediately, memory released

## Scope

- **This PR**: TaskOrchestrator + MainWindow decoupling ONLY
- **Future PR**: TorchBackend true frame-by-frame streaming refactor
- **Rationale**: Even partial streaming (Orchestrator doesn't double-buffer) saves 1x memory. Full backend streaming requires rewriting model inference logic.

## Current Architecture Problems

1. MainWindow (796 lines) directly creates Processor, bypasses ProcessingController/ProcessingViewModel
2. Processor.run() collects ALL frames into memory (frames.append()) before saving - double memory usage
3. ProcessingWorker is a 44-line placeholder, does nothing useful
4. No process pool for batch processing

## Existing Infrastructure

- `core/backends/` - BaseBackend ABC with process_frames() generator, BackendFactory, BackendConfig, ProcessingConfig
- `core/io/frame_reader.py` - FrameReader ABC with read_frames_iter(), VideoFrameReader, ImageSequenceReader, FrameReaderFactory
- `core/io/frame_writer.py` - FrameWriter ABC with open()/write_frame()/close() streaming pattern, VideoFrameWriter, ImageSequenceWriter, FrameWriterFactory
- `core/io/frame_data.py` - FrameData, ProcessedFrameData, VideoMetadata, FrameBatch
- `core/queue_manager.py` - QueueManager(QObject) with QueueItem/QueueItemStatus
- `core/events.py` - Blinker-based Qt-independent signals
- `core/codec_manager.py` - CodecManager, CodecConfig for FFmpeg command building
- `core/config/config_facade.py` - ConfigFacade, accessed via `core.get_config()`

## New Architecture

```
MainWindow (thin: UI layout + event routing only)
    |
    v
ProcessingController (Qt signals <-> Blinker events bridge)
    |
    v
TaskOrchestrator (pure logic coordinator, NOT a QThread)
    |
    +---> Backend (via BackendFactory) - process_frames() generator
    |
    +---> FrameWriter - write_frame() streaming
```

## TaskOrchestrator Design

### File: `core/task_orchestrator.py`

```python
class OrchestratorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    SHUTTING_DOWN = "shutting_down"

class TaskState(Enum):
    PENDING = "pending"
    LOADING = "loading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskProgress:
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    stage: str = ""
    elapsed_seconds: float = 0.0

@dataclass
class TaskContext:
    task_id: str
    video_path: str
    pipeline_config: dict
    state: TaskState = TaskState.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

class TaskOrchestrator:
    """Orchestrates video processing tasks.

    Reads settings -> schedules tasks -> coordinates backend + IO.
    Supports streaming pipeline: process frame -> write frame -> release memory.

    NOT a QThread. Runs processing in an internal worker thread.
    Communicates via Blinker events (core/events.py) for Qt-independence.
    """

    def __init__(self, config: "ConfigFacade"):
        # config is core.config.config_facade.ConfigFacade, via core.get_config()
        self._config = config
        self._current_task: Optional[TaskContext] = None
        self._task_queue: List[TaskContext] = []
        self._state: OrchestratorState = OrchestratorState.IDLE
        self._worker_thread: Optional[QThread] = None

    # Task submission
    def submit_task(self, video_path: str, pipeline_config: dict) -> str  # returns task_id
    def submit_batch(self, items: List[Tuple[str, dict]]) -> List[str]

    # Lifecycle control
    def start(self) -> None
    def pause(self) -> None
    def resume(self) -> None
    def cancel_current(self) -> None
    def cancel_all(self) -> None
    def shutdown(self) -> None

    # Query
    def get_state(self) -> OrchestratorState
    def get_current_task(self) -> Optional[TaskContext]
    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]
    def has_pending_tasks(self) -> bool

    # Internal
    def _run_task(self, task: TaskContext) -> None
    def _resolve_backend_config(self, pipeline_config: dict) -> BackendConfig
    def _resolve_processing_config(self, pipeline_config: dict) -> ProcessingConfig
    def _resolve_output_path(self, task: TaskContext, metadata: dict) -> Path
```

## Streaming Pipeline (_run_task)

```python
def _run_task(self, task: TaskContext) -> None:
    """Execute a single task with streaming pipeline."""
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
        writer = None

        for frame_idx, frame_data, metadata in backend.process_frames(
            video_path=task.video_path,
            processing_config=processing_config,
            progress_callback=...,
            stage_callback=...,
            log_callback=...,
        ):
            if self._state == OrchestratorState.CANCELLING:
                break

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
            writer.write_frame(ProcessedFrameData(
                data=frame_data,
                source_frame_idx=frame_idx,
            ))

            # Update progress
            task.progress.current_frame = frame_idx
            task.progress.total_frames = metadata.get("total_frames", 0)
            task_progress.send(self, task_id=task.task_id, progress=task.progress)

            # frame_data can now be garbage collected

        # Finalize
        if writer:
            writer.close()
        task.output_path = str(output_path) if not first_yield else None
        task.state = TaskState.COMPLETED

    except Exception as e:
        task.state = TaskState.FAILED
        task.error = str(e)
    finally:
        backend.cleanup()
```

## ProcessingController Changes

File: `ui/controllers/processing_controller.py` (rewrite)

```python
class ProcessingController(QObject):
    """Bridge between Qt UI and TaskOrchestrator.

    Connects Qt signals from MainWindow to TaskOrchestrator's Blinker events.
    Handles UI-specific logic (confirm dialogs, status messages).
    """

    # Qt signals for UI
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(bool, str)
    processing_cancelled = pyqtSignal()
    progress_updated = pyqtSignal(int, int, float)
    error_occurred = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(self, orchestrator: TaskOrchestrator, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._connect_events()

    def _connect_events(self):
        """Bridge Blinker events -> Qt signals."""
        from core.events import task_started, task_progress, task_finished, task_failed
        task_started.connect(self._on_task_started)
        task_progress.connect(self._on_task_progress)
        task_finished.connect(self._on_task_finished)
        task_failed.connect(self._on_task_failed)

    # Delegate to orchestrator
    def start_processing(self, video_path: str, pipeline_config: dict) -> str
    def cancel_processing(self) -> bool
    def pause_processing(self) -> None
    def resume_processing(self) -> None

    # Blinker -> Qt signal bridges
    def _on_task_started(self, sender, **kwargs): ...
    def _on_task_progress(self, sender, **kwargs): ...
    def _on_task_finished(self, sender, **kwargs): ...
    def _on_task_failed(self, sender, **kwargs): ...
```

## MainWindow Changes

### Before (direct Processor):
```python
# MainWindow._start_processing():
processor = Processor(backend_config)
processor.set_video(video_path)
processor.set_processing_config(processing_config)
processor.progress_updated.connect(self.progress_panel.update_progress)
processor.start()
```

### After (via ProcessingController):
```python
# MainWindow.__init__():
self._orchestrator = TaskOrchestrator(self.config)
self._controller = ProcessingController(self._orchestrator)

# MainWindow._on_start_processing():
pipeline_config = self.pipeline_config.get_config()
task_id = self._controller.start_processing(video_path, pipeline_config)

# Signal connections:
self._controller.progress_updated.connect(self.progress_panel.update_progress)
self._controller.processing_finished.connect(self._on_processing_finished)
```

### Removed from MainWindow:
- Direct `Processor` import and creation
- `BackendConfig`/`ProcessingConfig` construction
- Signal wiring to Processor
- `_start_processing()` business logic
- `_apply_performance_settings()` backend config manipulation

## Events Additions (core/events.py)

```python
# Task-level events (new)
task_started = events.signal("task-started")       # Args: task_id, video_path
task_progress = events.signal("task-progress")      # Args: task_id, progress (TaskProgress)
task_finished = events.signal("task-finished")      # Args: task_id, output_path
task_failed = events.signal("task-failed")          # Args: task_id, error
task_cancelled = events.signal("task-cancelled")    # Args: task_id
orchestrator_state_changed = events.signal("orchestrator-state-changed")  # Args: state
```

## File Changes Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `core/task_orchestrator.py` | NEW | ~300 |
| `core/events.py` | MODIFY | +8 |
| `ui/controllers/processing_controller.py` | REWRITE | ~150 (from 121) |
| `ui/main_window.py` | MODIFY | -150 lines of business logic |
| `ui/viewmodels/processing_viewmodel.py` | KEEP (deprecated) | 0 |
| `core/processor.py` | KEEP (deprecated) | 0 |
| `core/workers/processing_worker.py` | KEEP (deprecated) | 0 |

## Commit Strategy

| Commit | Content | Risk |
|--------|---------|------|
| 1/4 | `core/task_orchestrator.py` + `core/events.py` updates | Low - new file + small change |
| 2/4 | Rewrite `ui/controllers/processing_controller.py` | Medium - bridges new orchestrator |
| 3/4 | Refactor `ui/main_window.py` to use ProcessingController | High - main UI change |
| 4/4 | Add deprecation warnings to Processor/ProcessingWorker/ProcessingViewModel | Low - just warnings |

## Verification Criteria

1. TaskOrchestrator can process a single video end-to-end
2. MainWindow uses ProcessingController -> TaskOrchestrator, no direct Processor usage
3. Streaming: frames written immediately, not double-buffered in orchestrator
4. LSP diagnostics: no new errors in changed files
5. UI functionality preserved: single video processing, batch queue, pause/cancel
6. Blinker events fire correctly: task_started, task_progress, task_finished, task_failed

## Known Limitations (Deferred to Future PRs)

1. TorchBackend still reads ALL frames into memory before processing - true frame-level streaming requires backend refactor
2. TaskScheduler abstraction for parallel processing - YAGNI for now
3. ProcessingViewModel/Processor/ProcessingWorker deletion - just deprecated, not removed
4. No benchmark comparing memory usage before/after (would need large test video)
