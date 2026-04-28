"""Event system for VFI-gui core layer.

This module provides a Qt-independent event system using Blinker signals.
It allows core components to emit events that UI and other components can subscribe to,
without creating a dependency on PyQt6.

Usage:
    from core.events import models_updated, engines_updated

    # Subscribe to events
    models_updated.connect(my_callback)

    # Emit events
    models_updated.send(self, count=5)

    # Disconnect
    models_updated.disconnect(my_callback)
"""

from blinker import Signal, Namespace

# Create a namespace for VFI-gui events
# This prevents signal name collisions with other libraries
events = Namespace()

# ====================
# Model Manager Events
# ====================

# Emitted when TensorRT engines are rescanned
# Sender: ModelManager instance
# Args: None (check ModelManager._engines for new data)
engines_updated = events.signal("engines-updated")

# Emitted when model checkpoints are rescanned
# Sender: ModelManager instance
# Args: None (check ModelManager._model_types for new data)
models_updated = events.signal("models-updated")


# ====================
# Processing Events
# ====================

# Emitted when processing state changes
# Sender: Processor or ViewModel instance
# Args: state (str) - "idle", "processing", "cancelling", "error"
processing_state_changed = events.signal("processing-state-changed")

# Emitted during processing with progress info
# Sender: Processor instance
# Args: frame (int), total (int), fps (float)
processing_progress = events.signal("processing-progress")

# Emitted when processing completes
# Sender: Processor or ViewModel instance
# Args: success (bool), message (str)
processing_finished = events.signal("processing-finished")


# ====================
# Task Orchestrator Events (new in v2)
# ====================

# Emitted when a task starts processing
# Sender: TaskOrchestrator instance
# Args: task_id (str), video_path (str)
task_started = events.signal("task-started")

# Emitted during task processing with progress
# Sender: TaskOrchestrator instance
# Args: task_id (str), progress (TaskProgress)
task_progress = events.signal("task-progress")

# Emitted when a task completes successfully
# Sender: TaskOrchestrator instance
# Args: task_id (str), output_path (str)
task_finished = events.signal("task-finished")

# Emitted when a task fails
# Sender: TaskOrchestrator instance
# Args: task_id (str), error (str)
task_failed = events.signal("task-failed")

# Emitted when a task is cancelled
# Sender: TaskOrchestrator instance
# Args: task_id (str)
task_cancelled = events.signal("task-cancelled")

# Emitted when orchestrator state changes
# Sender: TaskOrchestrator instance
# Args: state (str) - "idle", "running", "paused", "cancelling", "shutting_down"
orchestrator_state_changed = events.signal("orchestrator-state-changed")


# ====================
# Queue Events
# ====================

# Emitted when queue changes (add/remove/reorder)
# Sender: QueueManager instance
# Args: None
queue_changed = events.signal("queue-changed")

# Emitted when a queue item status changes
# Sender: QueueManager instance
# Args: index (int), status (QueueItemStatus)
queue_item_status_changed = events.signal("queue-item-status-changed")


# ====================
# Download Events
# ====================

# Emitted during model download
# Sender: DownloadWorker instance
# Args: progress (int), message (str)
download_progress = events.signal("download-progress")

# Emitted when download completes
# Sender: DownloadWorker instance
# Args: success (bool), message (str)
download_finished = events.signal("download-finished")


# ====================
# Device Events
# ====================

# Emitted when available devices change
# Sender: DeviceManager instance
# Args: devices (List[DeviceInfo])
devices_changed = events.signal("devices-changed")


__all__ = [
    "events",
    "engines_updated",
    "models_updated",
    "processing_state_changed",
    "processing_progress",
    "processing_finished",
    "task_started",
    "task_progress",
    "task_finished",
    "task_failed",
    "task_cancelled",
    "orchestrator_state_changed",
    "queue_changed",
    "queue_item_status_changed",
    "download_progress",
    "download_finished",
    "devices_changed",
]