"""ViewModels for VFI-gui MVVM architecture.

Each ViewModel wraps a Core component and provides:
- Qt signals for UI binding
- Property getters/setters for state access
- Explicit persist() for configuration saving
- Value objects for list/table data

ViewModels are Widget-agnostic and can be reused across different UI implementations.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from ui.viewmodels.codec_viewmodel import CodecViewModel
    from ui.viewmodels.pipeline_viewmodel import PipelineViewModel
    from ui.viewmodels.task_viewmodel import TaskViewModel
    from ui.viewmodels.queue_viewmodel import QueueViewModel
    from ui.viewmodels.device_viewmodel import DeviceViewModel


@dataclass
class ViewModelContainer:
    """Container for all ViewModels in the application.
    
    Provides centralized access to ViewModels for Controllers and Widgets.
    Created by VFIApp during initialization.
    """
    pipeline: "PipelineViewModel"
    task: "TaskViewModel"
    queue: "QueueViewModel"
    device: "DeviceViewModel"
    codec: "CodecViewModel"


__all__ = [
    "ViewModelContainer",
    "CodecViewModel",
    "PipelineViewModel",
    "TaskViewModel",
    "QueueViewModel",
    "DeviceViewModel",
]


def __getattr__(name: str):
    """Lazy import for ViewModels."""
    if name == "CodecViewModel":
        from ui.viewmodels.codec_viewmodel import CodecViewModel
        return CodecViewModel
    if name == "PipelineViewModel":
        from ui.viewmodels.pipeline_viewmodel import PipelineViewModel
        return PipelineViewModel
    if name == "TaskViewModel":
        from ui.viewmodels.task_viewmodel import TaskViewModel
        return TaskViewModel
    if name == "QueueViewModel":
        from ui.viewmodels.queue_viewmodel import QueueViewModel
        return QueueViewModel
    if name == "DeviceViewModel":
        from ui.viewmodels.device_viewmodel import DeviceViewModel
        return DeviceViewModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")