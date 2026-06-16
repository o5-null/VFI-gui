"""Controllers for VFI-gui MVVM architecture.

Controllers are stateless bridges that:
- Connect Blinker signals from Core to Qt signals in ViewModels
- Provide action methods for UI commands
- Do NOT hold any state themselves

Controllers enable loose coupling between ViewModels and Core components.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from ui.controllers.processing_controller import ProcessingController
    from ui.controllers.queue_controller import QueueController
    from ui.controllers.settings_controller import SettingsController


@dataclass
class ControllerContainer:
    """Container for all Controllers in the application.
    
    Provides centralized access to Controllers for UI components.
    Created by VFIApp during initialization.
    """
    processing: "ProcessingController"
    queue: "QueueController"
    settings: "SettingsController"


__all__ = [
    "ControllerContainer",
    "ProcessingController",
    "QueueController",
    "SettingsController",
]


def __getattr__(name: str):
    """Lazy import for Controllers."""
    if name == "ProcessingController":
        from ui.controllers.processing_controller import ProcessingController
        return ProcessingController
    if name == "QueueController":
        from ui.controllers.queue_controller import QueueController
        return QueueController
    if name == "SettingsController":
        from ui.controllers.settings_controller import SettingsController
        return SettingsController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")