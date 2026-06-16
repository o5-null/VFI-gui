"""Dialog components for VFI-gui.

This module provides modal dialog widgets:
- AddTaskDialog: Add new task with configuration
- SettingsDialog: Application settings configuration
- BenchmarkDialog: Performance benchmarking
- AboutDialog: Application information
- ModelManagerDialog: Model checkpoint management
"""

from ui.widgets.dialogs.add_task_dialog import AddTaskDialog
from ui.widgets.dialogs.settings_dialog import SettingsDialog
from ui.widgets.dialogs.benchmark_dialog import BenchmarkDialog
from ui.widgets.dialogs.about_dialog import AboutDialog
from ui.widgets.dialogs.model_manager_dialog import ModelManagerDialog

__all__ = [
    "AddTaskDialog",
    "SettingsDialog",
    "BenchmarkDialog",
    "AboutDialog",
    "ModelManagerDialog",
]