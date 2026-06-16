"""VFI-gui widgets package.

This package contains reusable PyQt6 UI components:
- dialogs: Modal dialog components (Settings, Benchmark, About)
- progress: Progress monitoring widgets (ProgressBar, TaskLog, HwMonitorPlaceholder)
- queue: Queue management widgets (QueueList, QueueToolbar)
"""

# Sub-packages are imported via their own __init__.py
# Users should import from ui.widgets.progress or ui.widgets.queue directly

__all__ = ["dialogs", "progress", "queue"]