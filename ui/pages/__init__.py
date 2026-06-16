"""Page components for VFI-gui.

This module provides the main page widgets:
- ConfigPage: Configuration view (shown when task not running)
- ProcessPage: Processing view (shown when task running)

Pages are managed by MainWindow via QStackedWidget.
"""

from ui.pages.config_page import ConfigPage
from ui.pages.process_page import ProcessPage

__all__ = ["ConfigPage", "ProcessPage"]