"""
Styles module for VFI-gui.

This module provides the theme system, icon management, and stylesheet
for the PyQt6 application.

Usage:
    from ui.styles import Theme, IconManager, DARK_THEME, apply_dark_theme

    # Apply dark theme to application
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    # Get themed icon
    icon = IconManager.get("play", color=Theme.ACCENT)

    # Use design tokens
    background_color = Theme.BG_PRIMARY
"""

from ui.styles.theme import Theme
from ui.styles.icons import IconManager
from ui.styles.stylesheet import DARK_THEME, apply_dark_theme

__all__ = [
    "Theme",
    "IconManager",
    "DARK_THEME",
    "apply_dark_theme",
]