"""Minimal stylesheet for VFI-gui.

Uses Qt's default platform style (Fusion) with no custom QSS.
Theme tokens and IconManager remain available for programmatic use.
"""

from PyQt6.QtWidgets import QApplication


# No custom stylesheet — use Qt default style
DARK_THEME = ""


def apply_dark_theme(app: QApplication) -> None:
    """Apply default Qt Fusion style (no custom QSS).

    Args:
        app: QApplication instance
    """
    # Use Qt's built-in Fusion style — clean and stable
    app.setStyle("Fusion")