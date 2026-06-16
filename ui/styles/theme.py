"""
Design tokens for VFI-gui — centralized color, spacing, font constants.

This module provides a single source of truth for all visual constants
used throughout the application. All QSS stylesheets should reference
these constants, not hardcoded values.
"""


class Theme:
    """Design tokens — centralized color, spacing, font constants."""

    # ========================================
    # COLORS - Dark Theme (VS Code-inspired)
    # ========================================

    # Background colors
    BG_PRIMARY = "#1e1e1e"
    BG_SECONDARY = "#252525"
    BG_TERTIARY = "#2d2d2d"
    BG_HOVER = "#3d3d3d"
    BG_PRESSED = "#4d4d4d"

    # Border colors
    BORDER = "#3d3d3d"
    BORDER_LIGHT = "#4d4d4d"

    # Text colors
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#999999"
    TEXT_DISABLED = "#5d5d5d"

    # Accent colors (Windows blue)
    ACCENT = "#0078d4"
    ACCENT_HOVER = "#1084d8"
    ACCENT_PRESSED = "#006cbd"
    ACCENT_DISABLED = "#4a4a4a"

    # Semantic colors
    SUCCESS = "#4caf50"
    WARNING = "#ff9800"
    ERROR = "#d42a2a"

    # ========================================
    # SPACING
    # ========================================

    PADDING_SM = 4
    PADDING_MD = 8
    PADDING_LG = 16
    PADDING_XL = 24

    SPACING_SM = 4
    SPACING_MD = 8
    SPACING_LG = 16

    # ========================================
    # TYPOGRAPHY
    # ========================================

    FONT_FAMILY = '"Segoe UI", "Microsoft YaHei", sans-serif'

    FONT_SIZE_SM = "9pt"
    FONT_SIZE_MD = "10pt"
    FONT_SIZE_LG = "12pt"
    FONT_SIZE_XL = "14pt"
    FONT_SIZE_XXL = "18pt"

    # ========================================
    # BORDER RADIUS
    # ========================================

    RADIUS_SM = 2
    RADIUS_MD = 4
    RADIUS_LG = 8
    RADIUS_XL = 12

    # ========================================
    # COMPONENT-SPECIFIC COLORS
    # ========================================

    # Progress bar
    PROGRESS_BG = "#2d2d2d"
    PROGRESS_FILL = "#0078d4"
    PROGRESS_TEXT = "#e0e0e0"

    # Scrollbar
    SCROLLBAR_BG = "#1e1e1e"
    SCROLLBAR_HANDLE = "#3d3d3d"
    SCROLLBAR_HANDLE_HOVER = "#4d4d4d"

    # Drop zone (drag & drop area)
    DROP_ZONE_BORDER = "#0078d4"
    DROP_ZONE_BG = "#1a2a3a"
    DROP_ZONE_BG_HOVER = "#1a3a5a"

    # ========================================
    # LOG COLORS
    # ========================================

    LOG_INFO = "#e0e0e0"
    LOG_WARNING = "#ff9800"
    LOG_ERROR = "#d42a2a"
    LOG_SUCCESS = "#4caf50"

    # ========================================
    # STATUS COLORS
    # ========================================

    STATUS_QUEUED = "#999999"
    STATUS_PROCESSING = "#0078d4"
    STATUS_COMPLETED = "#4caf50"
    STATUS_FAILED = "#d42a2a"
    STATUS_CANCELLED = "#ff9800"