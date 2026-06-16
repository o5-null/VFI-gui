"""
SVG icon manager for VFI-gui — supports theme color tinting.

This module provides a centralized icon management system with:
- Icon caching for performance
- Dynamic color tinting via SVG fill attribute replacement
- Graceful degradation for missing icons
"""

import re
from pathlib import Path

from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtSvg import QSvgRenderer


class IconManager:
    """SVG icon manager — supports theme color tinting."""

    _instance: "IconManager | None" = None
    _icon_cache: dict[str, QIcon] = {}

    # Icon name to filename mapping
    ICONS: dict[str, str] = {
        "play": "play-circle.svg",
        "pause": "pause-circle.svg",
        "stop": "stop-circle.svg",
        "folder-open": "folder-open.svg",
        "settings": "cog.svg",
        "download": "download.svg",
        "delete": "delete.svg",
        "info": "information.svg",
        "warning": "alert.svg",
        "error": "close-circle.svg",
        "success": "check-circle.svg",
        "queued": "clock-outline.svg",
        "processing": "loading.svg",
        "video": "video.svg",
        "image": "image.svg",
        "back": "arrow-left.svg",
        "add": "plus.svg",
        "remove": "minus.svg",
        "up": "arrow-up.svg",
        "down": "arrow-down.svg",
        "clear": "close.svg",
        "refresh": "refresh.svg",
        "about": "help-circle.svg",
        "benchmark": "speedometer.svg",
    }

    # Default icon directory
    _icons_dir: "Path | None" = None

    @classmethod
    def initialize(cls, icons_dir: "Path | None" = None) -> None:
        """Initialize the icon manager with a custom icons directory."""
        if icons_dir is None:
            # Default: icons subdirectory adjacent to this module
            cls._icons_dir = Path(__file__).parent / "icons"
        else:
            cls._icons_dir = icons_dir

        # Clear cache on re-initialization
        cls._icon_cache.clear()

    @classmethod
    def get(cls, name: str, color: str = "#e0e0e0", size: "QSize | None" = None) -> QIcon:
        """Get icon with dynamic color tinting. Returns empty QIcon if resource missing.

        Args:
            name: Icon name (e.g., "play", "settings")
            color: Fill color in hex format (e.g., "#e0e0e0")
            size: Icon size in pixels (default: 24x24)

        Returns:
            QIcon with tinted SVG, or empty QIcon if not found
        """
        # Default size if not provided
        if size is None:
            size = QSize(24, 24)

        # Ensure initialization
        if cls._icons_dir is None:
            cls.initialize()

        # Create cache key
        key = f"{name}:{color}:{size.width()}x{size.height()}"

        if key not in cls._icon_cache:
            cls._icon_cache[key] = cls._load_and_tint(name, color, size)

        return cls._icon_cache[key]

    @classmethod
    def _load_and_tint(cls, name: str, color: str, size: QSize) -> QIcon:
        """Load SVG and apply color tint. Fallback: create a simple colored pixmap.

        Args:
            name: Icon name
            color: Fill color in hex format
            size: Icon size

        Returns:
            QIcon with tinted SVG, or fallback colored pixmap
        """
        if name not in cls.ICONS:
            return cls._create_fallback_icon(color, size)

        # Ensure _icons_dir is initialized (get() always calls initialize first)
        if cls._icons_dir is None:
            return cls._create_fallback_icon(color, size)

        filename = cls.ICONS[name]
        svg_path = cls._icons_dir / filename

        if not svg_path.exists():
            return cls._create_fallback_icon(color, size)

        try:
            # Load SVG content
            svg_content = svg_path.read_text(encoding="utf-8")

            # Apply color tinting - replace fill attributes
            # Common SVG fill patterns: fill="#xxx", fill='xxx', fill="currentColor"

            # Replace explicit fill colors and currentColor
            svg_tinted = re.sub(
                r'fill=["\'](?:#[0-9a-fA-F]+|currentColor)["\']',
                f'fill="{color}"',
                svg_content,
                flags=re.IGNORECASE
            )

            # If no fill attribute exists, add one to the main element
            if "fill=" not in svg_tinted and "<svg" in svg_tinted:
                # Add fill to the svg element
                svg_tinted = re.sub(
                    r'<svg([^>]*)>',
                    f'<svg\\1 fill="{color}">',
                    svg_tinted,
                    count=1
                )

            # Create QIcon from tinted SVG
            icon = QIcon()
            pixmap = QPixmap(size)
            pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(pixmap)
            renderer = QSvgRenderer(svg_tinted.encode("utf-8"))
            renderer.render(painter)
            _ = painter.end()

            icon.addPixmap(pixmap)
            return icon

        except Exception:
            # Fallback: simple colored square
            return cls._create_fallback_icon(color, size)

    @classmethod
    def _create_fallback_icon(cls, color: str, size: QSize) -> QIcon:
        """Create a simple colored pixmap as fallback for missing icons.

        Args:
            color: Fill color in hex format
            size: Icon size

        Returns:
            QIcon with colored circle
        """
        icon = QIcon()
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Parse hex color
        qcolor = QColor(color)
        painter.setBrush(qcolor)
        painter.setPen(Qt.PenStyle.NoPen)

        # Draw a circle
        margin = 2
        radius = (size.width() - margin * 2) // 2
        center_x = size.width() // 2
        center_y = size.height() // 2

        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        _ = painter.end()

        icon.addPixmap(pixmap)
        return icon

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the icon cache. Useful when theme colors change."""
        cls._icon_cache.clear()

    @classmethod
    def get_status_icon(cls, status: str, size: "QSize | None" = None) -> QIcon:
        """Get icon for a task status.

        Args:
            status: Status name (queued, processing, completed, failed, cancelled)
            size: Icon size (default: 16x16)

        Returns:
            QIcon with appropriate status color
        """
        # Default size for status icons
        if size is None:
            size = QSize(16, 16)

        from ui.styles.theme import Theme

        status_colors = {
            "queued": Theme.STATUS_QUEUED,
            "processing": Theme.STATUS_PROCESSING,
            "completed": Theme.STATUS_COMPLETED,
            "failed": Theme.STATUS_FAILED,
            "cancelled": Theme.STATUS_CANCELLED,
        }

        status_icons = {
            "queued": "queued",
            "processing": "processing",
            "completed": "success",
            "failed": "error",
            "cancelled": "warning",
        }

        icon_name = status_icons.get(status, "info")
        color = status_colors.get(status, Theme.TEXT_SECONDARY)

        return cls.get(icon_name, color, size)