"""StatusFilterPanel - Left sidebar for filtering tasks by status.

Similar to qBittorrent's left sidebar, shows task counts by status
and allows filtering the task list.
"""

from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea
)

from ui.styles.theme import Theme
from ui.styles.icons import IconManager

if TYPE_CHECKING:
    from ui.viewmodels.queue_viewmodel import QueueViewModel


class FilterButton(QPushButton):
    """Custom filter button with icon, label and count."""
    
    def __init__(self, icon_name: str, label: str, count: int = 0, parent=None):
        super().__init__(parent)
        self._icon_name = icon_name
        self._label = label
        self._count = count
        self._setup_ui()
        self._update_display()
    
    def _setup_ui(self):
        """Setup button appearance."""
        self.setCheckable(True)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: {Theme.RADIUS_SM}px;
                padding: 6px 12px;
                text-align: left;
                color: {Theme.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                color: {Theme.TEXT_PRIMARY};
            }}
            QPushButton:checked {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
            }}
        """)
    
    def _update_display(self):
        """Update button text with icon, label and count."""
        icon = IconManager.get(self._icon_name)
        text = f"{self._label} ({self._count})"
        self.setIcon(icon)
        self.setText(text)
    
    def set_count(self, count: int):
        """Update the count display."""
        self._count = count
        self._update_display()
    
    def set_icon(self, icon_name: str):
        """Update the icon."""
        self._icon_name = icon_name
        self._update_display()


class StatusFilterPanel(QWidget):
    """Left sidebar for filtering tasks by status.
    
    Structure similar to qBittorrent:
    - 状态 (Status) section with filters
    - 分类 (Categories) section
    - 标签 (Tags) section
    
    Signals:
        filter_changed(str): Emitted when filter selection changes
            Values: "all", "processing", "completed", "failed", "cancelled", "queued"
    """
    
    filter_changed = pyqtSignal(str)
    
    def __init__(self, vm: "QueueViewModel", parent=None):
        super().__init__(parent)
        self._vm = vm
        self._filter_buttons: dict[str, FilterButton] = {}
        self._current_filter = "all"
        self._setup_ui()
        self._bind_viewmodel()
        self._retranslate_ui()
    
    def _setup_ui(self):
        """Setup the sidebar UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {Theme.BG_SECONDARY};
            }}
            QScrollBar:vertical {{
                background-color: {Theme.BG_SECONDARY};
                width: 12px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Theme.SCROLLBAR_HANDLE};
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Theme.SCROLLBAR_HANDLE_HOVER};
            }}
        """)
        
        # Container widget
        container = QWidget()
        container.setStyleSheet(f"background-color: {Theme.BG_SECONDARY};")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 8, 0, 8)
        container_layout.setSpacing(0)
        
        # Status section
        self._status_section = self._create_section("status")
        container_layout.addWidget(self._status_section)
        
        # Add filter buttons
        self._add_filter_button("status", "all", "folder-open", 0)
        self._add_filter_button("status", "processing", "play", 0)
        self._add_filter_button("status", "queued", "pause", 0)
        self._add_filter_button("status", "completed", "check", 0)
        self._add_filter_button("status", "failed", "error", 0)
        self._add_filter_button("status", "cancelled", "stop", 0)
        
        # Select "all" by default
        self._filter_buttons["all"].setChecked(True)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"color: {Theme.BORDER};")
        container_layout.addWidget(separator)
        
        # Categories section (placeholder for future)
        self._category_section = self._create_section("categories")
        container_layout.addWidget(self._category_section)
        self._add_filter_button("categories", "cat_all", "folder-open", 0)
        self._add_filter_button("categories", "cat_uncat", "folder", 0)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet(f"color: {Theme.BORDER};")
        container_layout.addWidget(separator2)
        
        # Tags section (placeholder for future)
        self._tags_section = self._create_section("tags")
        container_layout.addWidget(self._tags_section)
        self._add_filter_button("tags", "tag_all", "tag", 0)
        self._add_filter_button("tags", "tag_none", "label", 0)
        
        container_layout.addStretch()
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Set fixed width for sidebar
        self.setFixedWidth(200)
        self.setStyleSheet(f"background-color: {Theme.BG_SECONDARY};")
    
    def _create_section(self, section_id: str) -> QWidget:
        """Create a collapsible section header."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Section header
        header = QLabel()
        header.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {Theme.FONT_SIZE_LG};
                font-weight: bold;
                padding: 4px 8px;
            }}
        """)
        # Store header reference using proper attribute name
        if section_id == "status":
            self._status_header = header
        elif section_id == "categories":
            self._categories_header = header
        elif section_id == "tags":
            self._tags_header = header
        layout.addWidget(header)
        
        # Button container
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(8, 0, 0, 0)
        btn_layout.setSpacing(2)
        setattr(self, f"_{section_id}_layout", btn_layout)
        layout.addWidget(btn_container)
        
        return widget
    
    def _add_filter_button(self, section: str, filter_id: str, icon: str, count: int):
        """Add a filter button to a section."""
        btn = FilterButton(icon, filter_id, count)
        btn.clicked.connect(lambda: self._on_filter_clicked(filter_id))
        self._filter_buttons[filter_id] = btn
        
        layout = getattr(self, f"_{section}_layout")
        layout.addWidget(btn)
    
    def _on_filter_clicked(self, filter_id: str):
        """Handle filter button click."""
        # Uncheck all buttons
        for btn in self._filter_buttons.values():
            btn.setChecked(False)
        
        # Check clicked button
        if filter_id in self._filter_buttons:
            self._filter_buttons[filter_id].setChecked(True)
            self._current_filter = filter_id
            
            # Map filter_id to filter value
            filter_map = {
                "all": "all",
                "processing": "processing",
                "queued": "queued",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }
            
            filter_value = filter_map.get(filter_id, "all")
            self.filter_changed.emit(filter_value)
    
    def _bind_viewmodel(self):
        """Bind to QueueViewModel signals."""
        if self._vm:
            self._vm.queue_changed.connect(self._update_counts)
            self._vm.item_status_changed.connect(self._update_counts)
            self._update_counts()
    
    def _update_counts(self):
        """Update filter button counts from ViewModel."""
        if not self._vm:
            return
        
        # Get counts
        total = self._vm.total_count
        completed = self._vm.completed_count
        failed = self._vm.failed_count
        
        # Calculate processing and queued
        processing = 0
        queued = 0
        cancelled = 0
        
        for item in self._vm.items():
            if item.status == "processing":
                processing += 1
            elif item.status in ("queued", "pending"):
                queued += 1
            elif item.status == "cancelled":
                cancelled += 1
        
        # Update buttons
        if "all" in self._filter_buttons:
            self._filter_buttons["all"].set_count(total)
        if "processing" in self._filter_buttons:
            self._filter_buttons["processing"].set_count(processing)
        if "queued" in self._filter_buttons:
            self._filter_buttons["queued"].set_count(queued)
        if "completed" in self._filter_buttons:
            self._filter_buttons["completed"].set_count(completed)
        if "failed" in self._filter_buttons:
            self._filter_buttons["failed"].set_count(failed)
        if "cancelled" in self._filter_buttons:
            self._filter_buttons["cancelled"].set_count(cancelled)
    
    def _retranslate_ui(self):
        """Update UI text for i18n."""
        # Section headers
        if hasattr(self, "_status_header"):
            self._status_header.setText(self.tr("状态"))
        if hasattr(self, "_categories_header"):
            self._categories_header.setText(self.tr("分类"))
        if hasattr(self, "_tags_header"):
            self._tags_header.setText(self.tr("标签"))
        
        # Update button labels
        label_map = {
            "all": self.tr("全部"),
            "processing": self.tr("处理中"),
            "queued": self.tr("等待中"),
            "completed": self.tr("已完成"),
            "failed": self.tr("失败"),
            "cancelled": self.tr("已取消"),
            "cat_all": self.tr("全部"),
            "cat_uncat": self.tr("未分类"),
            "tag_all": self.tr("全部"),
            "tag_none": self.tr("无标签"),
        }
        
        for filter_id, btn in self._filter_buttons.items():
            if filter_id in label_map:
                btn._label = label_map[filter_id]
                btn._update_display()
    
    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)
    
    def get_current_filter(self) -> str:
        """Get current filter value."""
        return self._current_filter


__all__ = ["StatusFilterPanel", "FilterButton"]
