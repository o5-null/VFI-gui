"""QueueList — task queue list for ProcessPage.

Shows all queued items with status, progress, and controls.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QListWidget, QListWidgetItem,
    QPushButton, QHBoxLayout, QAbstractItemView
)
from PyQt6.QtCore import QEvent, Qt, QSize

from ui.styles.theme import Theme
from ui.styles.icons import IconManager

if TYPE_CHECKING:
    from ui.viewmodels.queue_viewmodel import QueueViewModel, QueueItemVO
    from ui.controllers.queue_controller import QueueController


class QueueList(QWidget):
    """Task queue list for ProcessPage.
    
    Shows all queued items with status, progress, and controls.
    
    Features:
        - QGroupBox titled "Task Queue"
        - QListWidget showing items with:
          - Status icon/color
          - Video filename
          - Progress percentage
          - Status text
        - Remove button
        - Clear Completed button
    """

    def __init__(self, vm: "QueueViewModel", ctrl: "QueueController", parent=None):
        """Initialize QueueList.
        
        Args:
            vm: QueueViewModel for queue data binding
            ctrl: QueueController for queue actions
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        self._ctrl = ctrl
        self.setObjectName("queueListWidget")
        self._setup_ui()
        self._bind_viewmodel()
        self.retranslate_ui()

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change for i18n."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(a0)

    def retranslate_ui(self) -> None:
        """Update all user-visible text for i18n."""
        if hasattr(self, "_group_box"):
            self._group_box.setTitle(self.tr("Task Queue"))
            self._remove_btn.setText(self.tr("Remove"))
            self._clear_completed_btn.setText(self.tr("Clear Completed"))

    def _setup_ui(self) -> None:
        """Create widget structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Theme.SPACING_SM)

        # Group box
        self._group_box = QGroupBox(self)
        self._group_box.setObjectName("queueListGroupBox")
        group_layout = QVBoxLayout(self._group_box)
        group_layout.setContentsMargins(
            Theme.PADDING_MD, Theme.PADDING_MD,
            Theme.PADDING_MD, Theme.PADDING_MD
        )
        group_layout.setSpacing(Theme.SPACING_SM)

        # Queue list widget
        self._list_widget = QListWidget()
        self._list_widget.setObjectName("queueListListWidget")
        self._list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list_widget.setAlternatingRowColors(True)
        group_layout.addWidget(self._list_widget)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(Theme.SPACING_MD)

        self._remove_btn = QPushButton()
        self._remove_btn.setObjectName("queueListRemoveBtn")
        self._remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self._remove_btn)

        self._clear_completed_btn = QPushButton()
        self._clear_completed_btn.setObjectName("queueListClearCompletedBtn")
        self._clear_completed_btn.clicked.connect(self._clear_completed)
        btn_row.addWidget(self._clear_completed_btn)

        btn_row.addStretch()
        group_layout.addLayout(btn_row)

        layout.addWidget(self._group_box)

    def _bind_viewmodel(self) -> None:
        """Connect to QueueViewModel signals."""
        self._vm.queue_changed.connect(self._refresh_list)
        self._vm.item_added.connect(self._on_item_added)
        self._vm.item_removed.connect(self._on_item_removed)
        self._vm.item_status_changed.connect(self._on_item_status_changed)

        # Initialize display
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh entire list from ViewModel."""
        self._list_widget.clear()
        items = self._vm.items()
        for item in items:
            self._add_list_item(item)

    def _add_list_item(self, item: "QueueItemVO") -> None:
        """Add a single item to the list.
        
        Args:
            item: QueueItemVO to display
        """
        list_item = QListWidgetItem()
        list_item.setData(Qt.ItemDataRole.UserRole, item.index)

        # Set display text
        text = self._format_item_text(item)
        list_item.setText(text)

        # Set status icon
        icon = IconManager.get_status_icon(item.status, QSize(16, 16))
        list_item.setIcon(icon)

        # Set status color
        color = self._get_status_color(item.status)
        list_item.setForeground(Qt.GlobalColor(color) if color else Qt.GlobalColor.gray)

        self._list_widget.addItem(list_item)

    def _format_item_text(self, item: "QueueItemVO") -> str:
        """Format item display text.
        
        Args:
            item: QueueItemVO
            
        Returns:
            Formatted display text
        """
        if item.is_processing:
            progress_pct = int(item.progress * 100)
            return f"{item.video_name} — {progress_pct}% ({self.tr('processing')})"
        elif item.is_completed:
            return f"{item.video_name} — {self.tr('completed')}"
        elif item.is_failed:
            return f"{item.video_name} — {self.tr('failed')}: {item.error or ''}"
        else:
            return f"{item.video_name} — {self.tr('queued')}"

    def _get_status_color(self, status: str) -> str:
        """Get color for status.
        
        Args:
            status: Status string
            
        Returns:
            Hex color string
        """
        status_colors = {
            "queued": Theme.STATUS_QUEUED,
            "pending": Theme.STATUS_QUEUED,
            "processing": Theme.STATUS_PROCESSING,
            "completed": Theme.STATUS_COMPLETED,
            "failed": Theme.STATUS_FAILED,
            "cancelled": Theme.STATUS_CANCELLED,
        }
        return status_colors.get(status, Theme.TEXT_SECONDARY)

    def _on_item_added(self, index: int) -> None:
        """Handle item added.
        
        Args:
            index: Added item index
        """
        item = self._vm.item_at(index)
        if item:
            self._add_list_item(item)

    def _on_item_removed(self, index: int) -> None:
        """Handle item removed.
        
        Args:
            index: Removed item index
        """
        # Find and remove item by index
        for i in range(self._list_widget.count()):
            list_item = self._list_widget.item(i)
            if list_item and list_item.data(Qt.ItemDataRole.UserRole) == index:
                self._list_widget.takeItem(i)
                break

        # Re-index remaining items
        self._refresh_list()

    def _on_item_status_changed(self, index: int, status: str) -> None:
        """Handle item status changed.
        
        Args:
            index: Item index
            status: New status
        """
        item = self._vm.item_at(index)
        if item:
            # Find and update item
            for i in range(self._list_widget.count()):
                list_item = self._list_widget.item(i)
                if list_item and list_item.data(Qt.ItemDataRole.UserRole) == index:
                    list_item.setText(self._format_item_text(item))
                    icon = IconManager.get_status_icon(status, QSize(16, 16))
                    list_item.setIcon(icon)
                    color = self._get_status_color(status)
                    list_item.setForeground(Qt.GlobalColor(color) if color else Qt.GlobalColor.gray)
                    break

    def _remove_selected(self) -> None:
        """Remove selected item from queue."""
        current = self._list_widget.currentItem()
        if current:
            index = current.data(Qt.ItemDataRole.UserRole)
            self._ctrl.remove_item(index)

    def _clear_completed(self) -> None:
        """Clear all completed items from queue."""
        self._ctrl.clear_completed()


__all__ = ["QueueList"]