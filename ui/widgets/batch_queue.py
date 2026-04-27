"""Batch queue widget for managing multiple video processing tasks."""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction

from core.queue_manager import QueueManager, QueueItem, QueueItemStatus
from core import tr


class BatchQueueWidget(QWidget):
    """Widget for displaying and managing the batch processing queue."""

    start_batch_requested = pyqtSignal()
    item_selected = pyqtSignal(int)  # index

    # Status colors
    STATUS_COLORS = {
        QueueItemStatus.PENDING: "#e0e0e0",
        QueueItemStatus.PROCESSING: "#0078d4",
        QueueItemStatus.COMPLETED: "#4caf50",
        QueueItemStatus.FAILED: "#f44336",
        QueueItemStatus.CANCELLED: "#ff9800",
    }

    def __init__(
        self,
        queue_manager: QueueManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._queue_manager = queue_manager
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Queue group
        self.queue_group = QGroupBox(tr("Batch Queue"))
        group_layout = QVBoxLayout(self.queue_group)

        # Stats row
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel(tr("{} items").format(0))
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        group_layout.addLayout(stats_layout)

        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.queue_list.customContextMenuRequested.connect(self._show_context_menu)
        self.queue_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.queue_list.itemSelectionChanged.connect(self._on_selection_changed)
        group_layout.addWidget(self.queue_list)

        # Action buttons
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton(tr("Start Batch"))
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self.start_batch_requested.emit)
        btn_layout.addWidget(self.start_btn)

        self.clear_btn = QPushButton(tr("Clear"))
        self.clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.clear_btn)

        group_layout.addLayout(btn_layout)
        layout.addWidget(self.queue_group)
    
    def retranslate_ui(self):
        """Retranslate UI elements after language change."""
        self.queue_group.setTitle(tr("Batch Queue"))
        self.start_btn.setText(tr("Start Batch"))
        self.clear_btn.setText(tr("Clear"))
        self._update_stats()

    def _connect_signals(self):
        """Connect queue manager signals."""
        self._queue_manager.queue_changed.connect(self.refresh)
        self._queue_manager.item_added.connect(self._on_item_added)
        self._queue_manager.item_removed.connect(self._on_item_removed)
        self._queue_manager.item_status_changed.connect(self._on_item_status_changed)

    def refresh(self):
        """Refresh the queue display."""
        self.queue_list.clear()

        for index, item in enumerate(self._queue_manager.get_all_items()):
            list_item = QListWidgetItem()
            self._update_list_item(list_item, index, item)
            self.queue_list.addItem(list_item)

        self._update_stats()

    def _update_list_item(
        self,
        list_item: QListWidgetItem,
        index: int,
        item: QueueItem,
    ):
        """Update a QListWidgetItem with queue item data."""
        # Set display text
        status_icon = self._get_status_icon(item.status)
        list_item.setText(f"{status_icon} {item.filename}")
        list_item.setData(Qt.ItemDataRole.UserRole, index)

        # Set color based on status
        color = self.STATUS_COLORS.get(item.status, "#e0e0e0")
        list_item.setForeground(Qt.GlobalColor.white if item.status in [
            QueueItemStatus.PROCESSING,
            QueueItemStatus.COMPLETED,
        ] else Qt.GlobalColor.white)

        # Set tooltip
        tooltip_parts = [f"Path: {item.video_path}", f"Status: {item.status.value}"]
        if item.output_path:
            tooltip_parts.append(f"Output: {item.output_path}")
        if item.error_message:
            tooltip_parts.append(f"Error: {item.error_message}")
        list_item.setToolTip("\n".join(tooltip_parts))

    def _get_status_icon(self, status: QueueItemStatus) -> str:
        """Get status icon character."""
        icons = {
            QueueItemStatus.PENDING: "○",
            QueueItemStatus.PROCESSING: "◐",
            QueueItemStatus.COMPLETED: "●",
            QueueItemStatus.FAILED: "✗",
            QueueItemStatus.CANCELLED: "○",
        }
        return icons.get(status, "○")

    def _update_stats(self):
        """Update the stats label."""
        total = self._queue_manager.get_count()
        pending = self._queue_manager.get_pending_count()
        completed = self._queue_manager.get_completed_count()
        failed = self._queue_manager.get_failed_count()

        parts = [tr("{} items").format(total)]
        if pending > 0:
            parts.append(tr("{} pending").format(pending))
        if completed > 0:
            parts.append(tr("{} done").format(completed))
        if failed > 0:
            parts.append(tr("{} failed").format(failed))

        self.stats_label.setText(" | ".join(parts))

    def _show_context_menu(self, pos):
        """Show context menu for queue items."""
        item = self.queue_list.itemAt(pos)
        if not item:
            return

        index = item.data(Qt.ItemDataRole.UserRole)
        queue_item = self._queue_manager.get_item(index)
        if not queue_item:
            return

        menu = QMenu(self)

        # Remove action
        remove_action = QAction(tr("Remove"), self)
        remove_action.triggered.connect(lambda: self._on_remove_item(index))
        menu.addAction(remove_action)

        # Status-specific actions
        if queue_item.status == QueueItemStatus.FAILED:
            retry_action = QAction(tr("Retry"), self)
            retry_action.triggered.connect(lambda: self._on_retry_item(index))
            menu.addAction(retry_action)

        menu.exec(self.queue_list.mapToGlobal(pos))

    def _on_item_added(self, index: int):
        """Handle item added signal."""
        item = self._queue_manager.get_item(index)
        if item:
            list_item = QListWidgetItem()
            self._update_list_item(list_item, index, item)
            self.queue_list.addItem(list_item)
            self._update_stats()

    def _on_item_removed(self, index: int):
        """Handle item removed signal."""
        if 0 <= index < self.queue_list.count():
            self.queue_list.takeItem(index)
            self._update_stats()

    def _on_item_status_changed(self, index: int, status: str):
        """Handle item status changed signal."""
        if 0 <= index < self.queue_list.count():
            item = self._queue_manager.get_item(index)
            if item:
                list_item = self.queue_list.item(index)
                self._update_list_item(list_item, index, item)
                self._update_stats()

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on queue item."""
        index = item.data(Qt.ItemDataRole.UserRole)
        self.item_selected.emit(index)

    def _on_selection_changed(self):
        """Handle selection changed."""
        selected = self.queue_list.selectedItems()
        if selected:
            index = selected[0].data(Qt.ItemDataRole.UserRole)
            self.item_selected.emit(index)

    def _on_remove_item(self, index: int):
        """Remove an item from the queue."""
        self._queue_manager.remove_item(index)

    def _on_retry_item(self, index: int):
        """Retry a failed item."""
        self._queue_manager.set_item_status(index, QueueItemStatus.PENDING)

    def _on_clear(self):
        """Clear all completed items."""
        self._queue_manager.clear_completed()
