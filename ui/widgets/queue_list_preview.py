"""QueueListPreview — compact queue preview for ConfigPage sidebar.

A QGroupBox with QListWidget showing queued items with status indicators.
Full queue management happens via QueueController.
"""

from typing import TYPE_CHECKING, List

from PyQt6.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
)
from PyQt6.QtCore import QEvent, Qt

from ui.styles.icons import IconManager
from ui.styles.theme import Theme

if TYPE_CHECKING:
    from ui.viewmodels.queue_viewmodel import QueueViewModel, QueueItemVO
    from ui.controllers.queue_controller import QueueController


class QueueListPreview(QWidget):
    """Compact queue preview for ConfigPage sidebar.

    Features:
    - QListWidget showing each item's filename + status icon
    - Status display: ⏳ Queued, 🔄 Processing, ✅ Completed, ❌ Failed
    - Number of items count label
    - "Clear Completed" button
    - Empty state: "No items in queue"

    Refreshes when QueueViewModel signals fire.
    """

    def __init__(
        self,
        vm: "QueueViewModel",
        ctrl: "QueueController",
        parent=None,
    ):
        """Initialize QueueListPreview.

        Args:
            vm: QueueViewModel for state binding
            ctrl: QueueController for action delegation
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        self._ctrl = ctrl
        self.setObjectName("queueListPreview")
        self._setup_ui()
        self._bind_viewmodel()
        self._refresh_list()
        self.retranslate_ui()

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change for i18n."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(a0)

    def retranslate_ui(self) -> None:
        """Update all user-visible text for i18n."""
        self._group.setTitle(self.tr("Queue"))
        self._count_label.setText(self._get_count_text())
        self._clear_btn.setText(self.tr("Clear Completed"))
        self._empty_label.setText(self.tr("No items in queue"))

    def _setup_ui(self) -> None:
        """Create widget structure."""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(Theme.SPACING_SM)

        # Group box container
        self._group = QGroupBox(self)
        self._group.setObjectName("queueGroup")
        group_layout = QVBoxLayout(self._group)
        group_layout.setContentsMargins(
            Theme.PADDING_MD, Theme.PADDING_MD,
            Theme.PADDING_MD, Theme.PADDING_MD
        )
        group_layout.setSpacing(Theme.SPACING_SM)

        # Count label
        self._count_label = QLabel()
        self._count_label.setObjectName("queueCountLabel")
        group_layout.addWidget(self._count_label)

        # Queue list
        self._list_widget = QListWidget()
        self._list_widget.setObjectName("queueListWidget")
        self._list_widget.setMaximumHeight(150)
        group_layout.addWidget(self._list_widget)

        # Empty state label (shown when list is empty)
        self._empty_label = QLabel()
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setObjectName("queueEmptyLabel")
        group_layout.addWidget(self._empty_label)

        # Clear completed button
        self._clear_btn = QPushButton()
        self._clear_btn.setObjectName("clearCompletedBtn")
        self._clear_btn.clicked.connect(self._on_clear_completed)
        group_layout.addWidget(self._clear_btn)

        self._layout.addWidget(self._group)

    def _bind_viewmodel(self) -> None:
        """Connect to QueueViewModel signals."""
        self._vm.queue_changed.connect(self._refresh_list)
        self._vm.item_added.connect(self._on_item_added)
        self._vm.item_removed.connect(self._on_item_removed)
        self._vm.item_status_changed.connect(self._on_item_status_changed)
        self._vm.total_count_changed.connect(self._update_count)
        self._vm.completed_count_changed.connect(self._update_count)

    def _refresh_list(self) -> None:
        """Refresh the entire list from ViewModel."""
        self._list_widget.clear()
        items: List["QueueItemVO"] = self._vm.items()

        for item in items:
            self._add_list_item(item)

        self._update_empty_state()
        self._update_count()

    def _add_list_item(self, item: "QueueItemVO") -> None:
        """Add a single item to the list."""
        list_item = QListWidgetItem()
        list_item.setData(Qt.ItemDataRole.UserRole, item.index)

        # Set icon based on status
        icon = IconManager.get_status_icon(item.status, size=None)
        list_item.setIcon(icon)

        # Set text: filename + status indicator
        status_text = self._get_status_text(item.status)
        list_item.setText(f"{item.video_name} ({status_text})")

        # Set tooltip with full path
        list_item.setToolTip(item.video_path)

        self._list_widget.addItem(list_item)

    def _get_status_text(self, status: str) -> str:
        """Get localized status text."""
        status_map = {
            "pending": self.tr("Queued"),
            "processing": self.tr("Processing"),
            "completed": self.tr("Completed"),
            "failed": self.tr("Failed"),
            "cancelled": self.tr("Cancelled"),
        }
        return status_map.get(status, status)

    def _get_count_text(self) -> str:
        """Get count display text."""
        total = self._vm.total_count
        completed = self._vm.completed_count
        return self.tr(f"{total} items ({completed} completed)")

    def _update_count(self) -> None:
        """Update count label."""
        self._count_label.setText(self._get_count_text())

    def _update_empty_state(self) -> None:
        """Toggle empty state visibility."""
        has_items = self._list_widget.count() > 0
        self._list_widget.setVisible(has_items)
        self._empty_label.setVisible(not has_items)
        self._clear_btn.setEnabled(has_items and self._vm.completed_count > 0)

    def _on_item_added(self, index: int) -> None:
        """Handle item added signal."""
        item = self._vm.item_at(index)
        if item:
            self._add_list_item(item)
            self._update_empty_state()
            self._update_count()

    def _on_item_removed(self, index: int) -> None:
        """Handle item removed signal."""
        # Find and remove the item with matching index
        for i in range(self._list_widget.count()):
            list_item = self._list_widget.item(i)
            if list_item is not None:
                item_index = list_item.data(Qt.ItemDataRole.UserRole)
                if item_index == index:
                    self._list_widget.takeItem(i)
                    break
        self._update_empty_state()
        self._update_count()

    def _on_item_status_changed(self, index: int, status: str) -> None:
        """Handle item status change."""
        # Find and update the item
        for i in range(self._list_widget.count()):
            list_item = self._list_widget.item(i)
            if list_item is not None:
                item_index = list_item.data(Qt.ItemDataRole.UserRole)
                if item_index == index:
                    icon = IconManager.get_status_icon(status, size=None)
                    list_item.setIcon(icon)
                    status_text = self._get_status_text(status)
                    # Keep filename, update status
                    current_text = list_item.text()
                    filename = current_text.split(" (")[0]
                    list_item.setText(f"{filename} ({status_text})")
                    break
        self._update_empty_state()

    def _on_clear_completed(self) -> None:
        """Handle clear completed button click."""
        self._ctrl.clear_completed()


__all__ = ["QueueListPreview"]