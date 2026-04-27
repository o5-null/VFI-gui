"""Batch queue management for video processing."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger


class QueueItemStatus(Enum):
    """Status of a queue item."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """A single item in the processing queue."""
    video_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: QueueItemStatus = QueueItemStatus.PENDING
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: int = 0

    @property
    def filename(self) -> str:
        """Get the video filename."""
        return Path(self.video_path).name

    @property
    def is_pending(self) -> bool:
        """Check if item is pending processing."""
        return self.status == QueueItemStatus.PENDING

    @property
    def is_completed(self) -> bool:
        """Check if item processing completed successfully."""
        return self.status == QueueItemStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if item processing failed."""
        return self.status == QueueItemStatus.FAILED


class QueueManager(QObject):
    """Manager for the batch processing queue."""

    queue_changed = pyqtSignal()
    item_added = pyqtSignal(int)  # index
    item_removed = pyqtSignal(int)  # index
    item_status_changed = pyqtSignal(int, str)  # index, status

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._items: List[QueueItem] = []

    def add_item(
        self,
        video_path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a new item to the queue."""
        item = QueueItem(
            video_path=video_path,
            config=config or {},
        )
        self._items.append(item)
        index = len(self._items) - 1
        logger.info(f"Added to queue: {Path(video_path).name} (index: {index})")
        self.item_added.emit(index)
        self.queue_changed.emit()
        return index

    def remove_item(self, index: int) -> bool:
        """Remove an item from the queue."""
        if 0 <= index < len(self._items):
            del self._items[index]
            self.item_removed.emit(index)
            self.queue_changed.emit()
            return True
        return False

    def get_item(self, index: int) -> Optional[QueueItem]:
        """Get an item by index."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def get_all_items(self) -> List[QueueItem]:
        """Get all items in the queue."""
        return self._items.copy()

    def get_count(self) -> int:
        """Get total number of items."""
        return len(self._items)

    def get_pending_count(self) -> int:
        """Get number of pending items."""
        return sum(1 for item in self._items if item.is_pending)

    def get_completed_count(self) -> int:
        """Get number of completed items."""
        return sum(1 for item in self._items if item.is_completed)

    def get_failed_count(self) -> int:
        """Get number of failed items."""
        return sum(1 for item in self._items if item.is_failed)

    def has_pending(self) -> bool:
        """Check if there are pending items."""
        return any(item.is_pending for item in self._items)

    def get_next_pending(self) -> Optional[QueueItem]:
        """Get the next pending item."""
        for item in self._items:
            if item.is_pending:
                return item
        return None

    def set_item_status(
        self,
        index: int,
        status: QueueItemStatus,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Set the status of an item."""
        if 0 <= index < len(self._items):
            item = self._items[index]
            item.status = status
            if output_path:
                item.output_path = output_path
            if error_message:
                item.error_message = error_message
            logger.debug(f"Queue item {index} status changed to: {status.value}")
            self.item_status_changed.emit(index, status.value)
            self.queue_changed.emit()
            return True
        return False

    def set_item_progress(self, index: int, progress: int) -> bool:
        """Set the progress of an item."""
        if 0 <= index < len(self._items):
            self._items[index].progress = progress
            return True
        return False

    def clear(self):
        """Clear all items from the queue."""
        self._items.clear()
        self.queue_changed.emit()

    def clear_completed(self):
        """Remove all completed items from the queue."""
        self._items = [
            item for item in self._items
            if not item.is_completed
        ]
        self.queue_changed.emit()

    def clear_failed(self):
        """Remove all failed items from the queue."""
        self._items = [
            item for item in self._items
            if not item.is_failed
        ]
        self.queue_changed.emit()

    def retry_failed(self):
        """Reset all failed items to pending."""
        for item in self._items:
            if item.is_failed:
                item.status = QueueItemStatus.PENDING
                item.error_message = None
                item.progress = 0
        self.queue_changed.emit()

    def move_item(self, from_index: int, to_index: int) -> bool:
        """Move an item to a new position."""
        if (
            0 <= from_index < len(self._items)
            and 0 <= to_index < len(self._items)
            and from_index != to_index
        ):
            item = self._items.pop(from_index)
            self._items.insert(to_index, item)
            self.queue_changed.emit()
            return True
        return False
