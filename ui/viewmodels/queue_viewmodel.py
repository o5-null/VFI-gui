"""Queue ViewModel for VFI-gui.

Acts as intermediary between BatchQueueWidget and QueueManager,
providing a clean interface for queue operations.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal

from core.queue_manager import QueueManager, QueueItem, QueueItemStatus


class QueueViewModel(QObject):
    """ViewModel for batch queue management.
    
    Decouples queue UI from QueueManager implementation details.
    Provides observable properties for queue state.
    """
    
    # Signals for UI updates
    item_added = pyqtSignal(str)  # item_id
    item_removed = pyqtSignal(str)  # item_id
    item_updated = pyqtSignal(str)  # item_id
    item_status_changed = pyqtSignal(str, str)  # item_id, new_status
    queue_cleared = pyqtSignal()
    queue_started = pyqtSignal()
    queue_finished = pyqtSignal()
    progress_updated = pyqtSignal(int, int)  # completed, total
    
    def __init__(self, queue_manager: Optional[QueueManager] = None, parent=None):
        super().__init__(parent)
        self._queue_manager = queue_manager or QueueManager()
        self._setup_connections()
    
    def _setup_connections(self):
        """Connect to QueueManager signals."""
        # Forward QueueManager signals
        self._queue_manager.item_added.connect(self.item_added)
        self._queue_manager.item_removed.connect(self.item_removed)
        self._queue_manager.item_status_changed.connect(self.item_status_changed)
        self._queue_manager.queue_finished.connect(self.queue_finished)
    
    def add_item(
        self,
        video_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an item to the queue.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            config: Optional processing configuration
            
        Returns:
            Item ID
        """
        item_id = self._queue_manager.add_item(
            video_path=video_path,
            output_path=output_path,
            config=config,
        )
        return item_id
    
    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the queue.
        
        Args:
            item_id: ID of item to remove
            
        Returns:
            True if removed successfully
        """
        return self._queue_manager.remove_item(item_id)
    
    def clear_queue(self) -> None:
        """Clear all items from the queue."""
        self._queue_manager.clear()
        self.queue_cleared.emit()
    
    def get_items(self) -> List[QueueItem]:
        """Get all queue items."""
        return self._queue_manager.get_items()
    
    def get_item(self, item_id: str) -> Optional[QueueItem]:
        """Get a specific queue item."""
        return self._queue_manager.get_item(item_id)
    
    def get_pending_items(self) -> List[QueueItem]:
        """Get items waiting to be processed."""
        return self._queue_manager.get_pending_items()
    
    def get_completed_items(self) -> List[QueueItem]:
        """Get successfully completed items."""
        return self._queue_manager.get_completed_items()
    
    def get_failed_items(self) -> List[QueueItem]:
        """Get failed items."""
        return self._queue_manager.get_failed_items()
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics.
        
        Returns:
            Dictionary with pending, processing, completed, failed counts
        """
        return self._queue_manager.get_stats()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue_manager.is_empty()
    
    def get_count(self) -> int:
        """Get total number of items in queue."""
        return self._queue_manager.get_count()
    
    def start_queue(self) -> bool:
        """Start processing the queue.
        
        Returns:
            True if queue was started
        """
        if self._queue_manager.start():
            self.queue_started.emit()
            return True
        return False
    
    def stop_queue(self) -> None:
        """Stop queue processing."""
        self._queue_manager.stop()
    
    def is_running(self) -> bool:
        """Check if queue is currently being processed."""
        return self._queue_manager.is_running()
    
    def move_item_up(self, item_id: str) -> bool:
        """Move an item up in the queue.
        
        Args:
            item_id: ID of item to move
            
        Returns:
            True if moved successfully
        """
        success = self._queue_manager.move_item_up(item_id)
        if success:
            self.item_updated.emit(item_id)
        return success
    
    def move_item_down(self, item_id: str) -> bool:
        """Move an item down in the queue.
        
        Args:
            item_id: ID of item to move
            
        Returns:
            True if moved successfully
        """
        success = self._queue_manager.move_item_down(item_id)
        if success:
            self.item_updated.emit(item_id)
        return success
