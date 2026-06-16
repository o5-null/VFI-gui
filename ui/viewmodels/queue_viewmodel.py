"""QueueViewModel for batch queue management.

Wraps QueueManager to provide Qt-signals for UI binding.
Provides QueueItemVO for list/table display.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.queue_manager import QueueManager, QueueItem, QueueItemStatus


@dataclass
class QueueItemVO:
    """Value object for queue item display.
    
    Used by UI components for list/table binding.
    """
    index: int
    video_name: str
    video_path: str
    status: str
    progress: float
    fps: float
    error: Optional[str]
    
    @property
    def is_pending(self) -> bool:
        """Check if item is pending."""
        return self.status == "pending"
    
    @property
    def is_processing(self) -> bool:
        """Check if item is processing."""
        return self.status == "processing"
    
    @property
    def is_completed(self) -> bool:
        """Check if item is completed."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if item is failed."""
        return self.status == "failed"


class QueueViewModel(QObject):
    """ViewModel for batch queue management.
    
    Signals forwarded from QueueManager:
        queue_changed: Queue structure changed
        item_added: New item added (int index)
        item_removed: Item removed (int index)
        item_status_changed: Item status changed (int index, str status)
    
    Additional signals:
        total_count_changed: Total items count
        completed_count_changed: Completed items count
    
    Methods:
        items(): Get list of QueueItemVO for display
        item_at(index): Get specific QueueItemVO
    """
    
    # Forwarded from QueueManager
    queue_changed = pyqtSignal()
    item_added = pyqtSignal(int)
    item_removed = pyqtSignal(int)
    item_status_changed = pyqtSignal(int, str)
    
    # Additional signals
    total_count_changed = pyqtSignal(int)
    completed_count_changed = pyqtSignal(int)
    
    def __init__(
        self,
        queue_manager: QueueManager,
        parent=None,
    ):
        """Initialize QueueViewModel.
        
        Args:
            queue_manager: QueueManager instance to wrap
            parent: Parent QObject
        """
        super().__init__(parent)
        self._queue_manager = queue_manager
        
        # Forward QueueManager signals
        self._queue_manager.queue_changed.connect(self.queue_changed.emit)
        self._queue_manager.item_added.connect(self.item_added.emit)
        self._queue_manager.item_removed.connect(self.item_removed.emit)
        self._queue_manager.item_status_changed.connect(self._on_item_status_changed)
        
        # Track counts
        self._total_count: int = 0
        self._completed_count: int = 0
        self._update_counts()
        
        # Connect to queue_changed for count updates
        self._queue_manager.queue_changed.connect(self._update_counts)
    
    def _on_item_status_changed(self, index: int, status: str) -> None:
        """Forward item status change and update counts."""
        self.item_status_changed.emit(index, status)
        self._update_counts()
    
    def _update_counts(self) -> None:
        """Update total and completed counts."""
        total = self._queue_manager.get_count()
        completed = self._queue_manager.get_completed_count()
        
        if total != self._total_count:
            self._total_count = total
            self.total_count_changed.emit(total)
        
        if completed != self._completed_count:
            self._completed_count = completed
            self.completed_count_changed.emit(completed)
    
    # ====================
    # Properties
    # ====================
    
    @property
    def total_count(self) -> int:
        """Get total items count."""
        return self._total_count
    
    @property
    def completed_count(self) -> int:
        """Get completed items count."""
        return self._completed_count
    
    @property
    def pending_count(self) -> int:
        """Get pending items count."""
        return self._queue_manager.get_pending_count()
    
    @property
    def failed_count(self) -> int:
        """Get failed items count."""
        return self._queue_manager.get_failed_count()
    
    # ====================
    # Query Methods
    # ====================
    
    def items(self) -> List[QueueItemVO]:
        """Get all items as value objects.
        
        Returns:
            List of QueueItemVO for UI display
        """
        result: List[QueueItemVO] = []
        raw_items = self._queue_manager.get_all_items()
        
        for i, item in enumerate(raw_items):
            vo = QueueItemVO(
                index=i,
                video_name=item.filename,
                video_path=item.video_path,
                status=item.status.value,
                progress=float(item.progress),
                fps=0.0,  # FPS tracked separately by TaskViewModel
                error=item.error_message,
            )
            result.append(vo)
        
        return result
    
    def item_at(self, index: int) -> Optional[QueueItemVO]:
        """Get item at specific index.
        
        Args:
            index: Item index
            
        Returns:
            QueueItemVO or None if invalid index
        """
        item = self._queue_manager.get_item(index)
        if item is None:
            return None
        
        return QueueItemVO(
            index=index,
            video_name=item.filename,
            video_path=item.video_path,
            status=item.status.value,
            progress=float(item.progress),
            fps=0.0,
            error=item.error_message,
        )
    
    def has_pending(self) -> bool:
        """Check if there are pending items."""
        return self._queue_manager.has_pending()
    
    def get_next_pending(self) -> Optional[QueueItemVO]:
        """Get the next pending item.
        
        Returns:
            QueueItemVO or None if no pending items
        """
        item = self._queue_manager.get_next_pending()
        if item is None:
            return None
        
        # Find index
        items = self._queue_manager.get_all_items()
        for i, it in enumerate(items):
            if it == item:
                return QueueItemVO(
                    index=i,
                    video_name=item.filename,
                    video_path=item.video_path,
                    status=item.status.value,
                    progress=float(item.progress),
                    fps=0.0,
                    error=item.error_message,
                )
        
        return None
    
    def get_video_path(self, index: int) -> Optional[str]:
        """Get video path for item.
        
        Args:
            index: Item index
            
        Returns:
            Video path string or None
        """
        item = self._queue_manager.get_item(index)
        return item.video_path if item else None
    
    def get_item_config(self, index: int) -> Optional[dict]:
        """Get pipeline config for item.
        
        Args:
            index: Item index
            
        Returns:
            Config dict or None
        """
        item = self._queue_manager.get_item(index)
        return item.config if item else None
    
    def get_item_status(self, index: int) -> str:
        """Get status string for item.
        
        Args:
            index: Item index
            
        Returns:
            Status string or empty if invalid
        """
        item = self._queue_manager.get_item(index)
        return item.status.value if item else ""


__all__ = ["QueueViewModel", "QueueItemVO"]