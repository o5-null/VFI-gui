"""Queue Controller for VFI-gui.

Handles batch queue operations, separating queue logic from MainWindow.
"""

from typing import Optional, List, Callable
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger

from core.queue_manager import QueueManager, QueueItem, QueueItemStatus
from ui.viewmodels.queue_viewmodel import QueueViewModel


class QueueController(QObject):
    """Controller for batch queue management.
    
    Encapsulates queue logic that was previously in MainWindow,
    providing a clean interface for managing batch processing.
    """
    
    # Signals forwarded from ViewModel
    item_added = pyqtSignal(str)  # item_id
    item_removed = pyqtSignal(str)  # item_id
    item_status_changed = pyqtSignal(str, str)  # item_id, new_status
    queue_cleared = pyqtSignal()
    queue_started = pyqtSignal()
    queue_finished = pyqtSignal()
    progress_updated = pyqtSignal(int, int)  # completed, total
    
    def __init__(self, queue_manager: Optional[QueueManager] = None, parent=None):
        super().__init__(parent)
        self._viewmodel = QueueViewModel(queue_manager, self)
        self._setup_connections()
        
        # Processing callback
        self._process_item_callback: Optional[Callable[[str, str], bool]] = None
    
    def _setup_connections(self):
        """Connect ViewModel signals to controller signals."""
        self._viewmodel.item_added.connect(self.item_added)
        self._viewmodel.item_removed.connect(self.item_removed)
        self._viewmodel.item_status_changed.connect(self.item_status_changed)
        self._viewmodel.queue_cleared.connect(self.queue_cleared)
        self._viewmodel.queue_started.connect(self.queue_started)
        self._viewmodel.queue_finished.connect(self.queue_finished)
    
    def set_process_callback(self, callback: Callable[[str, str], bool]) -> None:
        """Set callback for processing individual items.
        
        Args:
            callback: Function that takes (video_path, output_path) and returns success
        """
        self._process_item_callback = callback
    
    def add_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Add a video to the queue.
        
        Args:
            video_path: Path to input video
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Item ID
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Auto-generate output path if not provided
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(
                video_path_obj.parent / f"{video_path_obj.stem}_interpolated{video_path_obj.suffix}"
            )
        
        item_id = self._viewmodel.add_item(
            video_path=video_path,
            output_path=output_path,
        )
        
        logger.info(f"Added to queue: {video_path} -> {output_path}")
        return item_id
    
    def add_videos_from_folder(self, folder_path: str) -> List[str]:
        """Add all videos from a folder to the queue.
        
        Args:
            folder_path: Path to folder containing videos
            
        Returns:
            List of added item IDs
        """
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Supported video extensions
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
        
        item_ids = []
        for video_file in folder.iterdir():
            if video_file.suffix.lower() in video_extensions:
                try:
                    item_id = self.add_video(str(video_file))
                    item_ids.append(item_id)
                except Exception as e:
                    logger.warning(f"Failed to add {video_file}: {e}")
        
        logger.info(f"Added {len(item_ids)} videos from {folder_path}")
        return item_ids
    
    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the queue."""
        success = self._viewmodel.remove_item(item_id)
        if success:
            logger.info(f"Removed from queue: {item_id}")
        return success
    
    def clear_queue(self) -> None:
        """Clear all items from the queue."""
        self._viewmodel.clear_queue()
        logger.info("Queue cleared")
    
    def get_items(self) -> List[QueueItem]:
        """Get all queue items."""
        return self._viewmodel.get_items()
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return self._viewmodel.get_stats()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._viewmodel.is_empty()
    
    def get_count(self) -> int:
        """Get total number of items."""
        return self._viewmodel.get_count()
    
    def start_queue(self) -> bool:
        """Start processing the queue.
        
        Returns:
            True if queue was started
        """
        if self._viewmodel.is_running():
            logger.warning("Queue is already running")
            return False
        
        if self._viewmodel.is_empty():
            logger.warning("Cannot start empty queue")
            return False
        
        logger.info("Starting queue processing")
        return self._viewmodel.start_queue()
    
    def stop_queue(self) -> None:
        """Stop queue processing."""
        self._viewmodel.stop_queue()
        logger.info("Queue processing stopped")
    
    def is_running(self) -> bool:
        """Check if queue is running."""
        return self._viewmodel.is_running()
    
    def move_item_up(self, item_id: str) -> bool:
        """Move an item up in the queue."""
        return self._viewmodel.move_item_up(item_id)
    
    def move_item_down(self, item_id: str) -> bool:
        """Move an item down in the queue."""
        return self._viewmodel.move_item_down(item_id)
    
    def get_status_text(self) -> str:
        """Get formatted status text for display."""
        stats = self.get_stats()
        total = sum(stats.values())
        
        if total == 0:
            return "Queue is empty"
        
        parts = []
        if stats.get("pending", 0) > 0:
            parts.append(f"{stats['pending']} pending")
        if stats.get("processing", 0) > 0:
            parts.append(f"{stats['processing']} processing")
        if stats.get("completed", 0) > 0:
            parts.append(f"{stats['completed']} completed")
        if stats.get("failed", 0) > 0:
            parts.append(f"{stats['failed']} failed")
        
        return ", ".join(parts) if parts else f"{total} items"
