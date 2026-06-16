"""QueueController - stateless proxy for queue operations.

Provides action methods for queue management without holding state.
All state is managed by QueueManager and exposed via QueueViewModel.
"""

from typing import Any, Dict

from loguru import logger

from core.queue_manager import QueueManager


class QueueController:
    """Stateless controller for queue operations.
    
    This controller provides action methods for queue management:
    - Add items to queue
    - Remove items from queue
    - Clear queue
    
    Controllers do NOT hold state - they delegate to QueueManager.
    """
    
    def __init__(self, queue_manager: QueueManager):
        """Initialize QueueController.
        
        Args:
            queue_manager: QueueManager instance
        """
        self._queue_manager = queue_manager
    
    def add_to_queue(self, video_path: str, pipeline_config: Dict[str, Any]) -> int:
        """Add a video to the processing queue.
        
        Args:
            video_path: Path to input video
            pipeline_config: Pipeline configuration dict
            
        Returns:
            Queue item index
        """
        logger.info(f"Adding to queue: {video_path}")
        index = self._queue_manager.add_item(video_path, pipeline_config)
        return index
    
    def remove_item(self, index: int) -> bool:
        """Remove an item from the queue.
        
        Args:
            index: Item index
            
        Returns:
            True if removed, False if invalid index
        """
        logger.info(f"Removing queue item: {index}")
        return self._queue_manager.remove_item(index)
    
    def clear_queue(self) -> None:
        """Clear all items from the queue."""
        logger.info("Clearing queue")
        self._queue_manager.clear()
    
    def clear_completed(self) -> None:
        """Remove all completed items from queue."""
        logger.info("Clearing completed items")
        self._queue_manager.clear_completed()
    
    def clear_failed(self) -> None:
        """Remove all failed items from queue."""
        logger.info("Clearing failed items")
        self._queue_manager.clear_failed()
    
    def retry_failed(self) -> None:
        """Reset all failed items to pending."""
        logger.info("Retrying failed items")
        self._queue_manager.retry_failed()
    
    def move_item(self, from_index: int, to_index: int) -> bool:
        """Move an item to a new position.
        
        Args:
            from_index: Current item index
            to_index: New position
            
        Returns:
            True if moved, False if invalid indices
        """
        logger.info(f"Moving queue item from {from_index} to {to_index}")
        return self._queue_manager.move_item(from_index, to_index)
    
    def get_count(self) -> int:
        """Get total items count."""
        return self._queue_manager.get_count()
    
    def get_pending_count(self) -> int:
        """Get pending items count."""
        return self._queue_manager.get_pending_count()


__all__ = ["QueueController"]