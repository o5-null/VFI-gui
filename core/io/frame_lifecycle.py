"""Frame lifecycle: track frame consumers and write-once semantics.

Rules:
1. Each original frame is written at most once
2. A frame is released only when all its consumers complete
3. Scene cut / duplicate frames → original frame written directly, no Backend
"""

from typing import Dict, Set


class FrameLifecycle:
    """Frame lifecycle manager for streaming interpolation.

    Manages two concerns:
    - Write-once: ensures each source frame appears exactly once in output
    - Consumer tracking: a frame is released only when all SubTasks
      that reference it have completed

    Usage:
        lifecycle = FrameLifecycle()
        lifecycle.register(frame_idx, subtask_id)
        ...
        lifecycle.can_write(frame_idx)  # True if not yet written
        lifecycle.mark_written(frame_idx)
        ...
        lifecycle.can_release(frame_idx, subtask_id)  # True if no consumers left
    """

    def __init__(self) -> None:
        self._consumers: Dict[int, Set[str]] = {}  # frame_index → {subtask_ids}
        self._written: Set[int] = set()  # frame indices already written

    def register(self, frame_index: int, subtask_id: str) -> None:
        """Register a subtask as a consumer of a frame.

        A frame may be consumed by multiple subtasks (e.g., frame_i is
        frame0 for pair (i, i+1) and frame1 for pair (i-1, i)).

        Args:
            frame_index: Source frame index.
            subtask_id: ID of the consuming subtask.
        """
        if frame_index not in self._consumers:
            self._consumers[frame_index] = set()
        self._consumers[frame_index].add(subtask_id)

    def can_write(self, frame_index: int) -> bool:
        """Check whether a frame can be written (has not been written yet).

        Args:
            frame_index: Source frame index.

        Returns:
            True if the frame has not been written yet.
        """
        return frame_index not in self._written

    def mark_written(self, frame_index: int) -> None:
        """Mark a frame as written so it won't be written again.

        Args:
            frame_index: Source frame index.
        """
        self._written.add(frame_index)

    def can_release(self, frame_index: int, subtask_id: str) -> bool:
        """Check whether a frame can be released after a subtask completes.

        A frame is releasable only when ALL its consumers have completed.
        Call this after each subtask finishes; when it returns True,
        the frame data can be freed from memory.

        Args:
            frame_index: Source frame index.
            subtask_id: ID of the subtask that just completed.

        Returns:
            True if no remaining consumers reference this frame.
        """
        if frame_index in self._consumers:
            self._consumers[frame_index].discard(subtask_id)
            return len(self._consumers[frame_index]) == 0
        return True  # No consumers = safe to release

    def get_consumer_count(self, frame_index: int) -> int:
        """Return the number of active consumers for a frame.

        Args:
            frame_index: Source frame index.

        Returns:
            Number of subtasks still referencing this frame.
        """
        return len(self._consumers.get(frame_index, set()))
