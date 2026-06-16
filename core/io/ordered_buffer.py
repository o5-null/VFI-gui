"""Ordered result buffer: parallel inference, sequential write.

Inference may complete out of order (e.g. SubTask_3 finishes before SubTask_1),
but video output must be in frame order.

Principle: maintain a next_write pointer. When consecutive frames arrive,
write them immediately. Non-consecutive frames are buffered until their
predecessors arrive.
"""

import threading
from typing import Dict

from core.types import ProcessedFrameData
from core.io.frame_writer import FrameWriter


class OrderedResultBuffer:
    """Parallel inference results written in frame order.

    Thread-safe: submit() can be called from any thread.
    The internal lock ensures that flush operations are atomic.

    Attributes:
        _writer: FrameWriter instance for actual frame output.
        _buffer: Temporary storage for out-of-order results.
        _next_write: Next expected frame index to write.
        _frames_written: Total frames successfully written.
    """

    def __init__(self, writer: FrameWriter) -> None:
        """Initialize the ordered result buffer.

        Args:
            writer: FrameWriter instance to write frames to.
        """
        self._writer = writer
        self._buffer: Dict[int, ProcessedFrameData] = {}
        self._next_write = 0
        self._lock = threading.Lock()
        self._frames_written = 0

    def submit(self, frame_index: int, data: ProcessedFrameData) -> None:
        """Submit an inference result (callable from any thread).

        If frame_index == next_write, write immediately and flush
        any consecutive buffered frames. Otherwise, buffer the result
        until its predecessors arrive.

        Args:
            frame_index: Output frame index (must be sequential in final output).
            data: Processed frame data to write.
        """
        with self._lock:
            self._buffer[frame_index] = data
            self._flush()

    def _flush(self) -> None:
        """Write all consecutive buffered results starting from next_write.

        Must be called while holding self._lock.
        """
        while self._next_write in self._buffer:
            data = self._buffer.pop(self._next_write)
            self._writer.write_frame(data)
            self._next_write += 1
            self._frames_written += 1

    def get_frames_written(self) -> int:
        """Return the total number of frames written so far."""
        return self._frames_written

    def get_buffer_size(self) -> int:
        """Return the number of frames currently buffered (waiting for predecessors)."""
        with self._lock:
            return len(self._buffer)

    def flush_all(self) -> None:
        """Force-write all buffered frames in order (call at task end).

        This handles any remaining frames that might be out of order
        due to incomplete inference or early termination.
        """
        with self._lock:
            sorted_keys = sorted(self._buffer.keys())
            for key in sorted_keys:
                data = self._buffer.pop(key)
                self._writer.write_frame(data)
                self._frames_written += 1
            self._buffer.clear()
