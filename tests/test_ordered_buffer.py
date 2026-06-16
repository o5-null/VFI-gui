"""Tests for core/io/ordered_buffer.py — ordered result buffer.

Tests the key invariant: parallel results submitted out of order
are written in sequential frame order.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from core.io.ordered_buffer import OrderedResultBuffer
from core.types import ProcessedFrameData


def _make_frame_data(source_frame_idx: int) -> ProcessedFrameData:
    """Helper to create test frame data with a specific source index."""
    return ProcessedFrameData(
        data=torch.rand(3, 64, 64),
        source_frame_idx=source_frame_idx,
    )


class TestOrderedResultBuffer:
    """OrderedResultBuffer write-order guarantees."""

    def test_sequential_submit(self):
        """Frames submitted in order are written immediately."""
        mock_writer = MagicMock()
        buf = OrderedResultBuffer(mock_writer)

        buf.submit(0, _make_frame_data(0))
        assert buf.get_frames_written() == 1
        mock_writer.write_frame.assert_called_once()

        buf.submit(1, _make_frame_data(1))
        assert buf.get_frames_written() == 2

    def test_out_of_order_submit(self):
        """Frames submitted out of order are buffered until predecessors arrive."""
        mock_writer = MagicMock()
        buf = OrderedResultBuffer(mock_writer)

        # Submit frame 2 first — should be buffered
        buf.submit(2, _make_frame_data(2))
        assert buf.get_frames_written() == 0
        assert buf.get_buffer_size() == 1

        # Submit frame 1 — still waiting for frame 0
        buf.submit(1, _make_frame_data(1))
        assert buf.get_frames_written() == 0
        assert buf.get_buffer_size() == 2

        # Submit frame 0 — triggers flush of 0, 1, 2
        buf.submit(0, _make_frame_data(0))
        assert buf.get_frames_written() == 3
        assert buf.get_buffer_size() == 0

    def test_partial_flush(self):
        """Only consecutive frames from next_write are flushed."""
        mock_writer = MagicMock()
        buf = OrderedResultBuffer(mock_writer)

        buf.submit(0, _make_frame_data(0))  # written immediately (next=0)
        buf.submit(2, _make_frame_data(2))  # buffered (next=1, missing 1)
        buf.submit(3, _make_frame_data(3))  # buffered (next=1, missing 1,2)

        assert buf.get_frames_written() == 1  # only frame 0 written
        assert buf.get_buffer_size() == 2  # frames 2 and 3 buffered

        buf.submit(1, _make_frame_data(1))  # triggers flush of 1,2,3
        assert buf.get_frames_written() == 4
        assert buf.get_buffer_size() == 0

    def test_flush_all(self):
        """flush_all force-writes all buffered frames in order."""
        mock_writer = MagicMock()
        buf = OrderedResultBuffer(mock_writer)

        buf.submit(3, _make_frame_data(3))
        buf.submit(1, _make_frame_data(1))
        buf.submit(2, _make_frame_data(2))

        assert buf.get_frames_written() == 0

        buf.flush_all()
        assert buf.get_frames_written() == 3
        assert buf.get_buffer_size() == 0

        # flush_all works with frame_index-based tracking
        assert buf.get_frames_written() == 3

    def test_empty_flush_all(self):
        """flush_all on empty buffer is a no-op."""
        mock_writer = MagicMock()
        buf = OrderedResultBuffer(mock_writer)
        buf.flush_all()
        assert buf.get_frames_written() == 0

    def test_thread_safety(self):
        """Buffer handles concurrent submit calls.

        This test is intentionally simple — full thread-safety verification
        requires concurrency stress testing with many threads.
        """
        import concurrent.futures

        mock_writer = MagicMock()
        buf = OrderedResultBuffer(mock_writer)

        # Submit frames 0-9 in reverse order, concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(buf.submit, i, _make_frame_data(i))
                for i in range(9, -1, -1)
            ]
            concurrent.futures.wait(futures)

        # All frames should be written
        assert buf.get_frames_written() == 10
