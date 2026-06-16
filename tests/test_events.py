"""Tests for core/events.py — Blinker signal system.

Verifies that signals can be connected, emitted, and disconnected.
Uses blinker's test utilities where appropriate.
"""

from __future__ import annotations

from core import events


class TestCoreEvents:
    """Core Blinker signal registration and emission."""

    def test_signals_exist(self):
        """All documented signals are registered in the namespace."""
        signal_names = {s.name for s in events.events.signals()}
        assert "engines-updated" in signal_names
        assert "models-updated" in signal_names
        assert "processing-state-changed" in signal_names
        assert "processing-progress" in signal_names
        assert "processing-finished" in signal_names
        assert "task-started" in signal_names
        assert "task-progress" in signal_names
        assert "task-finished" in signal_names
        assert "task-failed" in signal_names
        assert "task-cancelled" in signal_names
        assert "checkpoint-saved" in signal_names
        assert "checkpoint-loaded" in signal_names

    def test_signal_send_and_receive(self):
        """Signal emission reaches connected receiver."""
        received: list = []

        def handler(sender, **kwargs):
            received.append((sender, kwargs))

        events.task_started.connect(handler)
        events.task_started.send(self, task_id="abc", video_path="/test.mp4")
        events.task_started.disconnect(handler)

        assert len(received) == 1
        sender, kwargs = received[0]
        assert kwargs["task_id"] == "abc"
        assert kwargs["video_path"] == "/test.mp4"

    def test_signal_multiple_receivers(self):
        """Multiple receivers all get the signal."""
        results: list = []

        def handler_a(sender, **kwargs):
            results.append("a")

        def handler_b(sender, **kwargs):
            results.append("b")

        events.task_finished.connect(handler_a)
        events.task_finished.connect(handler_b)
        events.task_finished.send(self, task_id="x", output_path="/out.mp4")
        events.task_finished.disconnect(handler_a)
        events.task_finished.disconnect(handler_b)

        assert "a" in results
        assert "b" in results

    def test_signal_disconnect(self):
        """Disconnected handler no longer receives signals."""
        received: list = []

        def handler(sender, **kwargs):
            received.append(True)

        events.task_progress.connect(handler)
        events.task_progress.disconnect(handler)
        events.task_progress.send(self, frame=1, total=100, fps=30.0)

        assert len(received) == 0

    def test_signal_no_receivers(self):
        """Sending a signal with no receivers does not error."""
        # This signal has no default receivers — should be a no-op
        events.processing_finished.send(self, success=True, message="done")
        # If we get here without exception, the test passes

    def test_module_level_signals(self):
        """Module-level signal aliases match namespace signals."""
        assert events.engines_updated is events.events.signal("engines-updated")
        assert events.models_updated is events.events.signal("models-updated")
        assert events.task_started is events.events.signal("task-started")
        assert events.task_progress is events.events.signal("task-progress")
