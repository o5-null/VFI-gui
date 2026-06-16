"""Tests for core/events.py — Blinker signal system.

Verifies that signals can be connected, emitted, and disconnected.
Uses blinker's test utilities where appropriate.
"""

from __future__ import annotations

from core import events


class TestCoreEvents:
    """Core Blinker signal registration and emission."""

    def test_signals_exist(self):
        """All documented signals are registered via module-level aliases."""
        # blinker.Namespace has no .signals() method, so we verify by
        # checking that module-level signal aliases send and receive
        received: list = []

        def handler(sender, **kwargs):
            received.append(kwargs.get("signal_name"))

        for sig, name in [
            (events.engines_updated, "engines-updated"),
            (events.models_updated, "models-updated"),
            (events.processing_state_changed, "processing-state-changed"),
            (events.processing_progress, "processing-progress"),
            (events.processing_finished, "processing-finished"),
            (events.task_started, "task-started"),
            (events.task_progress, "task-progress"),
            (events.task_finished, "task-finished"),
            (events.task_failed, "task-failed"),
            (events.task_cancelled, "task-cancelled"),
            (events.checkpoint_saved, "checkpoint-saved"),
            (events.checkpoint_loaded, "checkpoint-loaded"),
        ]:
            sig.connect(handler)
            sig.send(self, signal_name=name)
            sig.disconnect(handler)

        assert len(received) == 12
        assert "engines-updated" in received
        assert "checkpoint-loaded" in received

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
