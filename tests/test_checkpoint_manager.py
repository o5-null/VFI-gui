"""Tests for core/checkpoint_manager.py — checkpoint save/load/cleanup."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from core.checkpoint_manager import CheckpointManager
from core.types import TaskCheckpoint


class TestCheckpointManager:
    """CheckpointManager persists and loads checkpoints correctly."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="vfi_checkpoint_")
        self.manager = CheckpointManager(temp_dir=self.temp_dir)

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_checkpoint(self, task_id: str, last_frame: int = 0) -> TaskCheckpoint:
        return TaskCheckpoint(
            task_id=task_id,
            video_path=f"/test/{task_id}.mp4",
            output_path=f"/output/{task_id}.mkv",
            last_completed_frame=last_frame,
            total_frames=100,
            multiplier=2,
        )

    def test_save_and_load(self):
        """Saved checkpoint can be loaded back with same fields."""
        cp = self._make_checkpoint("task_001", last_frame=50)
        self.manager.save(cp)

        loaded = self.manager.load("task_001")
        assert loaded is not None
        assert loaded.task_id == "task_001"
        assert loaded.last_completed_frame == 50
        assert loaded.total_frames == 100
        assert loaded.multiplier == 2
        assert loaded.video_path == "/test/task_001.mp4"
        assert loaded.output_path == "/output/task_001.mkv"

    def test_load_nonexistent(self):
        """Loading a non-existent checkpoint returns None."""
        loaded = self.manager.load("nonexistent_task")
        assert loaded is None

    def test_delete_checkpoint(self):
        """After delete(), checkpoint is removed."""
        cp = self._make_checkpoint("task_del")
        self.manager.save(cp)
        self.manager.delete("task_del")
        loaded = self.manager.load("task_del")
        assert loaded is None

    def test_multiple_checkpoints(self):
        """Multiple checkpoints can coexist and are separately retrievable."""
        for i in range(5):
            self.manager.save(self._make_checkpoint(f"task_{i}", last_frame=i * 20))

        for i in range(5):
            loaded = self.manager.load(f"task_{i}")
            assert loaded is not None
            assert loaded.last_completed_frame == i * 20

    def test_update_checkpoint(self):
        """Saving a checkpoint with same task_id overwrites."""
        cp = self._make_checkpoint("task_update", last_frame=10)
        self.manager.save(cp)

        cp2 = self._make_checkpoint("task_update", last_frame=50)
        self.manager.save(cp2)

        loaded = self.manager.load("task_update")
        assert loaded is not None
        assert loaded.last_completed_frame == 50

    def test_checkpoint_file_exists(self):
        """Checkpoint is persisted as a JSON file on disk."""
        cp = self._make_checkpoint("task_disk")
        self.manager.save(cp)

        checkpoint_path = Path(self.temp_dir) / "checkpoints" / "task_disk.json"
        assert checkpoint_path.exists()

        with open(checkpoint_path) as f:
            data = json.load(f)
        assert data["task_id"] == "task_disk"
        assert data["last_completed_frame"] == 0

    def test_cleanup_old(self):
        """cleanup_old removes checkpoints older than max_age."""
        from datetime import datetime, timedelta

        # Save a checkpoint
        self.manager.save(self._make_checkpoint("task_old"))
        checkpoint_path = Path(self.temp_dir) / "checkpoints" / "task_old.json"

        # Manually set its mtime to be very old
        old_time = (datetime.now() - timedelta(days=30)).timestamp()
        os.utime(str(checkpoint_path), (old_time, old_time))

        # Cleanup with max_age = 1 hour
        deleted = self.manager.cleanup_old(max_age_seconds=3600)
        assert deleted == 1
        assert self.manager.load("task_old") is None
