"""断点续传：检查点管理器

This module provides checkpoint management for resumable video processing.
Checkpoints store progress metadata (NOT frame data) to allow resuming
interrupted processing from the last completed frame.

Usage:
    from core.checkpoint_manager import CheckpointManager
    from core.types import TaskCheckpoint

    manager = CheckpointManager()

    # Save checkpoint
    checkpoint = TaskCheckpoint(
        task_id="abc123",
        video_path="/path/to/video.mp4",
        output_path="/path/to/output.mkv",
        last_completed_frame=100,
        total_frames=500,
        multiplier=2,
    )
    manager.save(checkpoint)

    # Load checkpoint for resume
    checkpoint = manager.load("abc123")
    if checkpoint:
        resume_from_frame = checkpoint.last_completed_frame

    # Cleanup old checkpoints
    deleted_count = manager.cleanup_old(max_age_seconds=86400)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.types import TaskCheckpoint
from core.events import checkpoint_saved, checkpoint_loaded


class CheckpointManager:
    """Checkpoint manager for resumable video processing.

    Manages checkpoint files stored as JSON in {temp_dir}/checkpoints/.
    Checkpoints store progress metadata (NOT frame data) to allow resuming
    interrupted processing from the last completed frame.

    Attributes:
        temp_dir: Base directory for temporary files.
        checkpoints_dir: Directory containing checkpoint JSON files.

    Events:
        checkpoint_saved: Emitted when a checkpoint is saved.
        checkpoint_loaded: Emitted when a checkpoint is loaded for resume.

    Example:
        manager = CheckpointManager(temp_dir="temp")
        manager.save(checkpoint)
        checkpoint = manager.load(task_id)
    """

    def __init__(self, temp_dir: str = "temp") -> None:
        """Initialize the checkpoint manager.

        Creates the checkpoints directory if it does not exist.

        Args:
            temp_dir: Base directory for temporary files. Checkpoints are
                stored in {temp_dir}/checkpoints/ subdirectory.
        """
        self._temp_dir = Path(temp_dir).resolve()
        self._checkpoints_dir = self._temp_dir / "checkpoints"

        # Create checkpoints directory
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"CheckpointManager initialized: {self._checkpoints_dir}")

    def save(self, checkpoint: TaskCheckpoint) -> None:
        """Save a checkpoint to JSON file.

        Saves checkpoint as JSON at {temp_dir}/checkpoints/{task_id}.json.
        Emits checkpoint_saved event. Updates updated_at timestamp.

        Args:
            checkpoint: TaskCheckpoint instance to save. The updated_at
                field is automatically set to current time.
        """
        # Update timestamp
        checkpoint.updated_at = datetime.now()

        # Convert to dict and serialize
        data = _checkpoint_to_dict(checkpoint)
        checkpoint_path = self._checkpoints_dir / f"{checkpoint.task_id}.json"

        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Checkpoint saved: {checkpoint.task_id} "
                f"(frame {checkpoint.last_completed_frame}/{checkpoint.total_frames})"
            )

            # Emit event
            checkpoint_saved.send(
                self,
                task_id=checkpoint.task_id,
                last_completed_frame=checkpoint.last_completed_frame,
            )

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.task_id}: {e}")
            raise

    def load(self, task_id: str) -> Optional[TaskCheckpoint]:
        """Load a checkpoint from JSON file.

        Loads checkpoint from {temp_dir}/checkpoints/{task_id}.json.
        Emits checkpoint_loaded event. Returns None if not found.

        Args:
            task_id: Unique task identifier to load.

        Returns:
            TaskCheckpoint instance if found, None otherwise.
        """
        checkpoint_path = self._checkpoints_dir / f"{task_id}.json"

        if not checkpoint_path.exists():
            logger.debug(f"Checkpoint not found: {task_id}")
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            checkpoint = _dict_to_checkpoint(data)

            logger.info(
                f"Checkpoint loaded: {task_id} "
                f"(frame {checkpoint.last_completed_frame}/{checkpoint.total_frames})"
            )

            # Emit event
            checkpoint_loaded.send(
                self,
                task_id=checkpoint.task_id,
                last_completed_frame=checkpoint.last_completed_frame,
            )

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {task_id}: {e}")
            return None

    def delete(self, task_id: str) -> None:
        """Delete a checkpoint file.

        Deletes {temp_dir}/checkpoints/{task_id}.json. No error if not found.

        Args:
            task_id: Unique task identifier to delete.
        """
        checkpoint_path = self._checkpoints_dir / f"{task_id}.json"

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info(f"Checkpoint deleted: {task_id}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {task_id}: {e}")
        else:
            logger.debug(f"Checkpoint not found for deletion: {task_id}")

    def list_checkpoints(self) -> List[TaskCheckpoint]:
        """List all saved checkpoints.

        Returns all checkpoints sorted by updated_at descending (most recent
        first).

        Returns:
            List of TaskCheckpoint instances sorted by updated_at desc.
        """
        checkpoints: List[TaskCheckpoint] = []

        if not self._checkpoints_dir.exists():
            return checkpoints

        for checkpoint_file in self._checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoint = _dict_to_checkpoint(data)
                checkpoints.append(checkpoint)
            except Exception as e:
                logger.warning(
                    f"Failed to load checkpoint {checkpoint_file.name}: {e}"
                )
                continue

        # Sort by updated_at descending (most recent first)
        checkpoints.sort(key=lambda c: c.updated_at, reverse=True)

        logger.debug(f"Listed {len(checkpoints)} checkpoints")
        return checkpoints

    def validate_checkpoint(self, checkpoint: TaskCheckpoint) -> bool:
        """Validate a checkpoint for resume.

        Validates:
        1. Output file exists (checkpoint.output_path is a valid file path)
        2. Output file has at least checkpoint.last_completed_frame * multiplier
           frames (heuristic: file exists and size > 0)
        3. Video path still exists (checkpoint.video_path)

        Args:
            checkpoint: TaskCheckpoint instance to validate.

        Returns:
            True if checkpoint is valid for resume, False otherwise.
        """
        # 1. Check output file exists
        output_path = Path(checkpoint.output_path)
        if not output_path.exists():
            logger.warning(
                f"Checkpoint validation failed: output file not found "
                f"{checkpoint.output_path}"
            )
            return False

        if not output_path.is_file():
            logger.warning(
                f"Checkpoint validation failed: output path is not a file "
                f"{checkpoint.output_path}"
            )
            return False

        # 2. Check output file size > 0 (heuristic for frame count)
        if output_path.stat().st_size == 0:
            logger.warning(
                f"Checkpoint validation failed: output file is empty "
                f"{checkpoint.output_path}"
            )
            return False

        # 3. Check video path exists
        video_path = Path(checkpoint.video_path)
        if not video_path.exists():
            logger.warning(
                f"Checkpoint validation failed: video file not found "
                f"{checkpoint.video_path}"
            )
            return False

        logger.info(
            f"Checkpoint validated: {checkpoint.task_id} "
            f"(frame {checkpoint.last_completed_frame}/{checkpoint.total_frames})"
        )
        return True

    def cleanup_old(self, max_age_seconds: int = 86400) -> int:
        """Remove checkpoints older than max_age_seconds.

        Removes checkpoint files whose updated_at timestamp is older than
        max_age_seconds from current time. Returns count of deleted checkpoints.

        Args:
            max_age_seconds: Maximum age in seconds. Default is 86400 (24 hours).

        Returns:
            Number of checkpoints deleted.
        """
        deleted_count = 0
        cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)

        if not self._checkpoints_dir.exists():
            return deleted_count

        for checkpoint_file in self._checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Parse updated_at
                updated_at_str = data.get("updated_at", "")
                if not updated_at_str:
                    # No timestamp, skip
                    continue

                updated_at = datetime.fromisoformat(updated_at_str)

                # Check age
                if updated_at < cutoff_time:
                    checkpoint_file.unlink()
                    task_id = checkpoint_file.stem
                    logger.info(
                        f"Checkpoint cleaned up: {task_id} "
                        f"(age: {(datetime.now() - updated_at).total_seconds()}s)"
                    )
                    deleted_count += 1

            except Exception as e:
                logger.warning(
                    f"Failed to process checkpoint {checkpoint_file.name}: {e}"
                )
                continue

        logger.info(f"Cleanup completed: {deleted_count} checkpoints removed")
        return deleted_count


# ====================
# JSON Serialization Helpers
# ====================


def _checkpoint_to_dict(checkpoint: TaskCheckpoint) -> Dict[str, Any]:
    """Convert TaskCheckpoint to JSON-serializable dict.

    Handles datetime serialization with ISO format strings.

    Args:
        checkpoint: TaskCheckpoint instance to convert.

    Returns:
        Dict suitable for JSON serialization.
    """
    return {
        "task_id": checkpoint.task_id,
        "video_path": checkpoint.video_path,
        "output_path": checkpoint.output_path,
        "last_completed_frame": checkpoint.last_completed_frame,
        "total_frames": checkpoint.total_frames,
        "multiplier": checkpoint.multiplier,
        "codec": checkpoint.codec,
        "created_at": checkpoint.created_at.isoformat(),
        "updated_at": checkpoint.updated_at.isoformat(),
    }


def _dict_to_checkpoint(data: Dict[str, Any]) -> TaskCheckpoint:
    """Convert dict to TaskCheckpoint instance.

    Handles datetime deserialization from ISO format strings.

    Args:
        data: Dict loaded from JSON file.

    Returns:
        TaskCheckpoint instance.
    """
    # Parse datetime fields
    created_at = datetime.fromisoformat(data.get("created_at", ""))
    updated_at = datetime.fromisoformat(data.get("updated_at", ""))

    return TaskCheckpoint(
        task_id=data.get("task_id", ""),
        video_path=data.get("video_path", ""),
        output_path=data.get("output_path", ""),
        last_completed_frame=data.get("last_completed_frame", 0),
        total_frames=data.get("total_frames", 0),
        multiplier=data.get("multiplier", 2),
        codec=data.get("codec", ""),
        created_at=created_at,
        updated_at=updated_at,
    )


# ====================
# CLI Entry Point
# ====================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CheckpointManager CLI")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--verify", metavar="TASK_ID", help="Validate a checkpoint")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old checkpoints")
    parser.add_argument(
        "--max-age",
        type=int,
        default=86400,
        help="Max age in seconds for cleanup (default: 86400)",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--temp-dir",
        default="temp",
        help="Temp directory path (default: temp)",
    )
    args = parser.parse_args()

    manager = CheckpointManager(temp_dir=args.temp_dir)
    result: Dict[str, Any] = {}

    try:
        if args.list:
            checkpoints = manager.list_checkpoints()
            result = {
                "success": True,
                "count": len(checkpoints),
                "checkpoints": [
                    {
                        "task_id": c.task_id,
                        "video_path": c.video_path,
                        "output_path": c.output_path,
                        "last_completed_frame": c.last_completed_frame,
                        "total_frames": c.total_frames,
                        "multiplier": c.multiplier,
                        "updated_at": c.updated_at.isoformat(),
                    }
                    for c in checkpoints
                ],
            }

        elif args.verify:
            checkpoint = manager.load(args.verify)
            if checkpoint is None:
                result = {
                    "success": False,
                    "error": f"Checkpoint not found: {args.verify}",
                }
            else:
                valid = manager.validate_checkpoint(checkpoint)
                result = {
                    "success": True,
                    "task_id": checkpoint.task_id,
                    "valid": valid,
                    "last_completed_frame": checkpoint.last_completed_frame,
                    "total_frames": checkpoint.total_frames,
                    "video_path": checkpoint.video_path,
                    "output_path": checkpoint.output_path,
                }

        elif args.cleanup:
            deleted_count = manager.cleanup_old(max_age_seconds=args.max_age)
            result = {
                "success": True,
                "deleted_count": deleted_count,
                "max_age_seconds": args.max_age,
            }

        else:
            parser.print_help()
            result = {"success": False, "error": "No action specified"}

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif result.get("success"):
            if args.list:
                print(f"Found {result['count']} checkpoints:")
                for c in result["checkpoints"]:
                    print(
                        f"  - {c['task_id']}: "
                        f"frame {c['last_completed_frame']}/{c['total_frames']} "
                        f"(updated: {c['updated_at']})"
                    )
            elif args.verify:
                status = "VALID" if result["valid"] else "INVALID"
                print(
                    f"Checkpoint {result['task_id']}: {status} "
                    f"(frame {result['last_completed_frame']}/{result['total_frames']})"
                )
            elif args.cleanup:
                print(f"Deleted {result['deleted_count']} checkpoints")
        elif result.get("error"):
            print(f"Error: {result['error']}")

    except Exception as e:
        result = {"success": False, "error": str(e)}
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Error: {e}")
        raise