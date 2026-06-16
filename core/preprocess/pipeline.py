"""Preprocessing pipeline for VFI-gui.

Combines scene detection and duplicate frame detection to decide
how each frame pair should be processed during interpolation.

The pipeline makes streaming-friendly decisions:
- INTERPOLATE: Normal interpolation, generate subtask
- SCENE_CUT: Scene boundary detected, write frame0 only
- DUPLICATE: Duplicate frame detected, write frame0 only
- LAST_FRAME: End of sequence, write directly
"""

from typing import Optional

import numpy as np

from core.types import (
    BackendType,
    FramePairAction,
    FramePairDecision,
    ProcessingConfig,
)
from core.preprocess.dup_detect import DuplicateDetector
from core.preprocess.scene_detect import SceneDetectorBase, SceneDetectorFactory


class PreprocessPipeline:
    """Preprocessing pipeline for frame pair decisions.

    Determines the processing action for each frame pair during streaming.
    The pipeline is designed for single-pass streaming: each call to decide()
    processes one frame pair without requiring knowledge of future frames.

    Decision priority:
        1. LAST_FRAME: frame1 is None (end of sequence)
        2. SCENE_CUT: Scene detector reports scene change
        3. DUPLICATE: Duplicate detector reports similar frames
        4. INTERPOLATE: Normal processing
    """

    def __init__(
        self,
        config: ProcessingConfig,
        backend_type: BackendType = BackendType.TORCH,
    ) -> None:
        """Initialize the preprocessing pipeline.

        Args:
            config: Processing configuration containing scene detection
                and duplicate detection settings
            backend_type: Backend type for neural scene detection
        """
        self._scene_detector: Optional[SceneDetectorBase] = (
            SceneDetectorFactory.create(config, backend_type)
        )
        self._dup_detector = DuplicateDetector(
            threshold=config.scene_detection.get("dup_threshold", 0.005)
        )

    def decide(
        self,
        frame0: np.ndarray,
        frame1: Optional[np.ndarray],
        frame0_index: int,
    ) -> FramePairDecision:
        """Make a preprocessing decision for a frame pair.

        This is a streaming-friendly single-pass decision: no pre-scanning
        or knowledge of future frames is required.

        Args:
            frame0: First frame [H, W, 3] RGB uint8 or float32
            frame1: Second frame [H, W, 3] RGB uint8 or float32,
                or None for the last frame
            frame0_index: Index of frame0 in the source sequence

        Returns:
            FramePairDecision indicating how to process this pair
        """
        # Case 1: Last frame (end of sequence)
        if frame1 is None:
            return FramePairDecision(
                frame0_index=frame0_index,
                frame1_index=-1,
                action=FramePairAction.LAST_FRAME,
                reason="last frame in sequence",
            )

        frame1_index = frame0_index + 1

        # Case 2: Scene cut detection
        if self._scene_detector is not None:
            if self._scene_detector.is_scene_cut(frame1):
                return FramePairDecision(
                    frame0_index=frame0_index,
                    frame1_index=frame1_index,
                    action=FramePairAction.SCENE_CUT,
                    reason="scene cut detected",
                )

        # Case 3: Duplicate frame detection
        if self._dup_detector.is_duplicate(frame0, frame1):
            return FramePairDecision(
                frame0_index=frame0_index,
                frame1_index=frame1_index,
                action=FramePairAction.DUPLICATE,
                reason="duplicate frame",
            )

        # Case 4: Normal interpolation
        return FramePairDecision(
            frame0_index=frame0_index,
            frame1_index=frame1_index,
            action=FramePairAction.INTERPOLATE,
            reason="",
        )

    def reset(self) -> None:
        """Reset pipeline state for a new video.

        Resets both scene detector and duplicate detector internal states.
        Should be called before processing a new video sequence.
        """
        if self._scene_detector is not None:
            self._scene_detector.reset()
        self._dup_detector.reset()

    @property
    def scene_detector(self) -> Optional[SceneDetectorBase]:
        """Get the scene detector instance."""
        return self._scene_detector

    @property
    def dup_detector(self) -> DuplicateDetector:
        """Get the duplicate detector instance."""
        return self._dup_detector


def run_pipeline(
    frames: list[np.ndarray],
    config: ProcessingConfig,
    backend_type: BackendType = BackendType.TORCH,
) -> list[FramePairDecision]:
    """Run the preprocessing pipeline on a sequence of frames.

    This is a convenience function for batch processing.
    For streaming use, create a PreprocessPipeline instance and call
    decide() for each frame pair.

    Args:
        frames: List of frames [H, W, 3] RGB
        config: Processing configuration
        backend_type: Backend type for neural scene detection

    Returns:
        List of FramePairDecision for each frame pair
    """
    pipeline = PreprocessPipeline(config, backend_type)
    decisions = []

    for i in range(len(frames)):
        frame0 = frames[i]
        frame1 = frames[i + 1] if i + 1 < len(frames) else None

        decision = pipeline.decide(frame0, frame1, i)
        decisions.append(decision)

    return decisions


def main() -> None:
    """CLI entry point for preprocessing pipeline."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline CLI for VFI-gui"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--scene-method",
        default="planestats",
        choices=["planestats", "neural"],
        help="Scene detection method",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.1,
        help="Scene cut threshold (default: 0.1)",
    )
    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=0.005,
        help="Duplicate detection threshold (default: 0.005)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    # Try to use PyAV for video reading
    try:
        import av

        container = av.open(args.input)
        stream = container.streams.video[0]

        # Create config
        config = ProcessingConfig(
            scene_detection={
                "enabled": True,
                "method": args.scene_method,
                "threshold": args.scene_threshold,
                "dup_threshold": args.dup_threshold,
            }
        )

        pipeline = PreprocessPipeline(config)
        decisions = []
        prev_frame = None
        prev_idx = 0
        frame_idx = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_idx % 100 == 0:
                    print(f"Processing frame {frame_idx}...", file=sys.stderr)

                # Convert to numpy RGB
                img = frame.to_ndarray(format="rgb24")

                if prev_frame is not None:
                    decision = pipeline.decide(prev_frame, img, prev_idx)
                    decisions.append(decision)

                prev_frame = img
                prev_idx = frame_idx
                frame_idx += 1

        # Handle last frame
        if prev_frame is not None:
            decision = pipeline.decide(prev_frame, None, prev_idx)
            decisions.append(decision)

        container.close()

        # Summarize results
        interpolate_count = sum(
            1 for d in decisions if d.action == FramePairAction.INTERPOLATE
        )
        scene_cut_count = sum(
            1 for d in decisions if d.action == FramePairAction.SCENE_CUT
        )
        duplicate_count = sum(
            1 for d in decisions if d.action == FramePairAction.DUPLICATE
        )
        last_frame_count = sum(
            1 for d in decisions if d.action == FramePairAction.LAST_FRAME
        )

        if args.json:
            result = {
                "input": args.input,
                "scene_method": args.scene_method,
                "scene_threshold": args.scene_threshold,
                "dup_threshold": args.dup_threshold,
                "total_frames": frame_idx,
                "decisions": [
                    {
                        "frame0": d.frame0_index,
                        "frame1": d.frame1_index,
                        "action": d.action.value,
                        "reason": d.reason,
                    }
                    for d in decisions
                ],
                "summary": {
                    "interpolate": interpolate_count,
                    "scene_cut": scene_cut_count,
                    "duplicate": duplicate_count,
                    "last_frame": last_frame_count,
                },
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Total frames: {frame_idx}")
            print(f"Decisions: {len(decisions)}")
            print(f"  Interpolate: {interpolate_count}")
            print(f"  Scene cut: {scene_cut_count}")
            print(f"  Duplicate: {duplicate_count}")
            print(f"  Last frame: {last_frame_count}")

    except ImportError:
        print(
            "Error: PyAV is required for CLI usage. "
            "Install with: pip install av",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
