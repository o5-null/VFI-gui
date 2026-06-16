"""Duplicate frame detection for VFI-gui.

Detects and skips completely identical or extremely similar frame pairs
during interpolation. This avoids wasting computation on frames that
would produce no visible difference.

The detection uses normalized SAD (Sum of Absolute Differences),
same metric as PlaneStatsSceneDetector but with inverse threshold logic:
- Scene detection: diff > threshold (looking for large changes)
- Duplicate detection: diff < threshold (looking for small changes)
"""

from typing import List

import numpy as np


class DuplicateDetector:
    """Duplicate frame detector using normalized SAD.

    Detects frame pairs with PlaneStatsDiff below threshold as duplicates.
    Uses the same metric as PlaneStatsSceneDetector but inverted logic.

    Threshold selection guide:
        - 0.001 (old design): Too strict, compression artifacts cause false negatives
        - 0.005 (recommended): Tolerates minor compression artifacts
        - 0.01: More tolerant, suitable for high-compression video
    """

    def __init__(self, threshold: float = 0.005) -> None:
        """Initialize the duplicate detector.

        Args:
            threshold: Duplicate detection threshold (0.0-1.0).
                Frame pairs with normalized difference below this value
                are considered duplicates. Default 0.005.
        """
        self._threshold = threshold

    def is_duplicate(self, frame0: np.ndarray, frame1: np.ndarray) -> bool:
        """Check if two frames are duplicates.

        Automatically handles both uint8 and float32 input formats.

        Args:
            frame0: First frame [H, W, 3] RGB uint8 or float32
            frame1: Second frame [H, W, 3] RGB uint8 or float32

        Returns:
            True if frames are duplicates (should skip interpolation)
        """
        # Handle uint8 input: normalize to [0, 1] range
        if frame0.dtype == np.uint8:
            diff = np.mean(np.abs(
                frame0.astype(np.float64) - frame1.astype(np.float64)
            )) / 255.0
        else:
            # Already float, compute difference directly
            diff = np.mean(np.abs(frame0 - frame1))

        return bool(diff < self._threshold)

    def reset(self) -> None:
        """Reset internal state.

        This detector is stateless, but the method is provided for API
        consistency with SceneDetectorBase.
        """
        pass

    @property
    def threshold(self) -> float:
        """Get the current threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the threshold.

        Args:
            value: New threshold value (0.0-1.0)
        """
        self._threshold = value


def detect_duplicates(
    frames: List[np.ndarray],
    threshold: float = 0.005,
) -> List[int]:
    """Detect duplicate frames in a sequence.

    Args:
        frames: List of frames [H, W, 3] RGB
        threshold: Duplicate detection threshold

    Returns:
        List of frame indices that are duplicates of the previous frame
    """
    detector = DuplicateDetector(threshold=threshold)
    duplicates = []

    for i in range(1, len(frames)):
        if detector.is_duplicate(frames[i - 1], frames[i]):
            duplicates.append(i)

    return duplicates


def main() -> None:
    """CLI entry point for duplicate frame detection."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Duplicate frame detection CLI for VFI-gui"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--threshold",
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

        detector = DuplicateDetector(threshold=args.threshold)
        duplicates = []
        prev_frame = None
        frame_idx = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_idx % 100 == 0:
                    print(f"Processing frame {frame_idx}...", file=sys.stderr)

                # Convert to numpy RGB
                img = frame.to_ndarray(format="rgb24")

                if prev_frame is not None:
                    if detector.is_duplicate(prev_frame, img):
                        duplicates.append(frame_idx)

                prev_frame = img
                frame_idx += 1

        container.close()

        if args.json:
            result = {
                "input": args.input,
                "threshold": args.threshold,
                "total_frames": frame_idx,
                "duplicates": duplicates,
                "duplicate_count": len(duplicates),
                "duplicate_ratio": len(duplicates) / frame_idx if frame_idx > 0 else 0,
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Total frames: {frame_idx}")
            print(f"Duplicate frames detected: {len(duplicates)}")
            if duplicates:
                ratio = len(duplicates) / frame_idx if frame_idx > 0 else 0
                print(f"Duplicate ratio: {ratio:.2%}")
                if len(duplicates) <= 50:
                    print(f"Duplicate frame indices: {duplicates}")

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
