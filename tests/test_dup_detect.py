"""Tests for core/preprocess/dup_detect.py — duplicate frame detection."""

from __future__ import annotations

import numpy as np

from core.preprocess.dup_detect import DuplicateDetector, detect_duplicates


class TestDuplicateDetector:
    """DuplicateDetector unit tests."""

    def setup_method(self):
        self.detector = DuplicateDetector(threshold=0.01)

    def test_identical_frames_are_duplicates(self):
        """Two identical uint8 frames are duplicates."""
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        assert self.detector.is_duplicate(frame, frame) is True

    def test_completely_different_frames(self):
        """Two very different frames are NOT duplicates."""
        frame0 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame1 = np.full((64, 64, 3), 255, dtype=np.uint8)
        assert self.detector.is_duplicate(frame0, frame1) is False

    def test_float32_frames(self):
        """Float32 input frames work correctly."""
        frame0 = np.random.rand(64, 64, 3).astype(np.float32)
        frame1 = frame0.copy()
        assert self.detector.is_duplicate(frame0, frame1) is True

    def test_near_identical_uint8(self):
        """Minor 1-pixel differences may still be duplicates if below threshold."""
        frame0 = np.full((64, 64, 3), 128, dtype=np.uint8)
        frame1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        frame1[0, 0, 0] = 129  # 1/255 = 0.0039 < threshold 0.01
        assert self.detector.is_duplicate(frame0, frame1) is True

    def test_threshold_behavior(self):
        """Frame pairs near threshold boundary behave correctly."""
        frame0 = np.full((16, 16, 3), 100, dtype=np.uint8)

        # diff = 1/255 ≈ 0.0039, should be < 0.005 (default threshold)
        frame1 = np.full((16, 16, 3), 101, dtype=np.uint8)
        default_detector = DuplicateDetector()  # threshold=0.005
        assert default_detector.is_duplicate(frame0, frame1) is True

        # Diff = 100/255 ≈ 0.392 > 0.005
        frame2 = np.full((16, 16, 3), 200, dtype=np.uint8)
        assert default_detector.is_duplicate(frame0, frame2) is False

    def test_threshold_setter(self):
        """Threshold can be updated after construction."""
        detector = DuplicateDetector(threshold=0.5)
        frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
        frame1 = np.ones((16, 16, 3), dtype=np.uint8) * 255
        assert detector.is_duplicate(frame0, frame1) is True  # diff=1.0 > 0.5

        detector.threshold = 1.5
        assert detector.is_duplicate(frame0, frame1) is True  # diff=1.0 < 1.5

    def test_reset(self):
        """reset() does nothing but doesn't error."""
        self.detector.reset()  # should not raise

    def test_threshold_property(self):
        """threshold getter returns current value."""
        detector = DuplicateDetector(threshold=0.1)
        assert detector.threshold == 0.1
        detector.threshold = 0.5
        assert detector.threshold == 0.5


class TestDetectDuplicates:
    """detect_duplicates helper function."""

    def test_no_duplicates(self):
        """All different frames."""
        frames = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        duplicates = detect_duplicates(frames, threshold=0.01)
        assert len(duplicates) == 0

    def test_all_duplicates(self):
        """All identical frames."""
        frame = np.full((32, 32, 3), 100, dtype=np.uint8)
        frames = [frame.copy() for _ in range(5)]
        duplicates = detect_duplicates(frames, threshold=0.01)
        assert len(duplicates) == 4  # frame[1]..frame[4] are duplicates of previous

    def test_some_duplicates(self):
        """Pattern: diff, same, same, diff, same."""
        frames = [
            np.full((16, 16, 3), 0, dtype=np.uint8),
            np.full((16, 16, 3), 0, dtype=np.uint8),   # dup of 0
            np.full((16, 16, 3), 0, dtype=np.uint8),   # dup of 1
            np.full((16, 16, 3), 200, dtype=np.uint8),  # diff
            np.full((16, 16, 3), 200, dtype=np.uint8),  # dup of 3
        ]
        duplicates = detect_duplicates(frames, threshold=0.01)
        assert duplicates == [1, 2, 4]

    def test_empty_list(self):
        """Empty frame list."""
        assert detect_duplicates([]) == []

    def test_single_frame(self):
        """Single frame — no pairs to compare."""
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        assert detect_duplicates([frame]) == []
