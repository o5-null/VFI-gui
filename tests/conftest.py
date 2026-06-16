"""Shared fixtures and utilities for VFI-gui tests.

Fixtures:
    - null_logger: loguru logger that discards output
    - sample_video_metadata: basic VideoMetadata for IO tests
    - sample_frame: 3×H×W uint8 torch tensor for cache/lifecycle tests
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
import torch

from core.types import ColorSpaceInfo, VideoMetadata


# ====================
# Logging Fixtures
# ====================


@pytest.fixture(autouse=True)
def null_logger():
    """Replace loguru logger with a no-op during tests.

    Prevents test output pollution from loguru handlers.
    Applied automatically to all tests via autouse=True.
    """
    import logging

    # Disable loguru by removing all handlers
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    yield
    # Restore default handler after test
    loguru_logger.add(lambda msg: None, level="ERROR")


# ====================
# Data Fixtures
# ====================


@pytest.fixture
def sample_video_metadata() -> VideoMetadata:
    """Basic VideoMetadata for IO tests.

    Returns a 1920×1080 @ 23.976fps video metadata fixture.
    """
    return VideoMetadata(
        width=1920,
        height=1080,
        fps=24000.0 / 1001.0,
        total_frames=100,
        duration=100 * (1001.0 / 24000.0),
        color_space=ColorSpaceInfo(
            matrix="bt709",
            transfer="sdr",
            primaries="bt709",
            range="limited",
            bit_depth=8,
        ),
    )


@pytest.fixture
def sample_frame() -> torch.Tensor:
    """3×64×64 uint8 torch tensor for cache/lifecycle tests."""
    return torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)


@pytest.fixture
def sample_float_frame() -> torch.Tensor:
    """3×64×64 float32 torch tensor for validator tests."""
    return torch.rand(3, 64, 64, dtype=torch.float32)


@pytest.fixture
def sample_numpy_frame() -> np.ndarray:
    """64×64×3 uint8 numpy array for dup_detect tests."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
