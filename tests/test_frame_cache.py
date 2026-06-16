"""Tests for core/io/frame_cache.py — reference-counted frame cache."""

from __future__ import annotations

import torch

from core.io.frame_cache import FrameCache
from core.types import CachedFrameBundle, ColorSpaceInfo, VideoMetadata


def _make_metadata() -> VideoMetadata:
    """Create minimal VideoMetadata fixture."""
    return VideoMetadata(width=64, height=64, fps=30.0, total_frames=100)


class TestFrameCache:
    """FrameCache refcount and eviction invariants."""

    def setup_method(self):
        self.cache = FrameCache(max_memory_mb=1024)
        self.metadata = _make_metadata()

    def test_put_and_is_cached(self):
        """After put(), item is cached."""
        frames = torch.rand(2, 3, 64, 64)
        self.cache.put("test.mp4", frames, self.metadata, "consumer_1")
        assert self.cache.is_cached("test.mp4") is True

    def test_acquire_existing(self):
        """Acquiring an existing item returns the bundle."""
        frames = torch.rand(2, 3, 64, 64)
        self.cache.put("test.mp4", frames, self.metadata, "consumer_1")
        bundle = self.cache.acquire("test.mp4", "consumer_2")
        assert bundle is not None
        assert "consumer_2" in bundle.consumers

    def test_acquire_missing(self):
        """Acquiring a non-cached item returns None."""
        bundle = self.cache.acquire("nonexistent.mp4", "consumer_1")
        assert bundle is None

    def test_release_to_zero(self):
        """Release last consumer — frames become evictable."""
        frames = torch.rand(2, 3, 64, 64)
        self.cache.put("test.mp4", frames, self.metadata, "consumer_1")
        result = self.cache.release("test.mp4", "consumer_1")
        assert result is True  # zero refs

    def test_release_nonzero(self):
        """Release one of multiple consumers — frames NOT evictable yet."""
        frames = torch.rand(2, 3, 64, 64)
        self.cache.put("test.mp4", frames, self.metadata, "consumer_1")
        self.cache.acquire("test.mp4", "consumer_2")
        result = self.cache.release("test.mp4", "consumer_1")
        assert result is False  # consumer_2 still active

    def test_evict_with_consumers(self):
        """Evict on an item with active consumers fails."""
        frames = torch.rand(2, 3, 64, 64)
        self.cache.put("test.mp4", frames, self.metadata, "consumer_1")
        assert self.cache.evict("test.mp4") is False

    def test_evict_zero_ref(self):
        """Evict on an item with zero consumers succeeds."""
        frames = torch.rand(2, 3, 64, 64)
        self.cache.put("test.mp4", frames, self.metadata, "consumer_1")
        self.cache.release("test.mp4", "consumer_1")
        assert self.cache.evict("test.mp4") is True
        assert self.cache.is_cached("test.mp4") is False

    def test_evict_nonexistent(self):
        """Evict on a nonexistent key returns False."""
        assert self.cache.evict("ghost.mp4") is False

    def test_get_stats(self):
        """get_stats() returns expected keys."""
        stats = self.cache.get_stats()
        assert "total_entries" in stats
        assert "current_memory_mb" in stats
        assert "max_memory_mb" in stats
        assert "zero_ref_count" in stats

    def test_clear(self):
        """clear() removes all cached items."""
        frames = torch.rand(2, 3, 64, 64)
        self.cache.put("a.mp4", frames, self.metadata, "c1")
        self.cache.put("b.mp4", frames, self.metadata, "c1")
        self.cache.clear()
        assert self.cache.is_cached("a.mp4") is False
        assert self.cache.is_cached("b.mp4") is False
        assert self.cache.get_stats()["total_entries"] == 0

    def test_evict_with_memory_pressure(self):
        """When memory limit is exceeded, zero-ref items are evicted."""
        cache = FrameCache(max_memory_mb=1)  # 1 MB limit

        frames = torch.rand(1, 3, 256, 256)  # ~0.75 MB
        cache.put("a.mp4", frames, self.metadata, "c1")
        cache.release("a.mp4", "c1")

        # Second put should trigger eviction due to memory pressure
        cache.put("b.mp4", frames, self.metadata, "c2")
        # a.mp4 should be evicted, b.mp4 should be cached
        assert cache.is_cached("a.mp4") is False
        assert cache.is_cached("b.mp4") is True
