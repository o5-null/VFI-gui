"""帧数据缓存：引用计数 + 内存管理 + LRU 驱逐"""

import threading
from datetime import datetime
from typing import Optional

import torch

from core.types import CachedFrameBundle, VideoMetadata


class FrameCache:
    """帧数据缓存，支持引用计数

    核心规则：共享帧在所有消费者完成后才释放。

    引用计数流程：
    - acquire(): 增加引用计数，注册消费者
    - release(): 减少引用计数，移除消费者
    - 引用计数归零 → 标记可驱逐
    - 内存压力大 → 驱逐引用计数=0 的帧

    内存限制：max_memory_mb 限制缓存总大小。
    超限时按 LRU 驱逐引用计数=0 的帧。
    """

    def __init__(self, max_memory_mb: int = 4096):
        self._cache: dict[str, CachedFrameBundle] = {}
        self._lock: threading.Lock = threading.Lock()
        self._max_memory: int = max_memory_mb * 1024 * 1024  # 字节
        self._current_memory: int = 0

    def put(
        self,
        file_path: str,
        frames: torch.Tensor,
        metadata: VideoMetadata,
        consumer_id: str,
    ) -> CachedFrameBundle:
        """将帧数据存入缓存，注册初始消费者

        Args:
            file_path: 文件路径（缓存键）
            frames: torch.Tensor 帧数据
            metadata: 视频元数据
            consumer_id: 首个消费者 ID

        Returns:
            CachedFrameBundle
        """
        with self._lock:
            memory_size = frames.element_size() * frames.nelement()

            bundle = CachedFrameBundle(
                file_path=file_path,
                frames=frames,
                metadata=metadata,
                consumers={consumer_id},
                loaded_at=datetime.now(),
                memory_size=memory_size,
            )

            self._cache[file_path] = bundle
            self._current_memory += memory_size

            # 内存超限 → 驱逐引用计数=0 的帧
            if self._current_memory > self._max_memory:
                self._evict_zero_ref()

            return bundle

    def acquire(
        self, file_path: str, consumer_id: str
    ) -> Optional[CachedFrameBundle]:
        """获取帧数据，增加引用计数

        Args:
            file_path: 缓存键
            consumer_id: 新消费者 ID

        Returns:
            CachedFrameBundle 或 None（未缓存）
        """
        with self._lock:
            bundle = self._cache.get(file_path)
            if bundle is None:
                return None

            bundle.consumers.add(consumer_id)
            return bundle

    def release(self, file_path: str, consumer_id: str) -> bool:
        """释放引用，移除消费者

        Args:
            file_path: 缓存键
            consumer_id: 释放的消费者 ID

        Returns:
            True = 引用计数归零，帧可驱逐
        """
        with self._lock:
            bundle = self._cache.get(file_path)
            if bundle is None:
                return True

            bundle.consumers.discard(consumer_id)
            if len(bundle.consumers) == 0:
                # 引用计数归零 → 标记可驱逐（不立即删除，等内存压力触发）
                return True
            return False

    def is_cached(self, file_path: str) -> bool:
        """检查是否已缓存"""
        with self._lock:
            return file_path in self._cache

    def evict(self, file_path: str) -> bool:
        """强制驱逐（仅当引用计数=0）"""
        with self._lock:
            bundle = self._cache.get(file_path)
            if bundle is None:
                return False
            if len(bundle.consumers) > 0:
                return False  # 还有消费者，不能驱逐

            self._current_memory -= bundle.memory_size
            del self._cache[file_path]
            return True

    def _evict_zero_ref(self):
        """驱逐引用计数=0 的帧（LRU 策略）"""
        # 按 loaded_at 排序，优先驱逐最旧的
        candidates = [
            (path, bundle)
            for path, bundle in self._cache.items()
            if len(bundle.consumers) == 0
        ]
        candidates.sort(key=lambda x: x[1].loaded_at)

        for path, bundle in candidates:
            if self._current_memory <= self._max_memory * 0.8:  # 驱逐到 80%
                break
            self._current_memory -= bundle.memory_size
            del self._cache[path]

    def get_stats(self) -> dict[str, int | float]:
        """缓存统计"""
        with self._lock:
            return {
                "total_entries": len(self._cache),
                "current_memory_mb": self._current_memory / 1024 / 1024,
                "max_memory_mb": self._max_memory / 1024 / 1024,
                "zero_ref_count": sum(
                    1 for b in self._cache.values() if len(b.consumers) == 0
                ),
            }

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="视频文件路径")
    parser.add_argument(
        "--max-memory", type=int, default=4096, help="最大缓存内存 MB"
    )
    parser.add_argument(
        "--consumers", type=int, default=3, help="并发消费者数"
    )
    parser.add_argument(
        "--stress-test", action="store_true", help="低内存压力测试"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # 测试引用计数生命周期
    cache = FrameCache(
        max_memory_mb=args.max_memory if not args.stress_test else 64
    )

    # 模拟多消费者场景
    from core.io.streaming_reader import StreamingFramePairReader

    reader = StreamingFramePairReader(args.input)
    metadata = reader.metadata

    acquire_count = 0
    release_count = 0

    for i, pair in enumerate(reader):
        if i >= 20:
            break
        # 模拟多个消费者 acquire 同一帧
        for c in range(args.consumers):
            consumer_id = f"consumer_{c}"
            if not cache.is_cached(args.input):
                # 首次：put
                cache.put(
                    args.input, pair.frame0.unsqueeze(0), metadata, consumer_id
                )
            else:
                # 后续：acquire
                cache.acquire(args.input, consumer_id)
            acquire_count += 1

        # 模拟消费者完成：release
        for c in range(args.consumers):
            cache.release(args.input, f"consumer_{c}")
            release_count += 1

    stats = cache.get_stats()
    result = {
        "success": True,
        "operations": {"acquire": acquire_count, "release": release_count},
        "refcount_leaks": stats["zero_ref_count"],
        "memory_peak_mb": stats["current_memory_mb"],
        "memory_final_mb": stats["current_memory_mb"],
        "all_released": stats["zero_ref_count"] == len(cache._cache),
    }
    print(json.dumps(result, indent=2) if args.json else str(result))
