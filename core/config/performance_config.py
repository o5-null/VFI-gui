"""Performance-specific configuration.

This configuration domain handles performance-related settings including:
- Multi-threaded inference configuration
- Cache clearing settings
- Cancellation timeout settings
"""

from typing import Any, Dict

from core.config.base_config import BaseConfig


DEFAULT_PERFORMANCE_SETTINGS: Dict[str, Any] = {
    "inference": {
        "enable_threading": True,
        "threads": 2,
        "task_queue_size": 100,
    },
    "cache": {
        "clear_interval": 10,
    },
    "cancellation": {
        "force_terminate": False,
        "timeout": 5,
    },
}


class PerformanceConfig(BaseConfig):
    """Configuration for performance-related settings.

    Manages inference threading, cache clearing, and cancellation behavior.
    """

    def _load_defaults(self) -> None:
        """Load default performance settings."""
        self._settings = DEFAULT_PERFORMANCE_SETTINGS.copy()

    # Inference settings
    def get_enable_threading(self) -> bool:
        """Get whether multi-threaded inference is enabled."""
        inference = self._settings.get("inference", {})
        return inference.get("enable_threading", True)

    def set_enable_threading(self, enabled: bool) -> None:
        """Set whether multi-threaded inference is enabled."""
        if "inference" not in self._settings:
            self._settings["inference"] = {}
        self._settings["inference"]["enable_threading"] = enabled
        self.save()

    def get_inference_threads(self) -> int:
        """Get number of inference threads."""
        inference = self._settings.get("inference", {})
        return inference.get("threads", 2)

    def set_inference_threads(self, threads: int) -> None:
        """Set number of inference threads."""
        if "inference" not in self._settings:
            self._settings["inference"] = {}
        self._settings["inference"]["threads"] = threads
        self.save()

    def get_task_queue_size(self) -> int:
        """Get task queue size."""
        inference = self._settings.get("inference", {})
        return inference.get("task_queue_size", 100)

    def set_task_queue_size(self, size: int) -> None:
        """Set task queue size."""
        if "inference" not in self._settings:
            self._settings["inference"] = {}
        self._settings["inference"]["task_queue_size"] = size
        self.save()

    # Cache settings
    def get_cache_clear_interval(self) -> int:
        """Get cache clearing interval (frames)."""
        cache = self._settings.get("cache", {})
        return cache.get("clear_interval", 10)

    def set_cache_clear_interval(self, interval: int) -> None:
        """Set cache clearing interval (frames)."""
        if "cache" not in self._settings:
            self._settings["cache"] = {}
        self._settings["cache"]["clear_interval"] = interval
        self.save()

    # Cancellation settings
    def get_force_terminate_on_cancel(self) -> bool:
        """Get whether to force terminate on cancel."""
        cancellation = self._settings.get("cancellation", {})
        return cancellation.get("force_terminate", False)

    def set_force_terminate_on_cancel(self, force: bool) -> None:
        """Set whether to force terminate on cancel."""
        if "cancellation" not in self._settings:
            self._settings["cancellation"] = {}
        self._settings["cancellation"]["force_terminate"] = force
        self.save()

    def get_cancel_timeout(self) -> int:
        """Get cancellation timeout (seconds)."""
        cancellation = self._settings.get("cancellation", {})
        return cancellation.get("timeout", 5)

    def set_cancel_timeout(self, timeout: int) -> None:
        """Set cancellation timeout (seconds)."""
        if "cancellation" not in self._settings:
            self._settings["cancellation"] = {}
        self._settings["cancellation"]["timeout"] = timeout
        self.save()

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all performance settings."""
        return {
            "inference": {
                "enable_threading": self.get_enable_threading(),
                "threads": self.get_inference_threads(),
                "task_queue_size": self.get_task_queue_size(),
            },
            "cache": {
                "clear_interval": self.get_cache_clear_interval(),
            },
            "cancellation": {
                "force_terminate": self.get_force_terminate_on_cancel(),
                "timeout": self.get_cancel_timeout(),
            },
        }