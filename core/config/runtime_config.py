"""Runtime configuration for VFI-gui.

This module manages the runtime environment selection (CUDA/XPU/CPU)
and persists the user's preference across sessions.
"""

from typing import Optional
from core.config.base_config import BaseConfig


DEFAULT_RUNTIME_SETTINGS = {
    "selected_runtime": "",  # "cuda", "xpu", "cpu", or "" for auto-detect
    "auto_detect": True,  # Auto-detect GPU on startup
    "last_detected_runtime": "",  # Last auto-detected runtime
}


class RuntimeConfig(BaseConfig):
    """Configuration for runtime environment selection.

    Manages GPU type detection results and user's runtime preference.
    The runtime determines which virtual environment is activated:
    - cuda: NVIDIA GPU runtime (runtime/cuda/)
    - xpu: Intel GPU runtime (runtime/xpu/)
    - cpu: CPU-only fallback
    """

    def _load_defaults(self) -> None:
        """Load default runtime settings."""
        self._settings = DEFAULT_RUNTIME_SETTINGS.copy()

    def get_selected_runtime(self) -> str:
        """Get user-selected runtime type.

        Returns:
            Runtime type string: "cuda", "xpu", "cpu", or "" for auto-detect
        """
        return self._settings.get("selected_runtime", "")

    def set_selected_runtime(self, runtime: str) -> None:
        """Set user's runtime preference.

        Args:
            runtime: Runtime type ("cuda", "xpu", "cpu") or "" for auto-detect
        """
        self._settings["selected_runtime"] = runtime
        self.save()

    def get_auto_detect(self) -> bool:
        """Check if auto-detect is enabled.

        Returns:
            True if GPU should be auto-detected on startup
        """
        return self._settings.get("auto_detect", True)

    def set_auto_detect(self, enabled: bool) -> None:
        """Enable or disable auto-detect.

        Args:
            enabled: Whether to auto-detect GPU on startup
        """
        self._settings["auto_detect"] = enabled
        self.save()

    def get_last_detected_runtime(self) -> str:
        """Get the last auto-detected runtime.

        Returns:
            Last detected runtime type string
        """
        return self._settings.get("last_detected_runtime", "")

    def set_last_detected_runtime(self, runtime: str) -> None:
        """Record the last auto-detected runtime.

        Args:
            runtime: Detected runtime type
        """
        self._settings["last_detected_runtime"] = runtime
        self.save()

    def get_effective_runtime(self) -> str:
        """Get the effective runtime to use.

        If user has selected a specific runtime, use that.
        Otherwise, use the last detected runtime.

        Returns:
            Effective runtime type string
        """
        selected = self.get_selected_runtime()
        if selected:
            return selected
        return self.get_last_detected_runtime() or "cpu"
