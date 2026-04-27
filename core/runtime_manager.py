"""Runtime Manager for VFI-gui.

This module manages runtime environment detection, selection, and activation.
It detects available GPU types (CUDA, Intel XPU) and activates the appropriate
virtual environment from the runtime/ directory.

Usage:
    from core.runtime_manager import runtime_manager

    # Auto-detect and activate runtime
    runtime_manager.auto_select_runtime()

    # Or manually select a runtime
    runtime_manager.select_runtime("cuda")

    # Get current runtime info
    info = runtime_manager.get_runtime_info()
"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger

from core.paths import paths
from core.config.runtime_config import RuntimeConfig


class RuntimeType(Enum):
    """Supported runtime types."""
    CUDA = "cuda"
    XPU = "xpu"
    CPU = "cpu"

    @classmethod
    def from_string(cls, value: str) -> "RuntimeType":
        """Convert string to RuntimeType.

        Args:
            value: String value ("cuda", "xpu", "cpu")

        Returns:
            RuntimeType enum value

        Raises:
            ValueError: If value is not a valid runtime type
        """
        value = value.lower()
        for rt in cls:
            if rt.value == value:
                return rt
        raise ValueError(f"Invalid runtime type: {value}")


@dataclass
class RuntimeInfo:
    """Information about a runtime environment."""
    runtime_type: RuntimeType
    name: str
    path: Path
    python_version: str
    is_available: bool
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    error_message: Optional[str] = None


class RuntimeManager:
    """Manages runtime environment detection and activation.

    This class handles:
    - GPU type detection (CUDA, Intel XPU)
    - Virtual environment validation
    - Runtime activation (sys.path, environment variables)
    - User preference persistence

    Runtime Directory Structure:
        runtime/
        ├── cuda/          # NVIDIA GPU environment
        │   ├── Lib/
        │   ├── Scripts/
        │   └── pyvenv.cfg
        └── xpu/           # Intel GPU environment
            ├── Lib/
            ├── Scripts/
            └── pyvenv.cfg
    """

    def __init__(self):
        """Initialize the runtime manager."""
        self._runtime_dir = paths.app_dir.parent / "runtime"
        self._config = RuntimeConfig(str(paths.config_dir / "runtime.json"))
        self._current_runtime: Optional[RuntimeType] = None
        self._runtime_cache: Dict[RuntimeType, RuntimeInfo] = {}

    @property
    def runtime_dir(self) -> Path:
        """Get the runtime directory path."""
        return self._runtime_dir

    @property
    def current_runtime(self) -> Optional[RuntimeType]:
        """Get the currently active runtime type."""
        return self._current_runtime

    def get_runtime_path(self, runtime_type: RuntimeType) -> Path:
        """Get the virtual environment path for a runtime type.

        Args:
            runtime_type: Type of runtime

        Returns:
            Path to the virtual environment
        """
        if runtime_type == RuntimeType.CPU:
            # CPU uses system Python, no venv path
            return Path(sys.executable).parent.parent
        return self._runtime_dir / runtime_type.value

    def detect_gpu(self) -> Tuple[RuntimeType, Optional[str], int]:
        """Detect available GPU type.

        Returns:
            Tuple of (runtime_type, gpu_name, gpu_count)
        """
        # Try CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                logger.info(f"Detected NVIDIA CUDA GPU: {gpu_name} (x{gpu_count})")
                return RuntimeType.CUDA, gpu_name, gpu_count
        except ImportError:
            logger.debug("PyTorch not available for CUDA detection")
        except Exception as e:
            logger.debug(f"CUDA detection failed: {e}")

        # Try Intel XPU
        try:
            import torch
            # Safe check for xpu attribute (may not exist in non-XPU builds)
            if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
                gpu_count = torch.xpu.device_count()
                gpu_name = torch.xpu.get_device_name(0) if gpu_count > 0 else "Intel GPU"
                logger.info(f"Detected Intel XPU GPU: {gpu_name} (x{gpu_count})")
                return RuntimeType.XPU, gpu_name, gpu_count
        except ImportError:
            logger.debug("PyTorch not available for XPU detection")
        except Exception as e:
            logger.debug(f"XPU detection failed: {e}")

        # Fallback to CPU
        logger.info("No GPU detected, using CPU mode")
        return RuntimeType.CPU, None, 0

    def check_runtime_available(self, runtime_type: RuntimeType) -> RuntimeInfo:
        """Check if a runtime environment is available and valid.

        Args:
            runtime_type: Type of runtime to check

        Returns:
            RuntimeInfo with availability status
        """
        if runtime_type in self._runtime_cache:
            return self._runtime_cache[runtime_type]

        if runtime_type == RuntimeType.CPU:
            info = RuntimeInfo(
                runtime_type=RuntimeType.CPU,
                name="CPU",
                path=Path(sys.executable).parent.parent,
                python_version=self._get_python_version(Path(sys.executable).parent.parent),
                is_available=True,
            )
            self._runtime_cache[runtime_type] = info
            return info

        venv_path = self.get_runtime_path(runtime_type)

        # Check if directory exists
        if not venv_path.exists():
            info = RuntimeInfo(
                runtime_type=runtime_type,
                name=runtime_type.value.upper(),
                path=venv_path,
                python_version="",
                is_available=False,
                error_message=f"Runtime directory not found: {venv_path}",
            )
            self._runtime_cache[runtime_type] = info
            return info

        # Check for pyvenv.cfg
        pyvenv_cfg = venv_path / "pyvenv.cfg"
        if not pyvenv_cfg.exists():
            info = RuntimeInfo(
                runtime_type=runtime_type,
                name=runtime_type.value.upper(),
                path=venv_path,
                python_version="",
                is_available=False,
                error_message=f"Not a valid virtual environment: {venv_path}",
            )
            self._runtime_cache[runtime_type] = info
            return info

        # Check for Scripts directory (Windows)
        scripts_dir = venv_path / "Scripts"
        if not scripts_dir.exists():
            info = RuntimeInfo(
                runtime_type=runtime_type,
                name=runtime_type.value.upper(),
                path=venv_path,
                python_version="",
                is_available=False,
                error_message=f"Scripts directory not found: {scripts_dir}",
            )
            self._runtime_cache[runtime_type] = info
            return info

        # Get Python version from pyvenv.cfg
        python_version = self._get_python_version(venv_path)

        info = RuntimeInfo(
            runtime_type=runtime_type,
            name=runtime_type.value.upper(),
            path=venv_path,
            python_version=python_version,
            is_available=True,
        )
        self._runtime_cache[runtime_type] = info
        return info

    def _get_python_version(self, venv_path: Path) -> str:
        """Extract Python version from pyvenv.cfg.

        Args:
            venv_path: Path to virtual environment

        Returns:
            Python version string or empty string
        """
        pyvenv_cfg = venv_path / "pyvenv.cfg"
        if not pyvenv_cfg.exists():
            return ""

        try:
            content = pyvenv_cfg.read_text(encoding="utf-8")
            for line in content.splitlines():
                if line.startswith("version_info"):
                    # Format: version_info = 3.12.10
                    return line.split("=", 1)[1].strip()
        except Exception as e:
            logger.debug(f"Failed to read pyvenv.cfg: {e}")

        return ""

    def get_available_runtimes(self) -> List[RuntimeInfo]:
        """Get list of all available runtimes.

        Returns:
            List of RuntimeInfo for all runtime types
        """
        runtimes = []
        for rt in [RuntimeType.CUDA, RuntimeType.XPU, RuntimeType.CPU]:
            runtimes.append(self.check_runtime_available(rt))
        return runtimes

    def auto_select_runtime(self) -> RuntimeType:
        """Auto-detect GPU and select the best available runtime.

        This method:
        1. Detects GPU type (CUDA, XPU, or CPU)
        2. Checks if the corresponding runtime is available
        3. Falls back to CPU if GPU runtime not available
        4. Persists the detected runtime

        Returns:
            The selected runtime type
        """
        # Check user preference
        selected = self._config.get_selected_runtime()
        if selected and not self._config.get_auto_detect():
            try:
                runtime_type = RuntimeType.from_string(selected)
                info = self.check_runtime_available(runtime_type)
                if info.is_available:
                    self._activate_runtime(runtime_type)
                    return runtime_type
                logger.warning(f"Selected runtime {selected} not available, auto-detecting")
            except ValueError:
                logger.warning(f"Invalid runtime type in config: {selected}")

        # Auto-detect GPU
        detected_type, gpu_name, gpu_count = self.detect_gpu()

        # Check if detected runtime is available
        info = self.check_runtime_available(detected_type)
        if info.is_available:
            info.gpu_name = gpu_name
            info.gpu_count = gpu_count
            self._activate_runtime(detected_type)

            # Persist detected runtime
            self._config.set_last_detected_runtime(detected_type.value)

            return detected_type

        # Fall back to CPU
        logger.warning(f"Detected {detected_type.value} but runtime not available, using CPU")
        self._activate_runtime(RuntimeType.CPU)
        self._config.set_last_detected_runtime(RuntimeType.CPU.value)

        return RuntimeType.CPU

    def select_runtime(self, runtime_type: RuntimeType) -> bool:
        """Manually select and activate a runtime.

        Args:
            runtime_type: Runtime type to activate

        Returns:
            True if activation succeeded
        """
        info = self.check_runtime_available(runtime_type)
        if not info.is_available:
            logger.error(f"Runtime {runtime_type.value} not available: {info.error_message}")
            return False

        self._activate_runtime(runtime_type)
        self._config.set_selected_runtime(runtime_type.value)
        self._config.set_auto_detect(False)

        return True

    def _activate_runtime(self, runtime_type: RuntimeType) -> None:
        """Activate a runtime environment.

        This sets up sys.path and environment variables for the selected runtime.

        Args:
            runtime_type: Runtime type to activate
        """
        if runtime_type == RuntimeType.CPU:
            logger.info("Using CPU runtime (system Python)")
            self._current_runtime = runtime_type
            return

        venv_path = self.get_runtime_path(runtime_type)

        # Add runtime's site-packages to sys.path
        site_packages = venv_path / "Lib" / "site-packages"
        if site_packages.exists() and str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
            logger.debug(f"Added to sys.path: {site_packages}")

        # Set VIRTUAL_ENV environment variable
        os.environ["VIRTUAL_ENV"] = str(venv_path)
        logger.debug(f"Set VIRTUAL_ENV={venv_path}")

        self._current_runtime = runtime_type
        logger.info(f"Activated {runtime_type.value.upper()} runtime: {venv_path}")

    def get_runtime_info(self) -> Dict:
        """Get information about current runtime state.

        Returns:
            Dictionary with runtime information
        """
        available = self.get_available_runtimes()

        return {
            "current": self._current_runtime.value if self._current_runtime else None,
            "detected": self._config.get_last_detected_runtime(),
            "selected": self._config.get_selected_runtime(),
            "auto_detect": self._config.get_auto_detect(),
            "available_runtimes": [
                {
                    "type": info.runtime_type.value,
                    "name": info.name,
                    "path": str(info.path),
                    "python_version": info.python_version,
                    "is_available": info.is_available,
                    "gpu_name": info.gpu_name,
                    "gpu_count": info.gpu_count,
                    "error": info.error_message,
                }
                for info in available
            ],
        }

    def reset_to_auto_detect(self) -> None:
        """Reset to auto-detect mode."""
        self._config.set_selected_runtime("")
        self._config.set_auto_detect(True)
        logger.info("Reset to auto-detect mode")


# Global singleton instance
runtime_manager = RuntimeManager()
