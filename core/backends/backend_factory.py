"""Inference strategy selector for backend optimization.

This module provides intelligent selection of inference strategies based on
model capabilities and hardware characteristics. The selector determines the
optimal parallelism approach for maximizing throughput.

Selection priority (highest to lowest):
    1. BATCH       - If model supports batch inference (best throughput)
    2. CUDA_STREAMS - If NVIDIA CUDA GPU available (stream parallelism)
    3. MULTI_MODEL - Fallback for CPU/XPU (model instance parallelism)
    4. SERIAL      - Default for single-threaded environments

Architecture constraints:
    - No torch import at module level (lazy import for runtime detection)
    - No cv2/PIL dependencies (pure strategy logic)
    - BackendFactory exists in base_backend.py (not duplicated here)
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.types import InferenceStrategy, BackendConfig


class InferenceStrategySelector:
    """Selects optimal inference strategy based on model and hardware.

    The selector evaluates model capabilities and GPU characteristics to
    determine the best parallelism strategy for maximizing inference throughput.

    Usage:
        # Direct selection with known GPU info
        gpu_info = {"is_cuda": True, "is_xpu": False, "device_name": "RTX 4090",
                    "vram_mb": 24000}
        strategy = InferenceStrategySelector.select(model, gpu_info)

        # Auto-detect from config
        strategy = InferenceStrategySelector.select_for_config(config)
    """

    @classmethod
    def select(
        cls,
        model: Any,
        gpu_info: Dict[str, Any],
    ) -> InferenceStrategy:
        """Select inference strategy based on model and hardware.

        Selection logic (priority order):
            1. BATCH - if model has interpolate_batch AND supports_batch_inference
            2. CUDA_STREAMS - if gpu_info['is_cuda'] == True
            3. MULTI_MODEL - fallback for CPU/XPU environments

        Args:
            model: Model instance to evaluate. Expected attributes:
                - interpolate_batch: method for batch processing
                - supports_batch_inference: bool flag for batch capability
            gpu_info: GPU information dict with keys:
                - is_cuda: bool - True if NVIDIA CUDA GPU available
                - is_xpu: bool - True if Intel XPU GPU available
                - device_name: str - GPU device name (e.g., "RTX 4090")
                - vram_mb: int - VRAM capacity in megabytes

        Returns:
            Selected InferenceStrategy enum value.

        Example:
            >>> gpu_info = {"is_cuda": True, "is_xpu": False,
            ...             "device_name": "RTX 3080", "vram_mb": 10000}
            >>> strategy = InferenceStrategySelector.select(rife_model, gpu_info)
            >>> strategy
            InferenceStrategy.CUDA_STREAMS
        """
        # Priority 1: Batch inference (best throughput for supported models)
        if cls._check_batch_capability(model):
            return InferenceStrategy.BATCH

        # Priority 2: CUDA streams (stream parallelism for NVIDIA GPUs)
        if gpu_info.get("is_cuda", False):
            return InferenceStrategy.CUDA_STREAMS

        # Priority 3: Multi-model (model instance parallelism for CPU/XPU)
        return InferenceStrategy.MULTI_MODEL

    @classmethod
    def select_for_config(cls, config: BackendConfig) -> InferenceStrategy:
        """Auto-detect and select strategy from BackendConfig.

        Performs GPU detection by checking torch availability, then selects
        the optimal strategy based on detected hardware and default model.

        Args:
            config: BackendConfig instance. Uses config.get_device() to resolve
                    device type for GPU detection.

        Returns:
            Selected InferenceStrategy based on auto-detected GPU info.

        Note:
            This method performs lazy torch import for GPU detection.
            Detection logic:
                - CUDA: torch.cuda.is_available()
                - XPU: torch.xpu.is_available() (Intel extension)

        Example:
            >>> config = BackendConfig(device="auto")
            >>> strategy = InferenceStrategySelector.select_for_config(config)
        """
        gpu_info = cls._detect_gpu_info(config)
        # Note: model is None for config-based selection, defaults to hardware
        # strategy since batch capability cannot be determined without model
        return cls.select(None, gpu_info)

    @staticmethod
    def _check_batch_capability(model: Any) -> bool:
        """Check if model supports batch inference.

        Args:
            model: Model instance to check.

        Returns:
            True if model has both interpolate_batch method and
            supports_batch_inference attribute set to True.
        """
        if model is None:
            return False

        # Check for interpolate_batch method
        has_batch_method = hasattr(model, "interpolate_batch") and callable(
            model.interpolate_batch
        )

        # Check for supports_batch_inference flag
        has_batch_flag = getattr(model, "supports_batch_inference", False)

        return has_batch_method and has_batch_flag

    @staticmethod
    def _detect_gpu_info(config: BackendConfig) -> Dict[str, Any]:
        """Detect GPU information from config and torch.

        Lazy imports torch to avoid module-level dependency issues.
        Uses DeviceManager for authoritative device resolution.

        Args:
            config: BackendConfig to resolve device from.

        Returns:
            GPU info dict with is_cuda, is_xpu, device_name, vram_mb.
        """
        # Lazy torch import (not at module level)
        import torch

        gpu_info: Dict[str, Any] = {
            "is_cuda": False,
            "is_xpu": False,
            "device_name": "CPU",
            "vram_mb": 0,
        }

        # Resolve device string via config
        device_str = config.get_device()

        # Check CUDA availability
        if device_str.startswith("cuda") or device_str == "auto":
            if torch.cuda.is_available():
                gpu_info["is_cuda"] = True
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
                # Get VRAM in MB (total memory)
                gpu_info["vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (
                    1024 * 1024
                )
                return gpu_info

        # Check XPU availability (Intel GPU)
        if device_str.startswith("xpu") or device_str == "auto":
            # Intel XPU extension check (may not exist in standard torch)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                gpu_info["is_xpu"] = True
                gpu_info["device_name"] = torch.xpu.get_device_name(0)
                # XPU memory query (if available)
                try:
                    gpu_info["vram_mb"] = torch.xpu.get_device_properties(
                        0
                    ).total_memory // (1024 * 1024)
                except (AttributeError, RuntimeError):
                    gpu_info["vram_mb"] = 0
                return gpu_info

        # CPU fallback
        return gpu_info