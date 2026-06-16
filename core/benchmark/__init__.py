"""Benchmark and device detection module for VFI-gui.

This module provides:
- Device detection (CUDA, Intel XPU, CPU)
- Performance benchmarking with real model inference
- Hardware capability analysis
- Dual-mode benchmarking (SINGLE / MULTI_RESOLUTION)
- Hardware monitoring per iteration
- PyTorch Profiler integration (opt-in)
- JSON output for results
"""

from .device_detector import DeviceDetector, DeviceInfo, DeviceType
from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkConfig,
    BenchmarkMode,
    HardwareStats,
    ResolutionResult,
    ProfilerConfig,
    ProfilerResult,
    OpStats,
)

__all__ = [
    "DeviceDetector",
    "DeviceInfo",
    "DeviceType",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkMode",
    "HardwareStats",
    "ResolutionResult",
    "ProfilerConfig",
    "ProfilerResult",
    "OpStats",
]
