"""Benchmark and device detection module for VFI-gui.

This module provides:
- Device detection (CUDA, Intel XPU, CPU)
- Performance benchmarking
- Hardware capability analysis
"""

from .device_detector import DeviceDetector, DeviceInfo, DeviceType
from .benchmark_runner import BenchmarkRunner, BenchmarkResult, BenchmarkConfig

__all__ = [
    "DeviceDetector",
    "DeviceInfo",
    "DeviceType",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkConfig",
]
