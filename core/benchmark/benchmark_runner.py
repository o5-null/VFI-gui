"""Benchmark runner for VFI-gui.

This module provides performance benchmarking capabilities:
- Frame interpolation speed testing
- Memory usage monitoring
- Multi-resolution performance testing
- Device comparison
"""

import time
import gc
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from pathlib import Path

from loguru import logger

from .device_detector import DeviceDetector, DeviceInfo, DeviceType


class BenchmarkType(Enum):
    """Types of benchmarks available."""
    INFERENCE_SPEED = "inference_speed"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Test parameters
    warmup_iterations: int = 3
    test_iterations: int = 10
    
    # Resolution tests (width, height)
    test_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (640, 480),   # SD
        (1280, 720),  # HD
        (1920, 1080), # FHD
        (2560, 1440), # QHD
        (3840, 2160), # 4K
    ])
    
    # Model configuration
    model_type: str = "rife"
    checkpoint_name: str = "rife49.pth"
    multiplier: int = 2
    dtype: str = "float16"
    
    # Device selection
    device_id: int = 0
    device_type: Optional[DeviceType] = None
    
    # Memory monitoring
    monitor_memory: bool = True
    clear_cache_between_runs: bool = True
    
    # Custom test data
    use_random_frames: bool = True
    custom_frame_path: Optional[str] = None


@dataclass
class ResolutionResult:
    """Benchmark result for a specific resolution."""
    resolution: Tuple[int, int]
    width: int = 0
    height: int = 0
    
    # Timing results (milliseconds)
    warmup_time_ms: float = 0.0
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    std_inference_time_ms: float = 0.0
    
    # Throughput
    fps: float = 0.0
    
    # Memory results (MB)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.width == 0 and self.resolution:
            self.width, self.height = self.resolution


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    # Test configuration
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    device_info: Optional[DeviceInfo] = None
    
    # Test metadata
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Results by resolution
    resolution_results: List[ResolutionResult] = field(default_factory=list)
    
    # Summary
    best_resolution: Optional[Tuple[int, int]] = None
    best_fps: float = 0.0
    recommended_batch_size: int = 1
    
    # System info during test
    system_load: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_duration(self) -> float:
        """Get total benchmark duration."""
        return self.end_time - self.start_time
    
    @property
    def success_count(self) -> int:
        """Get number of successful tests."""
        return sum(1 for r in self.resolution_results if r.success)
    
    @property
    def failed_count(self) -> int:
        """Get number of failed tests."""
        return sum(1 for r in self.resolution_results if not r.success)
    
    def get_result_for_resolution(self, width: int, height: int) -> Optional[ResolutionResult]:
        """Get result for a specific resolution."""
        for result in self.resolution_results:
            if result.width == width and result.height == height:
                return result
        return None


class BenchmarkRunner:
    """Runs performance benchmarks for VFI models.
    
    This class provides comprehensive benchmarking:
    - Tests multiple resolutions
    - Measures inference speed and throughput
    - Monitors memory usage
    - Provides recommendations
    
    Usage:
        runner = BenchmarkRunner()
        
        # Configure benchmark
        config = BenchmarkConfig(
            test_iterations=10,
            test_resolutions=[(1920, 1080), (3840, 2160)],
        )
        
        # Run benchmark
        result = runner.run(config)
        
        # Get results
        for res_result in result.resolution_results:
            print(f"{res_result.width}x{res_result.height}: {res_result.fps:.1f} FPS")
    """
    
    def __init__(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        """Initialize the benchmark runner.
        
        Args:
            progress_callback: Optional callback for progress updates.
                              Called with (message, progress_percent)
        """
        self.progress_callback = progress_callback
        self._device_detector = DeviceDetector()
        self._cancelled = False
    
    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        logger.info(f"Benchmark: {message} ({progress:.1f}%)")
    
    def cancel(self):
        """Cancel the current benchmark."""
        self._cancelled = True
    
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a benchmark with the given configuration.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            Benchmark results
        """
        self._cancelled = False
        result = BenchmarkResult(
            config=config,
            start_time=time.time(),
        )
        
        # Get device info
        if config.device_type:
            devices = self._device_detector.get_devices_by_type(config.device_type)
            if devices and config.device_id < len(devices):
                result.device_info = devices[config.device_id]
            else:
                result.device_info = self._device_detector.get_best_device()
        else:
            result.device_info = self._device_detector.get_best_device()
        
        self._report_progress("Initializing benchmark", 5.0)
        
        # Check if we can run benchmarks
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not available for benchmarking")
            result.end_time = time.time()
            return result
        
        # Run benchmarks for each resolution
        total_resolutions = len(config.test_resolutions)
        for idx, resolution in enumerate(config.test_resolutions):
            if self._cancelled:
                logger.info("Benchmark cancelled by user")
                break
            
            progress = 10.0 + (idx / total_resolutions) * 80.0
            self._report_progress(f"Testing {resolution[0]}x{resolution[1]}", progress)
            
            res_result = self._benchmark_resolution(config, resolution, result.device_info)
            result.resolution_results.append(res_result)
            
            if res_result.success and res_result.fps > result.best_fps:
                result.best_fps = res_result.fps
                result.best_resolution = resolution
        
        # Calculate recommendations
        self._report_progress("Calculating recommendations", 95.0)
        result.recommended_batch_size = self._calculate_recommendations(result)
        
        result.end_time = time.time()
        self._report_progress("Benchmark complete", 100.0)
        
        return result
    
    def _benchmark_resolution(
        self,
        config: BenchmarkConfig,
        resolution: Tuple[int, int],
        device: DeviceInfo,
    ) -> ResolutionResult:
        """Benchmark a specific resolution.
        
        Args:
            config: Benchmark configuration
            resolution: Resolution to test (width, height)
            device: Device to test on
            
        Returns:
            Benchmark result for the resolution
        """
        result = ResolutionResult(resolution=resolution)
        width, height = resolution
        
        try:
            import torch
            
            # Select device
            if device.device_type == DeviceType.CUDA:
                device_str = f"cuda:{device.device_id}"
                torch.cuda.set_device(device.device_id)
            elif device.device_type == DeviceType.XPU:
                device_str = f"xpu:{device.device_id}"
            else:
                device_str = "cpu"
            
            device_torch = torch.device(device_str)
            
            # Determine dtype
            dtype = torch.float32
            if config.dtype == "float16":
                dtype = torch.float16
            elif config.dtype == "bfloat16":
                dtype = torch.bfloat16
            
            # Create synthetic test frames
            # Shape: [batch=1, channels=3, height, width]
            frame0 = torch.randn(1, 3, height, width, dtype=dtype, device=device_torch)
            frame1 = torch.randn(1, 3, height, width, dtype=dtype, device=device_torch)
            timestep = torch.tensor([0.5], dtype=dtype, device=device_torch)
            
            # Synchronize before benchmark
            if device.device_type == DeviceType.CUDA:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device.device_id)
            
            # Warmup
            for _ in range(config.warmup_iterations):
                _ = self._simulate_inference(frame0, frame1, timestep)
                if device.device_type == DeviceType.CUDA:
                    torch.cuda.synchronize()
            
            # Clear cache if configured
            if config.clear_cache_between_runs and device.device_type == DeviceType.CUDA:
                torch.cuda.empty_cache()
            
            # Actual benchmark
            times = []
            memory_readings = []
            
            for i in range(config.test_iterations):
                if self._cancelled:
                    break
                
                if device.device_type == DeviceType.CUDA:
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                _ = self._simulate_inference(frame0, frame1, timestep)
                
                if device.device_type == DeviceType.CUDA:
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
                
                # Record memory
                if config.monitor_memory and device.device_type == DeviceType.CUDA:
                    memory_mb = torch.cuda.memory_allocated(device.device_id) / 1024 / 1024
                    memory_readings.append(memory_mb)
            
            # Calculate statistics
            if times:
                result.avg_inference_time_ms = sum(times) / len(times)
                result.min_inference_time_ms = min(times)
                result.max_inference_time_ms = max(times)
                
                # Calculate standard deviation
                if len(times) > 1:
                    mean = result.avg_inference_time_ms
                    variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
                    result.std_inference_time_ms = variance ** 0.5
                
                # Calculate FPS
                result.fps = 1000.0 / result.avg_inference_time_ms if result.avg_inference_time_ms > 0 else 0
            
            # Memory statistics
            if memory_readings:
                result.avg_memory_mb = sum(memory_readings) / len(memory_readings)
            
            if device.device_type == DeviceType.CUDA:
                result.peak_memory_mb = torch.cuda.max_memory_allocated(device.device_id) / 1024 / 1024
            
            result.success = True
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"Out of memory at {width}x{height}: {e}")
            result.success = False
            result.error_message = "Out of memory"
            
        except Exception as e:
            logger.error(f"Benchmark failed for {width}x{height}: {e}")
            result.success = False
            result.error_message = str(e)
        
        return result
    
    def _simulate_inference(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Simulate frame interpolation inference.
        
        This is a simplified simulation that performs operations similar
        to actual VFI inference without requiring model weights.
        
        Args:
            frame0: First frame
            frame1: Second frame
            timestep: Interpolation timestep
            
        Returns:
            Simulated output frame
        """
        # Simulate VFI operations: feature extraction + warping + blending
        
        # Simple convolution-like operation (feature extraction simulation)
        features0 = frame0 * 0.5 + torch.randn_like(frame0) * 0.01
        features1 = frame1 * 0.5 + torch.randn_like(frame1) * 0.01
        
        # Warping simulation
        warped = features0 * (1 - timestep.view(-1, 1, 1, 1)) + features1 * timestep.view(-1, 1, 1, 1)
        
        # Blending simulation
        output = warped * 0.9 + frame0 * 0.05 + frame1 * 0.05
        
        return output
    
    def _calculate_recommendations(self, result: BenchmarkResult) -> int:
        """Calculate recommended settings based on benchmark results.
        
        Args:
            result: Benchmark results
            
        Returns:
            Recommended batch size
        """
        if not result.resolution_results:
            return 1
        
        # Find the highest resolution that achieves >10 FPS
        recommended_batch = 1
        
        for res_result in reversed(result.resolution_results):
            if res_result.success and res_result.fps >= 10.0:
                # Based on FPS, suggest batch size
                if res_result.fps >= 30:
                    recommended_batch = 4
                elif res_result.fps >= 20:
                    recommended_batch = 2
                else:
                    recommended_batch = 1
                break
        
        return recommended_batch
    
    def quick_test(self, device: Optional[DeviceInfo] = None) -> Dict[str, Any]:
        """Run a quick performance test.
        
        Args:
            device: Device to test on. Uses best device if None.
            
        Returns:
            Quick test results
        """
        if device is None:
            device = self._device_detector.get_best_device()
        
        config = BenchmarkConfig(
            warmup_iterations=1,
            test_iterations=3,
            test_resolutions=[(1280, 720), (1920, 1080)],
            monitor_memory=True,
        )
        
        result = self.run(config)
        
        return {
            "device": device.display_name,
            "device_type": device.device_type.value,
            "memory_gb": device.memory_gb,
            "test_results": [
                {
                    "resolution": f"{r.width}x{r.height}",
                    "fps": round(r.fps, 1),
                    "avg_time_ms": round(r.avg_inference_time_ms, 2),
                    "memory_mb": round(r.peak_memory_mb, 0),
                }
                for r in result.resolution_results
            ],
            "best_resolution": result.best_resolution,
            "recommended_batch_size": result.recommended_batch_size,
        }


if __name__ == "__main__":
    # Test benchmark runner
    import json
    
    print("=" * 60)
    print("VFI Benchmark Test")
    print("=" * 60)
    
    def progress_callback(message: str, progress: float):
        print(f"[{progress:5.1f}%] {message}")
    
    runner = BenchmarkRunner(progress_callback)
    
    # Quick test
    print("\nRunning quick test...")
    quick_result = runner.quick_test()
    print("\nQuick Test Results:")
    print(json.dumps(quick_result, indent=2))
    
    # Full benchmark
    print("\n" + "=" * 60)
    print("Running full benchmark...")
    print("=" * 60)
    
    config = BenchmarkConfig(
        warmup_iterations=2,
        test_iterations=5,
        test_resolutions=[
            (640, 480),
            (1280, 720),
            (1920, 1080),
        ],
    )
    
    result = runner.run(config)
    
    print(f"\nBenchmark completed in {result.total_duration:.1f}s")
    print(f"Successful tests: {result.success_count}/{len(result.resolution_results)}")
    
    print("\nResults:")
    for res in result.resolution_results:
        status = "✓" if res.success else "✗"
        if res.success:
            print(f"  {status} {res.width}x{res.height}: {res.fps:.1f} FPS "
                  f"({res.avg_inference_time_ms:.1f}ms, {res.peak_memory_mb:.0f}MB)")
        else:
            print(f"  {status} {res.width}x{res.height}: Failed - {res.error_message}")
