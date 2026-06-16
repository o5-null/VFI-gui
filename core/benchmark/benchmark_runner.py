"""Benchmark runner for VFI-gui.

This module provides performance benchmarking capabilities:
- Frame interpolation speed testing with REAL model inference
- Memory usage monitoring via psutil + torch APIs
- Multi-resolution performance testing
- Device comparison (CUDA/XPU/CPU)
- JSON output for results
"""

import time
import gc
import sys
import json
import argparse
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple, TYPE_CHECKING
from enum import Enum
from pathlib import Path
from PIL import Image

# Import torch for runtime - type checker should understand this
import torch
import torch.profiler

from loguru import logger

from .device_detector import DeviceDetector, DeviceInfo, DeviceType, device_detector


# ============================================================================
# Type Definitions
# ============================================================================

class BenchmarkMode(Enum):
    """Benchmark mode selection."""
    SINGLE = "single"          # Single resolution, quick test
    MULTI_RESOLUTION = "multi_res"  # Multiple resolutions, full test


@dataclass
class HardwareStats:
    """Hardware statistics for a single iteration."""
    gpu_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    gpu_util_percent: float = 0.0


@dataclass
class ProfilerConfig:
    """Configuration for PyTorch Profiler integration."""
    # Enable/disable profiler
    enabled: bool = False
    
    # Activities to trace
    trace_cpu: bool = True        # Trace CPU ops
    trace_cuda: bool = True       # Trace CUDA ops (if available)
    
    # Profiling options
    record_shapes: bool = True    # Record input tensor shapes
    profile_memory: bool = True   # Track memory allocation
    with_stack: bool = False      # Record call stack (adds overhead)
    with_flops: bool = False      # Estimate FLOPs (not always accurate)
    
    # Output configuration
    output_dir: str = "profiler_logs"       # Directory for trace files
    export_chrome_trace: bool = True        # Export .json for chrome://tracing
    export_tensorboard: bool = False        # Export for TensorBoard viewer
    
    # Analysis
    top_ops_limit: int = 15     # Number of top ops to include in result
    sort_by: str = "cuda_time_total"  # Sort key for op breakdown


@dataclass
class OpStats:
    """Statistics for a single operator from profiler."""
    name: str = ""
    cpu_time_ms: float = 0.0
    cuda_time_ms: float = 0.0
    self_cpu_time_ms: float = 0.0
    self_cuda_time_ms: float = 0.0
    calls: int = 0
    cuda_memory_mb: float = 0.0


@dataclass
class ProfilerResult:
    """Results from PyTorch Profiler for a single resolution test."""
    # Whether profiler was active
    enabled: bool = False
    
    # Aggregate timing from profiler (microseconds → milliseconds)
    total_cpu_time_ms: float = 0.0
    total_cuda_time_ms: float = 0.0
    
    # Total memory from profiler
    total_cpu_memory_mb: float = 0.0
    total_cuda_memory_mb: float = 0.0
    
    # Per-op breakdown (top N ops)
    top_ops: List[OpStats] = field(default_factory=list)
    
    # Full table string (for display/logging)
    op_table: str = ""
    
    # Chrome trace file path (if exported)
    chrome_trace_path: str = ""
    
    # TensorBoard log directory (if exported)
    tensorboard_dir: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Benchmark mode
    mode: BenchmarkMode = BenchmarkMode.MULTI_RESOLUTION
    
    # Test parameters
    warmup_iterations: int = 3
    
    # Number of times to repeat the full frame sequence
    # Each repetition processes ALL frames through the production pipeline.
    # Higher values give more stable results under sustained load.
    # (e.g., 40 frames × 3 repetitions = 117 inferences per resolution)
    sequence_repetitions: int = 3
    
    # Legacy: test_iterations is no longer used by _benchmark_resolution.
    # Kept for backward compatibility with external callers.
    test_iterations: int = 10
    
    # Resolution tests (width, height)
    test_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (640, 480),   # SD
        (1280, 720),  # HD
        (1920, 1080), # FHD
        (2560, 1440), # QHD
        (3840, 2160), # 4K
    ])
    
    # Example frame directory (relative to VFI-gui root)
    example_dir: str = "example/1080p"
    
    # Model configuration
    model_type: str = "rife"          # Model type string (rife, film, amt, ifrnet)
    checkpoint_name: str = "rife49.pth"  # Checkpoint filename
    multiplier: int = 2
    dtype: str = "float16"
    
    # Device selection
    device_id: int = 0
    device_type: Optional[DeviceType] = None
    
    # Pre-resolved device info (bypasses re-detection in worker threads)
    # Set this from the main thread to avoid torch XPU/CUDA calls in QThread
    resolved_device: Optional[DeviceInfo] = None
    
    # Threading
    num_threads: int = 0  # 0 = use PyTorch default (all logical cores)
    
    # Hardware monitoring
    monitor_hardware: bool = True
    
    # Torch compile
    torch_compile: bool = False
    
    # PyTorch Profiler
    profiler_config: ProfilerConfig = field(default_factory=ProfilerConfig)
    
    # Output configuration
    output_path: str = "benchmark_results.json"


@dataclass
class ResolutionResult:
    """Benchmark result for a specific resolution."""
    resolution: Tuple[int, int]
    width: int = 0
    height: int = 0
    
    # Timing results (milliseconds)
    first_frame_ms: float = 0.0       # First inference including model load
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    std_inference_time_ms: float = 0.0
    
    # Throughput
    fps: float = 0.0
    
    # Memory results (MB)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Hardware stats per iteration
    hardware_stats: List[HardwareStats] = field(default_factory=list)
    
    # Profiler results (from torch.profiler)
    profiler_result: Optional[ProfilerResult] = None
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BenchmarkResult to dict for JSON serialization."""
        def convert_for_json(obj: Any) -> Any:
            """Recursively convert non-JSON-serializable types."""
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, tuple):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dataclass_fields__'):
                d = asdict(obj)
                return convert_for_json(d)
            return obj
        
        data = asdict(self)
        return convert_for_json(data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def save_to_file(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# Benchmark Runner
# ============================================================================


class BenchmarkRunner:
    """Runs performance benchmarks for VFI models with REAL inference.
    
    This class provides comprehensive benchmarking:
    - Tests multiple resolutions with REAL model inference
    - Measures inference speed and throughput
    - Monitors memory usage and hardware stats
    - Provides recommendations
    - Outputs JSON results
    
    Usage:
        runner = BenchmarkRunner()
        
        # Configure benchmark
        config = BenchmarkConfig(
            mode=BenchmarkMode.SINGLE,
            test_iterations=10,
            test_resolutions=[(1920, 1080)],
        )
        
        # Run benchmark
        result = runner.run(config)
        
        # Get results
        for res_result in result.resolution_results:
            print(f"{res_result.width}x{res_result.height}: {res_result.fps:.1f} FPS")
        
        # Save to JSON
        result.save_to_file("benchmark_results.json")
    """
    
    # Project root path (VFI-gui directory)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    def __init__(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        """Initialize the benchmark runner.
        
        Args:
            progress_callback: Optional callback for progress updates.
                              Called with (message, progress_percent)
        """
        self.progress_callback = progress_callback
        self._device_detector = device_detector
        self._cancelled = False
        self._model: Any = None  # Loaded VFI model
    
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
        
        Uses the production FrameProcessor pipeline with ALL test frames,
        not just a single pair repeated. This ensures benchmark results
        reflect real-world performance including cache behavior, memory
        pressure, and frame-dependent computation variance.
        
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
        
        # Get device info — prefer pre-resolved device to avoid
        # torch.xpu/cuda calls in worker threads where they may fail
        if config.resolved_device is not None:
            result.device_info = config.resolved_device
        elif config.device_type:
            devices = self._device_detector.get_devices_by_type(config.device_type)
            if devices and config.device_id < len(devices):
                result.device_info = devices[config.device_id]
            else:
                result.device_info = self._device_detector.get_best_device()
        else:
            result.device_info = self._device_detector.get_best_device()
        
        self._report_progress("Initializing benchmark", 5.0)
        
        # Resolve example directory path and load ALL frames
        example_path = self.PROJECT_ROOT / config.example_dir
        
        try:
            self._report_progress("Loading test frames", 10.0)
            
            # Load ALL PNG frames from example directory
            frame_paths = sorted(example_path.glob("*.png"))
            if len(frame_paths) < 2:
                logger.error(f"Not enough test frames in {example_path} (found {len(frame_paths)})")
                result.end_time = time.time()
                return result
            
            # Load as PIL → numpy → stack to [N, H, W, C] NHWC uint8
            frames_list = []
            for fp in frame_paths:
                img = Image.open(fp).convert('RGB')
                frames_list.append(np.array(img))
            base_frames = np.stack(frames_list, axis=0)
            
            logger.info(
                f"Loaded {len(frame_paths)} frames "
                f"({base_frames.shape[1]}x{base_frames.shape[2]}) from {example_path}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load test frames: {e}")
            result.end_time = time.time()
            return result
        
        # Determine resolutions to test based on mode
        if config.mode == BenchmarkMode.SINGLE:
            if config.test_resolutions:
                resolutions_to_test: List[Tuple[int, int]] = config.test_resolutions
            else:
                native_h, native_w = base_frames.shape[1], base_frames.shape[2]
                resolutions_to_test = [(native_w, native_h)]
        else:
            resolutions_to_test = config.test_resolutions
        
        # Load model ONCE for the entire benchmark
        try:
            self._report_progress("Loading model", 15.0)
            self._load_model(config, result.device_info)
            
            if self._model is None:
                raise RuntimeError("Model failed to load")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            result.end_time = time.time()
            return result
        
        # Run benchmarks for each resolution
        total_resolutions = len(resolutions_to_test)
        
        for idx, resolution in enumerate(resolutions_to_test):
            if self._cancelled:
                logger.info("Benchmark cancelled by user")
                break
            
            res_result = self._benchmark_resolution(
                config, resolution, result.device_info, base_frames,
                idx, total_resolutions,
            )
            result.resolution_results.append(res_result)
            
            if res_result.success and res_result.fps > result.best_fps:
                result.best_fps = res_result.fps
                result.best_resolution = resolution
            
            # Clear GPU cache between resolution tests
            self._clear_gpu_cache(result.device_info)
        
        # Unload model to free memory
        self._unload_model()
        
        # Calculate recommendations
        self._report_progress("Calculating recommendations", 95.0)
        result.recommended_batch_size = self._calculate_recommendations(result)
        
        result.end_time = time.time()
        self._report_progress("Benchmark complete", 100.0)
        
        return result
    
    def _load_model(self, config: BenchmarkConfig, device: DeviceInfo) -> None:
        """Load the VFI model using the production VFIConfig + MODEL_REGISTRY.
        
        Resolves checkpoint path via ModelManager's path resolution logic,
        then constructs model via VFIConfig (the canonical production path).
        
        Args:
            config: Benchmark configuration
            device: Device to load model on
        """
        from core.pytorch_models.vfi_torch import MODEL_REGISTRY, ModelType, VFIConfig
        from core.pytorch_models.vfi_torch.base import get_model
        from core.pytorch_models import ModelManager, DType
        
        # Map model_type string to ModelType enum — supports all defined types
        model_type_map = {
            "rife": ModelType.RIFE,
            "film": ModelType.FILM,
            "ifrnet": ModelType.IFRNET,
            "amt": ModelType.AMT,
            "gmfss": ModelType.GMFSS,
            "stmfnet": ModelType.STMFNET,
            "flavr": ModelType.FLAVR,
            "cain": ModelType.CAIN,
            "xvfi": ModelType.XVFI,
        }
        model_type = model_type_map.get(config.model_type.lower())
        if model_type is None:
            raise ValueError(
                f"Unknown model type: {config.model_type}. "
                f"Supported types: {list(model_type_map.keys())}"
            )
        
        # Resolve checkpoint path — use ModelManager's path resolution
        models_dir = str(self.PROJECT_ROOT / "models")
        manager = ModelManager(models_dir=models_dir, auto_download=True)
        checkpoint_path = manager._ensure_checkpoint(
            config.model_type.lower(), config.checkpoint_name
        )
        
        # Resolve device string
        if device.device_type == DeviceType.CUDA:
            device_str = f"cuda:{device.device_id}"
        elif device.device_type == DeviceType.XPU:
            device_str = f"xpu:{device.device_id}"
        else:
            device_str = "cpu"
        
        # Map dtype string to precision string
        precision_map = {
            "float16": "fp16", "fp16": "fp16",
            "float32": "fp32", "fp32": "fp32",
            "bfloat16": "bf16", "bf16": "bf16",
        }
        precision = precision_map.get(config.dtype.lower(), "fp16")
        
        # Extract model version from checkpoint name (e.g., "rife49.pth" -> "4.9")
        model_version = self._extract_model_version(config.checkpoint_name, config.model_type)
        
        # Create VFIConfig — production path
        vfi_config = VFIConfig(
            model_type=model_type,
            model_version=model_version,
            checkpoint_path=checkpoint_path,
            device=device_str,
            precision=precision,
            multiplier=config.multiplier,
            scale=1.0,
            fast_mode=False,
        )
        
        # Create model instance — try MODEL_REGISTRY first, then get_model() fallback
        if model_type in MODEL_REGISTRY:
            self._model = MODEL_REGISTRY[model_type](vfi_config)
        else:
            # Fallback to get_model() which has extended model class imports
            self._model = get_model(vfi_config)
        
        # Load model weights with resolved path
        self._model.load_model(checkpoint_path)
        
        # Optionally torch.compile for optimized inference
        if config.torch_compile:
            logger.info("Compiling model with torch.compile()...")
            self._model.compile()
        
        logger.info(f"Loaded model: {config.model_type} v{model_version} on {device_str}"
                     f"{' + compile' if config.torch_compile else ''}")
    
    def _extract_model_version(self, checkpoint_name: str, model_type: str) -> str:
        """Extract model version from checkpoint filename.
        
        Args:
            checkpoint_name: Checkpoint filename (e.g., "rife49.pth")
            model_type: Model type string
            
        Returns:
            Model version string (e.g., "4.9")
        """
        # Checkpoint-to-version mappings per model type
        version_maps: Dict[str, Dict[str, str]] = {
            "rife": {
                "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
                "rife47.pth": "4.7",
                "rife49.pth": "4.9",
                "rife417.pth": "4.17",
                "rife422.pth": "4.22",
                "rife426.pth": "4.26",
            },
            "film": {
                "film_net_fp32.pt": "fp32",
            },
            "ifrnet": {
                "IFRNet_S_Vimeo90K.pth": "S_Vimeo90K",
                "IFRNet_L_Vimeo90K.pth": "L_Vimeo90K",
                "IFRNet_S_GoPro.pth": "S_GoPro",
                "IFRNet_L_GoPro.pth": "L_GoPro",
            },
            "amt": {
                "amt-s.pth": "s",
                "amt-l.pth": "l",
                "amt-g.pth": "g",
                "gopro_amt-s.pth": "gopro-s",
            },
        }

        type_map = version_maps.get(model_type.lower(), {})
        if checkpoint_name in type_map:
            return type_map[checkpoint_name]

        # Fallback: try to extract version from RIFE-like filenames
        if model_type.lower() == "rife":
            match = re.search(r'rife(\d+)', checkpoint_name)
            if match:
                version_num = match.group(1)
                if len(version_num) >= 2:
                    return f"{version_num[0]}.{version_num[1:]}"
                return f"4.{version_num}"

        # Default: use filename stem as version
        return Path(checkpoint_name).stem
    
    def _unload_model(self) -> None:
        """Unload the model and free memory."""
        if self._model is not None:
            self._model.unload()
            self._model = None
    
    def _benchmark_resolution(
        self,
        config: BenchmarkConfig,
        resolution: Tuple[int, int],
        device: DeviceInfo,
        base_frames: np.ndarray,
        res_index: int = 0,
        total_resolutions: int = 1,
    ) -> ResolutionResult:
        """Benchmark a specific resolution using the production FrameProcessor.
        
        Uses FrameProcessor.process() with ALL loaded frames (e.g. 40 frames = 39 pairs)
        for each repetition, giving real-world performance numbers including:
        - Varied frame content (not just one repeated pair)
        - Real cache behavior and memory pressure
        - Full pipeline overhead (preprocess, interpolate, postprocess, cache mgmt)
        
        Per-inference timing is collected via FrameProcessor's enable_perf_stats.
        
        Args:
            config: Benchmark configuration
            resolution: Resolution to test (width, height)
            device: Device to test on
            base_frames: Base frames in NHWC uint8 format [N, H, W, C]
            res_index: Current resolution index (for progress reporting)
            total_resolutions: Total resolutions to test (for progress reporting)
            
        Returns:
            Benchmark result for the resolution
        """
        from core.pytorch_models import FrameProcessor, InterpolationConfig, DType
        
        result = ResolutionResult(resolution=resolution)
        width, height = resolution
        
        # Ensure model is loaded
        if self._model is None:
            result.success = False
            result.error_message = "Model not loaded"
            return result
        
        # Save original thread count for restoration
        original_num_threads = torch.get_num_threads()
        
        try:
            # Apply thread count
            if config.num_threads > 0:
                torch.set_num_threads(config.num_threads)
                logger.debug(f"Set inference threads: {config.num_threads}")
            
            # Resize frames to target resolution if needed
            target_w, target_h = resolution
            cur_h, cur_w = base_frames.shape[1], base_frames.shape[2]
            
            if (cur_w, cur_h) != (target_w, target_h):
                from PIL import Image as PILImage
                resized = []
                for i in range(len(base_frames)):
                    img = PILImage.fromarray(base_frames[i])
                    img = img.resize((target_w, target_h), PILImage.Resampling.BILINEAR)
                    resized.append(np.array(img))
                frames = np.stack(resized, axis=0)
            else:
                frames = base_frames
            
            # Convert to torch tensor NHWC float32 [0, 1] for FrameProcessor
            frames_tensor = torch.from_numpy(frames).float() / 255.0
            
            # Map dtype string to DType enum
            dtype_map = {
                "float16": DType.FLOAT16, "fp16": DType.FLOAT16,
                "float32": DType.FLOAT32, "fp32": DType.FLOAT32,
                "bfloat16": DType.BFLOAT16, "bf16": DType.BFLOAT16,
            }
            dtype_enum = dtype_map.get(config.dtype.lower(), DType.FLOAT16)
            
            # Create InterpolationConfig (same as production)
            interp_config = InterpolationConfig(
                multiplier=config.multiplier,
                dtype=dtype_enum,
            )
            
            # Create FrameProcessor with perf_stats enabled
            processor = FrameProcessor(self._model, interp_config, enable_perf_stats=True)
            
            # Progress calculation
            bench_start = 15.0
            bench_range = 80.0
            res_weight = bench_range / max(total_resolutions, 1)
            res_base = bench_start + res_index * res_weight
            
            # Total runs: warmup + (sequence_repetitions test runs)
            total_runs = config.warmup_iterations + config.sequence_repetitions
            run_weight = res_weight / max(total_runs, 1)
            
            self._report_progress(f"Testing {width}x{height}", res_base)
            
            # Reset GPU memory stats before benchmark
            self._reset_gpu_memory_stats(device)
            self._synchronize_device(device)
            
            # ---- Warmup: full FrameProcessor path ----
            for wi in range(config.warmup_iterations):
                _ = processor.process(
                    frames_tensor, multiplier=config.multiplier,
                    clear_cache_after_n_frames=999,
                )
                self._synchronize_device(device)
                pct = res_base + (wi + 1) * run_weight
                self._report_progress(
                    f"Warmup {wi+1}/{config.warmup_iterations} {width}x{height}", pct
                )
            
            # Clear cache after warmup
            self._clear_gpu_cache(device)
            self._reset_gpu_memory_stats(device)
            self._synchronize_device(device)
            
            warmup_end = res_base + config.warmup_iterations * run_weight
            
            # ---- Test runs: full FrameProcessor with ALL frames ----
            # Each run processes all N frames through the production pipeline.
            # sequence_repetitions controls how many times we repeat the full sequence.
            all_inference_times: List[float] = []
            all_fps_values: List[float] = []
            memory_readings: List[float] = []
            hardware_stats_list: List[HardwareStats] = []
            
            # First frame timing (includes any lazy initialization)
            first_frame_ms = 0.0
            
            for rep in range(config.sequence_repetitions):
                if self._cancelled:
                    break
                
                # Run full production pipeline on all frames
                self._synchronize_device(device)
                seq_start = time.perf_counter()
                
                interp_result = processor.process(
                    frames_tensor, multiplier=config.multiplier,
                    clear_cache_after_n_frames=10,
                )
                
                self._synchronize_device(device)
                seq_end = time.perf_counter()
                seq_ms = (seq_end - seq_start) * 1000
                
                # Extract per-inference timing from PerfStats
                if interp_result.perf_stats is not None:
                    perf = interp_result.perf_stats
                    
                    # First repetition: record first-frame time
                    if rep == 0 and perf.entries:
                        first_frame_ms = perf.entries[0].inference_ms
                    
                    # Collect all per-inference times from this run
                    for entry in perf.entries:
                        all_inference_times.append(entry.inference_ms)
                    
                    # FPS based on avg per-inference time
                    if perf.avg_inference_ms > 0:
                        all_fps_values.append(1000.0 / perf.avg_inference_ms)
                
                # Collect hardware stats
                if config.monitor_hardware:
                    hw_stats = self._collect_hardware_stats(device)
                    hardware_stats_list.append(hw_stats)
                    
                    if device.device_type in (DeviceType.CUDA, DeviceType.XPU):
                        memory_mb = hw_stats.gpu_memory_mb
                        memory_readings.append(memory_mb)
                
                # Progress
                pct = warmup_end + (rep + 1) * run_weight
                n_pairs = len(frames_tensor) - 1
                seq_fps = (n_pairs * 1000.0 / seq_ms) if seq_ms > 0 else 0
                self._report_progress(
                    f"Run {rep+1}/{config.sequence_repetitions} "
                    f"{width}x{height}: {seq_ms:.0f}ms total, "
                    f"{seq_fps:.1f} seq FPS",
                    pct,
                )
            
            # === Profiler: separate run if enabled ===
            profiler_cfg = config.profiler_config
            if profiler_cfg.enabled and not self._cancelled:
                prof = self._create_profiler(profiler_cfg, device, 1)
                prof.start()
                _ = processor.process(
                    frames_tensor, multiplier=config.multiplier,
                    clear_cache_after_n_frames=10,
                )
                self._synchronize_device(device)
                prof.stop()
                result.profiler_result = self._extract_profiler_results(
                    prof, profiler_cfg, resolution, device
                )
            
            # Calculate statistics from all per-inference times
            if all_inference_times:
                result.avg_inference_time_ms = sum(all_inference_times) / len(all_inference_times)
                result.min_inference_time_ms = min(all_inference_times)
                result.max_inference_time_ms = max(all_inference_times)
                
                if len(all_inference_times) > 1:
                    mean = result.avg_inference_time_ms
                    variance = sum((t - mean) ** 2 for t in all_inference_times) / (len(all_inference_times) - 1)
                    result.std_inference_time_ms = variance ** 0.5
                
                result.fps = 1000.0 / result.avg_inference_time_ms if result.avg_inference_time_ms > 0 else 0
            
            # First frame
            result.first_frame_ms = first_frame_ms
            
            # Memory statistics
            if memory_readings:
                result.avg_memory_mb = sum(memory_readings) / len(memory_readings)
            
            result.peak_memory_mb = self._get_peak_memory(device)
            result.hardware_stats = hardware_stats_list
            
            result.success = True
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"Out of memory at {width}x{height}: {e}")
            result.success = False
            result.error_message = "Out of memory"
            
        except Exception as e:
            logger.error(f"Benchmark failed for {width}x{height}: {e}")
            result.success = False
            result.error_message = str(e)
        
        finally:
            # Restore original thread count
            if config.num_threads > 0:
                torch.set_num_threads(original_num_threads)
        
        return result
    
    def _collect_hardware_stats(self, device: DeviceInfo) -> HardwareStats:
        """Collect hardware statistics after inference.
        
        Args:
            device: Device being tested
            
        Returns:
            HardwareStats with current measurements
        """
        stats = HardwareStats()
        
        # GPU memory
        if device.device_type == DeviceType.CUDA:
            try:
                stats.gpu_memory_mb = torch.cuda.memory_allocated(device.device_id) / 1024 / 1024
            except Exception:
                pass
        elif device.device_type == DeviceType.XPU:
            try:
                if hasattr(torch, 'xpu'):
                    stats.gpu_memory_mb = torch.xpu.memory_allocated(device.device_id) / 1024 / 1024
            except Exception:
                pass
        
        # CPU/RAM via psutil
        try:
            import psutil
            stats.cpu_percent = psutil.cpu_percent(interval=0)
            stats.ram_used_mb = psutil.virtual_memory().used / 1024 / 1024
        except ImportError:
            pass
        
        # GPU utilization via pynvml
        if device.device_type == DeviceType.CUDA:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.device_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats.gpu_util_percent = float(util.gpu)
                pynvml.nvmlShutdown()
            except (ImportError, Exception):
                stats.gpu_util_percent = 0.0
        
        return stats
    
    def _synchronize_device(self, device: DeviceInfo) -> None:
        """Synchronize device for accurate timing."""
        if device.device_type == DeviceType.CUDA:
            torch.cuda.synchronize(device.device_id)
        elif device.device_type == DeviceType.XPU:
            if hasattr(torch, 'xpu'):
                torch.xpu.synchronize(device.device_id)
    
    def _reset_gpu_memory_stats(self, device: DeviceInfo) -> None:
        """Reset GPU memory statistics."""
        if device.device_type == DeviceType.CUDA:
            torch.cuda.reset_peak_memory_stats(device.device_id)
        elif device.device_type == DeviceType.XPU:
            # XPU doesn't have reset_peak_memory_stats in some versions
            pass
    
    def _get_peak_memory(self, device: DeviceInfo) -> float:
        """Get peak GPU memory in MB."""
        if device.device_type == DeviceType.CUDA:
            return torch.cuda.max_memory_allocated(device.device_id) / 1024 / 1024
        elif device.device_type == DeviceType.XPU:
            if hasattr(torch, 'xpu'):
                try:
                    return torch.xpu.max_memory_allocated(device.device_id) / 1024 / 1024
                except AttributeError:
                    return 0.0
        return 0.0
    
    def _clear_gpu_cache(self, device: DeviceInfo) -> None:
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        
        if device.device_type == DeviceType.CUDA:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device.device_id)
        elif device.device_type == DeviceType.XPU:
            if hasattr(torch, 'xpu'):
                torch.xpu.empty_cache()
                torch.xpu.synchronize(device.device_id)
    
    def _create_profiler(
        self,
        config: ProfilerConfig,
        device: DeviceInfo,
        iterations: int = 10
    ) -> torch.profiler.profile:
        """Create a PyTorch profiler instance.
        
        Args:
            config: Profiler configuration
            device: Device being tested
            iterations: Number of test iterations (for schedule)
            
        Returns:
            Configured torch.profiler.profile instance
        """
        # Determine activities
        activities = [torch.profiler.ProfilerActivity.CPU]
        if config.trace_cuda and device.device_type == DeviceType.CUDA:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        # Build kwargs
        kwargs: Dict[str, Any] = {
            "activities": activities,
            "record_shapes": config.record_shapes,
            "profile_memory": config.profile_memory,
            "with_stack": config.with_stack,
            "with_flops": config.with_flops,
        }
        
        # Add schedule (no wait/warmup since we handle those manually)
        schedule = torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=iterations,
            repeat=1,
        )
        kwargs["schedule"] = schedule
        
        # Add TensorBoard handler if requested
        if config.export_tensorboard:
            output_dir = str(self.PROJECT_ROOT / config.output_dir)
            kwargs["on_trace_ready"] = torch.profiler.tensorboard_trace_handler(output_dir)
        
        return torch.profiler.profile(**kwargs)
    
    def _extract_profiler_results(
        self,
        prof: torch.profiler.profile,
        config: ProfilerConfig,
        resolution: Tuple[int, int],
        device: DeviceInfo,
    ) -> ProfilerResult:
        """Extract results from PyTorch Profiler.
        
        Args:
            prof: Completed profiler instance
            config: Profiler configuration
            resolution: Test resolution (for trace filename)
            device: Device being tested
            
        Returns:
            ProfilerResult with extracted data
        """
        result = ProfilerResult(enabled=True)
        
        try:
            # Get key averages
            key_avgs = prof.key_averages()
            
            # Get total average
            total_avg = prof.total_average()
            result.total_cpu_time_ms = total_avg.cpu_time_total / 1000  # μs → ms
            result.total_cuda_time_ms = total_avg.cuda_time_total / 1000
            
            # Memory from profiler
            if hasattr(total_avg, 'cpu_memory_usage'):
                result.total_cpu_memory_mb = total_avg.cpu_memory_usage / 1024 / 1024
            if hasattr(total_avg, 'cuda_memory_usage'):
                result.total_cuda_memory_mb = total_avg.cuda_memory_usage / 1024 / 1024
            
            # Determine sort key
            sort_key = config.sort_by
            if device.device_type != DeviceType.CUDA and sort_key.startswith("cuda"):
                sort_key = "cpu_time_total"  # Fallback for non-CUDA
            
            # Extract top N ops
            sorted_avgs = sorted(
                key_avgs,
                key=lambda evt: getattr(evt, sort_key, 0),
                reverse=True,
            )
            
            for evt in sorted_avgs[:config.top_ops_limit]:
                op = OpStats(
                    name=evt.key,
                    cpu_time_ms=evt.cpu_time_total / 1000,
                    cuda_time_ms=getattr(evt, 'cuda_time_total', 0) / 1000,
                    self_cpu_time_ms=evt.self_cpu_time_total / 1000,
                    self_cuda_time_ms=getattr(evt, 'self_cuda_time_total', 0) / 1000,
                    calls=evt.count,
                )
                if hasattr(evt, 'cuda_memory_usage'):
                    op.cuda_memory_mb = evt.cuda_memory_usage / 1024 / 1024
                result.top_ops.append(op)
            
            # Generate op table string
            result.op_table = key_avgs.table(
                sort_by=sort_key,
                row_limit=config.top_ops_limit,
            )
            
            # Export Chrome trace
            if config.export_chrome_trace:
                trace_dir = self.PROJECT_ROOT / config.output_dir
                trace_dir.mkdir(parents=True, exist_ok=True)
                w, h = resolution
                trace_path = trace_dir / f"trace_{w}x{h}.json"
                prof.export_chrome_trace(str(trace_path))
                result.chrome_trace_path = str(trace_path)
                logger.info(f"Chrome trace exported: {trace_path}")
            
            # TensorBoard dir
            if config.export_tensorboard:
                result.tensorboard_dir = str(self.PROJECT_ROOT / config.output_dir)
            
        except Exception as e:
            logger.warning(f"Failed to extract profiler results: {e}")
            result.enabled = False
        
        return result
    
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
            Quick test results as dict
        """
        if device is None:
            device = self._device_detector.get_best_device()
        
        config = BenchmarkConfig(
            mode=BenchmarkMode.SINGLE,
            warmup_iterations=1,
            sequence_repetitions=1,
            test_resolutions=[(1280, 720)],
            monitor_hardware=True,
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
                    "first_frame_ms": round(r.first_frame_ms, 2),
                    "memory_mb": round(r.peak_memory_mb, 0),
                }
                for r in result.resolution_results
            ],
            "best_resolution": result.best_resolution,
            "recommended_batch_size": result.recommended_batch_size,
        }


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_cli():
    """CLI entry point for benchmark runner."""
    # Pre-parse --debug to set log level before other imports emit logs
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--debug", action="store_true", help="Enable debug-level logging (per-frame details)")
    pre_args, _ = pre_parser.parse_known_args()
    
    log_level = "DEBUG" if pre_args.debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    parser = argparse.ArgumentParser(
        description="VFI-gui Benchmark Runner - Test frame interpolation performance"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["single", "multi_resolution"],
        default="multi_resolution",
        help="Benchmark mode: single (quick 1080p test) or multi_resolution (full test)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: reduced iterations and fewer resolutions"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path (default: benchmark_results.json)"
    )
    
    parser.add_argument(
        "--model", "-M",
        type=str,
        default="rife",
        help="Model type: rife, film, amt, ifrnet (default: rife)"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="rife49.pth",
        help="Checkpoint filename (default: rife49.pth)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=0,
        help="Number of CPU threads for inference (0 = PyTorch default, all logical cores)"
    )
    
    parser.add_argument(
        "--repetitions", "-r",
        type=int,
        default=3,
        help="Number of times to repeat the full frame sequence (default: 3, higher = more stable)"
    )
    
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile() for optimized inference"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging (per-frame interpolation details)"
    )
    
    parser.add_argument(
        "--profiler", "-p",
        action="store_true",
        help="Enable PyTorch Profiler for detailed op-level analysis"
    )
    
    parser.add_argument(
        "--profiler-output",
        type=str,
        default="profiler_logs",
        help="Directory for profiler trace output (default: profiler_logs)"
    )
    
    parser.add_argument(
        "--profiler-tensorboard",
        action="store_true",
        help="Export profiler traces to TensorBoard format"
    )
    
    args = parser.parse_args()
    
    # Create config based on args
    mode = BenchmarkMode.SINGLE if args.mode == "single" else BenchmarkMode.MULTI_RESOLUTION
    
    # Create profiler config
    profiler_config = ProfilerConfig(
        enabled=args.profiler,
        output_dir=args.profiler_output,
        export_tensorboard=args.profiler_tensorboard,
    )
    
    if args.quick:
        config = BenchmarkConfig(
            mode=mode,
            warmup_iterations=1,
            sequence_repetitions=1,
            test_resolutions=[(1280, 720)],
            model_type=args.model,
            checkpoint_name=args.checkpoint,
            device_id=args.device,
            num_threads=args.threads,
            torch_compile=args.compile,
            output_path=args.output,
            monitor_hardware=True,
            profiler_config=profiler_config,
        )
    else:
        config = BenchmarkConfig(
            mode=mode,
            warmup_iterations=3,
            sequence_repetitions=args.repetitions,
            model_type=args.model,
            checkpoint_name=args.checkpoint,
            device_id=args.device,
            num_threads=args.threads,
            torch_compile=args.compile,
            output_path=args.output,
            monitor_hardware=True,
            profiler_config=profiler_config,
        )
    
    # Run benchmark
    print("=" * 60)
    print("VFI-gui Benchmark")
    print("=" * 60)
    print(f"Mode: {mode.value}")
    print(f"Model: {args.model} ({args.checkpoint})")
    print(f"Device: {args.device}")
    print(f"Sequence repetitions: {args.repetitions}")
    if args.compile:
        print(f"Torch compile: enabled")
    if args.debug:
        print(f"Log level: DEBUG")
    if args.threads > 0:
        print(f"Threads: {args.threads}")
    print()
    
    def progress_callback(message: str, progress: float):
        print(f"[{progress:5.1f}%] {message}")
    
    runner = BenchmarkRunner(progress_callback)
    
    try:
        result = runner.run(config)
        
        # Save results
        result.save_to_file(args.output)
        print(f"\nResults saved to: {args.output}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        
        for res in result.resolution_results:
            status = "OK" if res.success else "FAIL"
            if res.success:
                profiler_info = ""
                if res.profiler_result and res.profiler_result.enabled:
                    profiler_info = (
                        f", profiler: {len(res.profiler_result.top_ops)} top ops, "
                        f"trace: {res.profiler_result.chrome_trace_path or 'N/A'}"
                    )
                print(f"  [{status}] {res.width}x{res.height}: "
                      f"{res.fps:.1f} FPS "
                      f"(avg: {res.avg_inference_time_ms:.1f}ms, "
                      f"first: {res.first_frame_ms:.1f}ms, "
                      f"mem: {res.peak_memory_mb:.0f}MB{profiler_info})")
            else:
                print(f"  [{status}] {res.width}x{res.height}: {res.error_message}")
        
        print(f"\nBest FPS: {result.best_fps:.1f} at {result.best_resolution}")
        print(f"Recommended batch size: {result.recommended_batch_size}")
        print(f"Total duration: {result.total_duration:.1f}s")
        
        # Output JSON to stdout if requested
        if args.json:
            print("\n" + "=" * 60)
            print("JSON Output")
            print("=" * 60)
            print(result.to_json())
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    run_cli()