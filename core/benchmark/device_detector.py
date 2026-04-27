"""Device detection for VFI-gui.

This module provides comprehensive device detection capabilities:
- GPU type detection (NVIDIA CUDA, AMD ROCm, Intel XPU)
- Device properties and capabilities
- Memory information
- Compute capability detection
"""

import os
import sys
import platform
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from loguru import logger

from core.device_type import DeviceType


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device_type: DeviceType
    name: str
    device_id: int = 0
    total_memory_mb: int = 0
    available_memory_mb: int = 0
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    pci_bus_id: Optional[str] = None
    multi_processor_count: int = 0
    max_threads_per_block: int = 0
    is_available: bool = True
    platform: str = ""  # "cuda", "rocm", or "xpu" for GPU backends
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def display_name(self) -> str:
        """Get a display-friendly name."""
        if self.device_type == DeviceType.CUDA:
            return f"NVIDIA {self.name}"
        elif self.device_type == DeviceType.ROCM:
            return f"AMD {self.name}"
        elif self.device_type == DeviceType.XPU:
            return f"Intel {self.name}"
        elif self.device_type == DeviceType.CPU:
            return f"CPU: {self.name}"
        return self.name
    
    @property
    def memory_gb(self) -> float:
        """Get total memory in GB."""
        return self.total_memory_mb / 1024
    
    @property
    def memory_status(self) -> str:
        """Get memory status string."""
        used = self.total_memory_mb - self.available_memory_mb
        return f"{used / 1024:.1f} / {self.memory_gb:.1f} GB"


@dataclass
class SystemInfo:
    """System-wide information."""
    os_name: str
    os_version: str
    cpu_count: int
    total_ram_mb: int
    python_version: str
    pytorch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    hip_version: Optional[str] = None  # ROCm/HIP version for AMD GPUs
    
    @property
    def total_ram_gb(self) -> float:
        """Get total RAM in GB."""
        return self.total_ram_mb / 1024


class DeviceDetector:
    """Detects and provides information about available compute devices.
    
    This class handles detection of:
    - NVIDIA GPUs via CUDA
    - AMD GPUs via ROCm
    - Intel GPUs via XPU
    - CPU fallback
    
    Usage:
        detector = DeviceDetector()
        
        # Get all available devices
        devices = detector.get_all_devices()
        
        # Get specific device type
        cuda_devices = detector.get_devices_by_type(DeviceType.CUDA)
        
        # Get best available device
        best_device = detector.get_best_device()
        
        # Get full system info
        system_info = detector.get_system_info()
    """
    
    def __init__(self):
        """Initialize the device detector."""
        self._device_cache: Dict[DeviceType, List[DeviceInfo]] = {}
        self._system_info: Optional[SystemInfo] = None
        self._torch_available = self._check_torch()
    
    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            logger.warning("PyTorch not available, device detection limited")
            return False
    
    def detect_all(self, force_refresh: bool = False) -> List[DeviceInfo]:
        """Detect all available devices.
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            List of all available devices
        """
        if not force_refresh and self._device_cache:
            return [
                device
                for devices in self._device_cache.values()
                for device in devices
            ]
        
        self._device_cache = {}
        
        # Detect CUDA devices (NVIDIA)
        cuda_devices = self._detect_cuda()
        if cuda_devices:
            self._device_cache[DeviceType.CUDA] = cuda_devices
        
        # Detect ROCm devices (AMD)
        rocm_devices = self._detect_rocm()
        if rocm_devices:
            self._device_cache[DeviceType.ROCM] = rocm_devices
        
        # Detect XPU devices (Intel)
        xpu_devices = self._detect_xpu()
        if xpu_devices:
            self._device_cache[DeviceType.XPU] = xpu_devices
        
        # Always add CPU
        cpu_device = self._detect_cpu()
        self._device_cache[DeviceType.CPU] = [cpu_device]
        
        return [
            device
            for devices in self._device_cache.values()
            for device in devices
        ]
    
    def _detect_cuda(self) -> List[DeviceInfo]:
        """Detect NVIDIA CUDA devices.
        
        Note: This only detects true NVIDIA CUDA devices.
        AMD ROCm devices are detected separately in _detect_rocm().
        """
        if not self._torch_available:
            return []
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.debug("CUDA not available")
                return []
            
            # Check if this is ROCm (torch.version.hip is set)
            # ROCm uses the same torch.cuda API but should be detected separately
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                logger.debug("torch.cuda.is_available() is True but this is ROCm, not CUDA")
                return []
            
            devices = []
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                total = props.total_memory / 1024 / 1024
                available = total - allocated
                
                # Get compute capability
                compute_capability = f"{props.major}.{props.minor}"
                
                device_info = DeviceInfo(
                    device_type=DeviceType.CUDA,
                    name=props.name,
                    device_id=i,
                    total_memory_mb=int(total),
                    available_memory_mb=int(available),
                    compute_capability=compute_capability,
                    multi_processor_count=props.multi_processor_count,
                    max_threads_per_block=props.max_threads_per_multi_processor,
                    platform="cuda",
                    extra_info={
                        "warp_size": props.warp_size,
                        "is_integrated": props.is_integrated,
                    }
                )
                devices.append(device_info)
                logger.info(f"Detected CUDA device: {props.name} ({device_info.memory_gb:.1f} GB)")
            
            return devices
            
        except Exception as e:
            logger.debug(f"CUDA detection failed: {e}")
            return []
    
    def _detect_rocm(self) -> List[DeviceInfo]:
        """Detect AMD ROCm devices.
        
        ROCm uses the same torch.cuda API as CUDA, but is identified
        by torch.version.hip being set.
        """
        if not self._torch_available:
            return []
        
        try:
            import torch
            
            # ROCm reports as torch.cuda.is_available() == True
            # but torch.version.hip is set to the ROCm version
            if not torch.cuda.is_available():
                logger.debug("ROCm not available (torch.cuda not available)")
                return []
            
            # Check if this is actually ROCm
            if not hasattr(torch.version, 'hip') or torch.version.hip is None:
                logger.debug("Not ROCm (torch.version.hip is None)")
                return []
            
            devices = []
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                total = props.total_memory / 1024 / 1024
                available = total - allocated
                
                # Get compute capability (architecture info)
                compute_capability = f"{props.major}.{props.minor}"
                
                # Try to get HIP/ROCm version
                hip_version = torch.version.hip if hasattr(torch.version, 'hip') else None
                
                device_info = DeviceInfo(
                    device_type=DeviceType.ROCM,
                    name=props.name,
                    device_id=i,
                    total_memory_mb=int(total),
                    available_memory_mb=int(available),
                    compute_capability=compute_capability,
                    multi_processor_count=props.multi_processor_count,
                    max_threads_per_block=props.max_threads_per_multi_processor,
                    platform="rocm",
                    extra_info={
                        "warp_size": props.warp_size,
                        "hip_version": hip_version,
                        "is_integrated": props.is_integrated,
                    }
                )
                devices.append(device_info)
                logger.info(f"Detected ROCm device: {props.name} ({device_info.memory_gb:.1f} GB)")
            
            return devices
            
        except Exception as e:
            logger.debug(f"ROCm detection failed: {e}")
            return []
    
    def _detect_xpu(self) -> List[DeviceInfo]:
        """Detect Intel XPU devices."""
        if not self._torch_available:
            return []
        
        try:
            import torch
            
            # Check for XPU support
            if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
                logger.debug("XPU not available")
                return []
            
            devices = []
            device_count = torch.xpu.device_count()
            
            for i in range(device_count):
                device_name = torch.xpu.get_device_name(i)
                
                device_info = DeviceInfo(
                    device_type=DeviceType.XPU,
                    name=device_name,
                    device_id=i,
                    platform="xpu",
                    extra_info={"device_count": device_count}
                )
                devices.append(device_info)
                logger.info(f"Detected XPU device: {device_name}")
            
            return devices
            
        except Exception as e:
            logger.debug(f"XPU detection failed: {e}")
            return []
    
    def _detect_cpu(self) -> DeviceInfo:
        """Detect CPU information."""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True) or 1
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            cpu_name = platform.processor() or "Unknown CPU"
            if not cpu_name or cpu_name == "":
                # Try to get CPU name from environment
                cpu_name = os.environ.get("PROCESSOR_IDENTIFIER", "CPU")
            
            return DeviceInfo(
                device_type=DeviceType.CPU,
                name=cpu_name,
                device_id=0,
                multi_processor_count=cpu_count,
                platform="cpu",
                extra_info={
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "usage_percent": cpu_percent,
                }
            )
        except ImportError:
            logger.debug("psutil not available, using basic CPU detection")
            return DeviceInfo(
                device_type=DeviceType.CPU,
                name=platform.processor() or "CPU",
                device_id=0,
                platform="cpu",
            )
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """Get devices of a specific type.
        
        Args:
            device_type: Type of device to get
            
        Returns:
            List of devices of the specified type
        """
        self.detect_all()
        return self._device_cache.get(device_type, [])
    
    def get_best_device(self) -> DeviceInfo:
        """Get the best available device for inference.
        
        Priority: CUDA > ROCm > XPU > CPU
        
        Returns:
            Best available device
        """
        self.detect_all()
        
        # Check CUDA first (NVIDIA)
        if DeviceType.CUDA in self._device_cache and self._device_cache[DeviceType.CUDA]:
            return self._device_cache[DeviceType.CUDA][0]
        
        # Then ROCm (AMD)
        if DeviceType.ROCM in self._device_cache and self._device_cache[DeviceType.ROCM]:
            return self._device_cache[DeviceType.ROCM][0]
        
        # Then XPU (Intel)
        if DeviceType.XPU in self._device_cache and self._device_cache[DeviceType.XPU]:
            return self._device_cache[DeviceType.XPU][0]
        
        # Fallback to CPU
        return self._device_cache[DeviceType.CPU][0]
    
    def get_all_devices(self) -> List[DeviceInfo]:
        """Get all available devices.
        
        Returns:
            List of all available devices
        """
        return self.detect_all()
    
    def get_system_info(self) -> SystemInfo:
        """Get system information.
        
        Returns:
            System information
        """
        if self._system_info is not None:
            return self._system_info
        
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / 1024 / 1024
            cpu_count = psutil.cpu_count(logical=True) or 1
        except ImportError:
            total_ram = 0
            cpu_count = os.cpu_count() or 1
        
        pytorch_version = None
        cuda_version = None
        hip_version = None
        
        if self._torch_available:
            try:
                import torch
                pytorch_version = torch.__version__
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda
                    # Check for ROCm/HIP version
                    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                        hip_version = torch.version.hip
            except Exception:
                pass
        
        self._system_info = SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            cpu_count=cpu_count,
            total_ram_mb=int(total_ram),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            pytorch_version=pytorch_version,
            cuda_version=cuda_version,
            hip_version=hip_version,
        )
        
        return self._system_info
    
    def is_device_available(self, device_type: DeviceType) -> bool:
        """Check if a device type is available.
        
        Args:
            device_type: Type of device to check
            
        Returns:
            True if device type is available
        """
        self.detect_all()
        return device_type in self._device_cache and len(self._device_cache[device_type]) > 0
    
    def get_device_capabilities(self, device: DeviceInfo) -> Dict[str, Any]:
        """Get detailed capabilities for a device.
        
        Args:
            device: Device to get capabilities for
            
        Returns:
            Dictionary of device capabilities
        """
        capabilities = {
            "device_type": device.device_type.value,
            "name": device.name,
            "memory_gb": device.memory_gb,
            "supports_fp16": False,
            "supports_bf16": False,
            "supports_tensor_cores": False,
        }
        
        if device.device_type == DeviceType.CUDA:
            capabilities["supports_fp16"] = True
            
            # Check for tensor cores (compute capability >= 7.0)
            if device.compute_capability:
                try:
                    major = int(device.compute_capability.split('.')[0])
                    capabilities["supports_tensor_cores"] = major >= 7
                    capabilities["supports_bf16"] = major >= 8 or (major == 7 and int(device.compute_capability.split('.')[1]) >= 5)
                except (ValueError, IndexError):
                    pass
        
        elif device.device_type == DeviceType.ROCM:
            capabilities["supports_fp16"] = True
            # AMD ROCm typically supports FP16
            # BF16 support depends on architecture (RDNA3+)
            if device.compute_capability:
                try:
                    # AMD usesgfx architecture codes
                    # For simplicity, assume newer AMD GPUs support BF16
                    capabilities["supports_bf16"] = True
                except (ValueError, IndexError):
                    pass
        
        elif device.device_type == DeviceType.XPU:
            capabilities["supports_fp16"] = True
        
        return capabilities


# Singleton instance
device_detector = DeviceDetector()


if __name__ == "__main__":
    # Test device detection
    detector = DeviceDetector()
    
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    sys_info = detector.get_system_info()
    print(f"OS: {sys_info.os_name} {sys_info.os_version}")
    print(f"CPU Count: {sys_info.cpu_count}")
    print(f"Total RAM: {sys_info.total_ram_gb:.1f} GB")
    print(f"Python: {sys_info.python_version}")
    print(f"PyTorch: {sys_info.pytorch_version or 'Not installed'}")
    print(f"CUDA Version: {sys_info.cuda_version or 'N/A'}")
    print(f"HIP Version: {sys_info.hip_version or 'N/A'}")
    
    print("\n" + "=" * 60)
    print("Detected Devices")
    print("=" * 60)
    
    devices = detector.get_all_devices()
    for device in devices:
        print(f"\n{device.display_name}")
        print(f"  Type: {device.device_type.value}")
        print(f"  Memory: {device.memory_gb:.1f} GB")
        if device.compute_capability:
            print(f"  Compute Capability: {device.compute_capability}")
        
        caps = detector.get_device_capabilities(device)
        print(f"  FP16: {'Yes' if caps['supports_fp16'] else 'No'}")
        print(f"  BF16: {'Yes' if caps['supports_bf16'] else 'No'}")
        print(f"  Tensor Cores: {'Yes' if caps['supports_tensor_cores'] else 'No'}")
