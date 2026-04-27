"""Unified Device Manager for VFI-gui.

This module provides a unified API for device detection and management.
It acts as a facade over DeviceDetector, offering a single entry point
for all device-related operations.

Supported device types:
- CUDA: NVIDIA GPUs
- ROCm: AMD GPUs (via ROCm)
- XPU: Intel GPUs
- CPU: CPU fallback

Usage:
    from core.device_manager import device_manager
    
    # Get all available devices
    devices = device_manager.get_devices()
    
    # Get best device for inference
    best = device_manager.get_best_device()
    
    # Check if specific device type is available
    if device_manager.is_available(DeviceType.CUDA):
        # Use CUDA device
        pass
"""

from typing import Dict, List, Optional, Any

from loguru import logger

from core.device_type import DeviceType
from core.benchmark.device_detector import DeviceDetector, DeviceInfo, SystemInfo


class DeviceManager:
    """Unified device management facade.
    
    This class provides a simplified API for device detection and queries.
    It wraps DeviceDetector and potentially other device-related components.
    
    Key features:
    - Unified device type enumeration (CUDA, ROCm, XPU, CPU)
    - Device detection and capability queries
    - Best device selection
    - System information retrieval
    
    This class does NOT handle runtime/venv activation.
    For runtime management, use RuntimeManager directly.
    """
    
    def __init__(self):
        """Initialize the device manager."""
        self._detector = DeviceDetector()
        self._current_device: Optional[DeviceInfo] = None
    
    def get_devices(self, force_refresh: bool = False) -> List[DeviceInfo]:
        """Get all available devices.
        
        Args:
            force_refresh: Force re-detection even if cached
        
        Returns:
            List of all available devices
        """
        return self._detector.detect_all(force_refresh)
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """Get devices of a specific type.
        
        Args:
            device_type: Type of device to get
        
        Returns:
            List of devices of the specified type
        """
        return self._detector.get_devices_by_type(device_type)
    
    def get_best_device(self) -> DeviceInfo:
        """Get the best available device for inference.
        
        Priority: CUDA > ROCm > XPU > CPU
        
        Returns:
            Best available device
        """
        return self._detector.get_best_device()
    
    def get_device(self, device_type: DeviceType, device_id: int = 0) -> Optional[DeviceInfo]:
        """Get a specific device by type and ID.
        
        Args:
            device_type: Type of device
            device_id: Device index (default 0)
        
        Returns:
            DeviceInfo if found, None otherwise
        """
        devices = self.get_devices_by_type(device_type)
        if device_id < len(devices):
            return devices[device_id]
        return None
    
    def is_available(self, device_type: DeviceType) -> bool:
        """Check if a device type is available.
        
        Args:
            device_type: Type of device to check
        
        Returns:
            True if device type is available
        """
        return self._detector.is_device_available(device_type)
    
    def get_system_info(self) -> SystemInfo:
        """Get system information.
        
        Returns:
            System information including OS, CPU, RAM, PyTorch version
        """
        return self._detector.get_system_info()
    
    def get_device_capabilities(self, device: DeviceInfo) -> Dict[str, Any]:
        """Get detailed capabilities for a device.
        
        Args:
            device: Device to get capabilities for
        
        Returns:
            Dictionary with capabilities (fp16, bf16, tensor_cores support)
        """
        return self._detector.get_device_capabilities(device)
    
    def set_current_device(self, device: DeviceInfo) -> None:
        """Set the current active device.
        
        This is a logical selection and does not modify the actual runtime.
        Used for tracking which device is being used for inference.
        
        Args:
            device: Device to set as current
        """
        self._current_device = device
        logger.info(f"Set current device: {device.display_name}")
    
    def get_current_device(self) -> Optional[DeviceInfo]:
        """Get the current active device.
        
        Returns:
            Current device if set, None otherwise
        """
        return self._current_device
    
    def get_gpu_devices(self) -> List[DeviceInfo]:
        """Get all GPU devices (CUDA + ROCm + XPU).
        
        Returns:
            List of all GPU devices
        """
        devices = []
        for device_type in [DeviceType.CUDA, DeviceType.ROCM, DeviceType.XPU]:
            devices.extend(self.get_devices_by_type(device_type))
        return devices
    
    def has_gpu(self) -> bool:
        """Check if any GPU is available.
        
        Returns:
            True if any GPU (CUDA, ROCm, or XPU) is available
        """
        return (
            self.is_available(DeviceType.CUDA) or
            self.is_available(DeviceType.ROCM) or
            self.is_available(DeviceType.XPU)
        )
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get a summary of all devices.
        
        Returns:
            Dictionary with device summary for display/logging
        """
        devices = self.get_devices()
        system_info = self.get_system_info()
        
        return {
            "system": {
                "os": f"{system_info.os_name} {system_info.os_version}",
                "cpu_count": system_info.cpu_count,
                "ram_gb": system_info.total_ram_gb,
                "python": system_info.python_version,
                "pytorch": system_info.pytorch_version,
                "cuda": system_info.cuda_version,
                "hip": system_info.hip_version,
            },
            "devices": [
                {
                    "type": d.device_type.value,
                    "name": d.display_name,
                    "memory_gb": d.memory_gb,
                    "compute_capability": d.compute_capability,
                }
                for d in devices
            ],
            "best_device": self.get_best_device().display_name,
            "has_gpu": self.has_gpu(),
        }
    
    def refresh(self) -> None:
        """Force refresh all device information."""
        self._detector.detect_all(force_refresh=True)
        logger.info("Device information refreshed")
    
    def resolve_device(self, device_str: str = "auto") -> str:
        """Resolve device string to actual device.
        
        Args:
            device_str: Device string ("auto", "cuda:0", "rocm:0", "xpu:0", "cpu")
        
        Returns:
            Resolved device string (e.g., "cuda:0", "rocm:0", "xpu:0", "cpu")
        """
        if device_str == "auto":
            best_device = self.get_best_device()
            
            if best_device.device_type == DeviceType.CUDA:
                return f"cuda:{best_device.device_id}"
            elif best_device.device_type == DeviceType.ROCM:
                return f"rocm:{best_device.device_id}"
            elif best_device.device_type == DeviceType.XPU:
                return f"xpu:{best_device.device_id}"
            else:
                return "cpu"
        
        return device_str
    
    def get_torch_device(self, device_str: str = "auto"):
        """Get torch.device object for the specified device.
        
        This method handles the mapping from device string to torch.device:
        - ROCm devices use "cuda" namespace in PyTorch
        - CUDA devices use "cuda" namespace
        - XPU devices use "xpu" namespace
        
        Args:
            device_str: Device string ("auto", "cuda:0", "rocm:0", "xpu:0", "cpu")
        
        Returns:
            torch.device instance
        """
        import torch
        
        resolved = self.resolve_device(device_str)
        
        # Handle ROCm (uses cuda namespace in PyTorch)
        if resolved.startswith("rocm:"):
            # ROCm devices use "cuda" namespace in PyTorch
            device_id = resolved.split(":")[1] if ":" in resolved else "0"
            return torch.device(f"cuda:{device_id}")
        
        return torch.device(resolved)
    
    def parse_device_id(self, device_str: str) -> int:
        """Parse device ID from device string.
        
        Args:
            device_str: Device string (e.g., "cuda:0", "rocm:1")
        
        Returns:
            Device ID (0 if not specified)
        """
        if ":" in device_str:
            try:
                return int(device_str.split(":")[1])
            except (ValueError, IndexError):
                return 0
        return 0


# Singleton instance
device_manager = DeviceManager()


# Convenience functions for backward compatibility
def get_available_devices() -> List[DeviceInfo]:
    """Get all available devices."""
    return device_manager.get_devices()


def get_best_device() -> DeviceInfo:
    """Get the best available device."""
    return device_manager.get_best_device()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return device_manager.is_available(DeviceType.CUDA)


def is_rocm_available() -> bool:
    """Check if ROCm is available."""
    return device_manager.is_available(DeviceType.ROCM)


def is_xpu_available() -> bool:
    """Check if XPU is available."""
    return device_manager.is_available(DeviceType.XPU)


def resolve_device(device_str: str = "auto") -> str:
    """Resolve device string to actual device."""
    return device_manager.resolve_device(device_str)


def get_torch_device(device_str: str = "auto"):
    """Get torch.device for the specified device."""
    return device_manager.get_torch_device(device_str)
