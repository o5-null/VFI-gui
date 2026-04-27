"""Unified device type enumeration for VFI-gui.

This module provides a single source of truth for device types
across runtime management, device detection, and UI components.

Supported device types:
- CUDA: NVIDIA GPUs
- ROCM: AMD GPUs (via ROCm)
- XPU: Intel GPUs
- CPU: CPU fallback
"""

from enum import Enum


class DeviceType(Enum):
    """Supported device/runtime types.
    
    This unified enum replaces both RuntimeType and DeviceType
    from the previous separate implementations.
    
    Values:
        CUDA: NVIDIA GPU with CUDA support
        ROCM: AMD GPU with ROCm support
        XPU: Intel GPU with XPU support
        CPU: CPU-only fallback
    """
    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"
    CPU = "cpu"
    
    @classmethod
    def from_string(cls, value: str) -> "DeviceType":
        """Convert string to DeviceType.
        
        Args:
            value: String value ("cuda", "rocm", "xpu", "cpu")
        
        Returns:
            DeviceType enum value
        
        Raises:
            ValueError: If value is not a valid device type
        """
        value = value.lower()
        for dt in cls:
            if dt.value == value:
                return dt
        raise ValueError(f"Invalid device type: {value}")
    
    @property
    def is_gpu(self) -> bool:
        """Check if this device type is a GPU.
        
        Returns:
            True if device type is GPU (CUDA, ROCM, XPU), False for CPU
        """
        return self in (DeviceType.CUDA, DeviceType.ROCM, DeviceType.XPU)
    
    @property
    def display_name(self) -> str:
        """Get a display-friendly name.
        
        Returns:
            Human-readable device type name
        """
        names = {
            DeviceType.CUDA: "NVIDIA CUDA",
            DeviceType.ROCM: "AMD ROCm",
            DeviceType.XPU: "Intel XPU",
            DeviceType.CPU: "CPU",
        }
        return names[self]
    
    @property
    def vendor(self) -> str:
        """Get the vendor name for this device type.
        
        Returns:
            Vendor name (NVIDIA, AMD, Intel, or Unknown)
        """
        vendors = {
            DeviceType.CUDA: "NVIDIA",
            DeviceType.ROCM: "AMD",
            DeviceType.XPU: "Intel",
            DeviceType.CPU: "Unknown",
        }
        return vendors[self]


def get_device_type_priority(device_type: DeviceType) -> int:
    """Get priority for device type selection.
    
    Higher priority = preferred for auto-selection.
    
    Args:
        device_type: Device type to get priority for
    
    Returns:
        Priority value (higher is better)
    """
    priorities = {
        DeviceType.CUDA: 100,
        DeviceType.ROCM: 90,
        DeviceType.XPU: 80,
        DeviceType.CPU: 0,
    }
    return priorities.get(device_type, 0)


# Backward compatibility alias
# RuntimeType was previously defined in runtime_manager.py
RuntimeType = DeviceType
