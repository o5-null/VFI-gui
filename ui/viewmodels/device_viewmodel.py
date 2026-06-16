"""DeviceViewModel for device/runtime monitoring.

Wraps DeviceManager and RuntimeManager to provide Qt-signals for UI binding.
Uses QTimer polling for device refresh (5-second interval).
"""

from dataclasses import dataclass
from typing import List

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from core.device_manager import device_manager, DeviceInfo
from core.runtime_manager import runtime_manager, RuntimeInfo
from core.device_type import DeviceType


@dataclass
class DeviceInfoVO:
    """Value object for device display.
    
    Used by UI components for device list/table.
    """
    name: str
    device_type: str
    vram_total_gb: float
    is_available: bool
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        return f"{self.name} ({self.device_type.upper()})"


class DeviceViewModel(QObject):
    """ViewModel for device/runtime monitoring.
    
    Signals:
        devices_changed: Available devices list changed
        runtime_changed: Current runtime changed
        current_device_changed: Current active device changed
    
    Inner dataclass:
        DeviceInfoVO: Device information value object
    
    Methods:
        start_polling(): Start periodic device refresh
        stop_polling(): Stop polling
        refresh(): Force immediate refresh
    
    Properties:
        current_runtime: Current runtime type string
        devices: List of DeviceInfoVO
        runtime_info: Current runtime info dict
    """
    
    # Signals
    devices_changed = pyqtSignal(list)  # List[DeviceInfoVO]
    runtime_changed = pyqtSignal(str)  # Runtime type string
    current_device_changed = pyqtSignal(str)  # Device name
    
    def __init__(
        self,
        device_mgr=None,  # Uses global device_manager if None
        runtime_mgr=None,  # Uses global runtime_manager if None
        parent=None,
    ):
        """Initialize DeviceViewModel.
        
        Args:
            device_mgr: DeviceManager instance (optional)
            runtime_mgr: RuntimeManager instance (optional)
            parent: Parent QObject
        """
        super().__init__(parent)
        self._device_manager = device_mgr or device_manager
        self._runtime_manager = runtime_mgr or runtime_manager
        
        # Private fields
        self._devices: List[DeviceInfoVO] = []
        self._current_runtime: str = ""
        self._current_device: str = ""
        self._force_refresh_next: bool = False
        
        # Polling timer (5-second interval)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(5000)  # 5 seconds
        self._poll_timer.timeout.connect(self._refresh)
        
        # Initial refresh
        self._refresh()
    
    # ====================
    # Properties
    # ====================
    
    @property
    def current_runtime(self) -> str:
        """Get current runtime type string."""
        return self._current_runtime
    
    @property
    def devices(self) -> List[DeviceInfoVO]:
        """Get available devices list."""
        return self._devices.copy()
    
    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return self._device_manager.has_gpu()
    
    @property
    def runtime_info(self) -> dict:
        """Get runtime info dict."""
        return self._runtime_manager.get_runtime_info()
    
    # ====================
    # Polling Control
    # ====================
    
    def start_polling(self) -> None:
        """Start periodic device refresh."""
        self._poll_timer.start()
    
    def stop_polling(self) -> None:
        """Stop polling."""
        self._poll_timer.stop()
    
    def refresh(self) -> None:
        """Force immediate refresh (public method).
        
        Triggers full device re-detection on next _refresh call.
        """
        self._force_refresh_next = True
        self._refresh()
    
    def _refresh(self) -> None:
        """Refresh device and runtime information.
        
        Uses cached device data by default. Only force_refresh=True
        (via public refresh()) triggers full re-detection.
        """
        # Refresh devices (use cache in polling, force only on manual refresh)
        raw_devices = self._device_manager.get_devices(force_refresh=self._force_refresh_next)
        self._force_refresh_next = False
        new_devices = [
            DeviceInfoVO(
                name=d.display_name,
                device_type=d.device_type.value,
                vram_total_gb=d.memory_gb,
                is_available=True,
            )
            for d in raw_devices
        ]
        
        if new_devices != self._devices:
            self._devices = new_devices
            self.devices_changed.emit(new_devices)
        
        # Refresh runtime
        current_runtime = self._runtime_manager.current_runtime
        runtime_str = current_runtime.value if current_runtime else "cpu"
        
        if runtime_str != self._current_runtime:
            self._current_runtime = runtime_str
            self.runtime_changed.emit(runtime_str)
        
        # Update current device
        best_device = self._device_manager.get_best_device()
        device_name = best_device.display_name if best_device else "CPU"
        
        if device_name != self._current_device:
            self._current_device = device_name
            self.current_device_changed.emit(device_name)
    
    # ====================
    # Query Methods
    # ====================
    
    def get_device_summary(self) -> dict:
        """Get device summary dict."""
        return self._device_manager.get_device_summary()
    
    def get_available_runtimes(self) -> List[RuntimeInfo]:
        """Get available runtimes list."""
        return self._runtime_manager.get_available_runtimes()
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._device_manager.is_available(DeviceType.CUDA)
    
    def is_xpu_available(self) -> bool:
        """Check if XPU is available."""
        return self._device_manager.is_available(DeviceType.XPU)
    
    def get_best_device(self) -> DeviceInfoVO:
        """Get the best available device.
        
        Returns:
            DeviceInfoVO for best device, or CPU if no GPU
        """
        best = self._device_manager.get_best_device()
        return DeviceInfoVO(
            name=best.display_name,
            device_type=best.device_type.value,
            vram_total_gb=best.memory_gb,
            is_available=True,
        )
    
    def get_gpu_devices(self) -> List[DeviceInfoVO]:
        """Get GPU devices only."""
        gpu_devices = self._device_manager.get_gpu_devices()
        return [
            DeviceInfoVO(
                name=d.display_name,
                device_type=d.device_type.value,
                vram_total_gb=d.memory_gb,
                is_available=True,
            )
            for d in gpu_devices
        ]
    
    def resolve_device(self, device_str: str = "auto") -> str:
        """Resolve device string to actual device.
        
        Args:
            device_str: Device string (auto, cuda:0, xpu:0, cpu)
            
        Returns:
            Resolved device string
        """
        return self._device_manager.resolve_device(device_str)


__all__ = ["DeviceViewModel", "DeviceInfoVO"]