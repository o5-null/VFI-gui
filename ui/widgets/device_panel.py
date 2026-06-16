"""DevicePanel — read-only device and runtime info display.

A QGroupBox showing device name, VRAM, runtime type, and precision.
Reads DeviceViewModel and updates when devices change.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QFormLayout
from PyQt6.QtCore import QEvent

from ui.styles.theme import Theme

if TYPE_CHECKING:
    from ui.viewmodels.device_viewmodel import DeviceViewModel


class DevicePanel(QGroupBox):
    """Device information panel — read-only display of GPU/Runtime status.

    Features:
    - Device name (e.g., NVIDIA RTX 3080)
    - VRAM total (e.g., 8 GB)
    - Runtime type (CUDA/XPU/CPU) with colored label
    - Precision (FP16/BF16/FP32)

    Refreshes when DeviceViewModel.devices_changed / runtime_changed fire.
    """

    def __init__(self, vm: "DeviceViewModel", parent=None):
        """Initialize DevicePanel.

        Args:
            vm: DeviceViewModel for state binding
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        self.setObjectName("devicePanel")
        self._setup_ui()
        self._bind_viewmodel()
        self._update_display()
        self.retranslate_ui()

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change for i18n."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(a0)

    def retranslate_ui(self) -> None:
        """Update all user-visible text for i18n."""
        self.setTitle(self.tr("Device Info"))
        if hasattr(self, "_device_name_label"):
            # Update label labels (the left side of form)
            pass  # Labels are set in _setup_ui with tr()

    def _setup_ui(self) -> None:
        """Create widget structure."""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(
            Theme.PADDING_MD, Theme.PADDING_MD,
            Theme.PADDING_MD, Theme.PADDING_MD
        )
        self._layout.setSpacing(Theme.SPACING_SM)

        self._form = QFormLayout()
        self._form.setSpacing(Theme.SPACING_SM)

        # Device name
        self._device_name_value = QLabel(self.tr("Detecting..."))
        self._device_name_value.setObjectName("deviceNameValue")
        self._form.addRow(self.tr("Device:"), self._device_name_value)

        # VRAM
        self._vram_value = QLabel(self.tr("N/A"))
        self._vram_value.setObjectName("vramValue")
        self._form.addRow(self.tr("VRAM:"), self._vram_value)

        # Runtime type
        self._runtime_value = QLabel(self.tr("CPU"))
        self._runtime_value.setObjectName("runtimeValue")
        self._form.addRow(self.tr("Runtime:"), self._runtime_value)

        # Precision
        self._precision_value = QLabel("FP16")
        self._precision_value.setObjectName("precisionValue")
        self._form.addRow(self.tr("Precision:"), self._precision_value)

        self._layout.addLayout(self._form)

    def _bind_viewmodel(self) -> None:
        """Connect to DeviceViewModel signals."""
        self._vm.devices_changed.connect(self._update_display)
        self._vm.runtime_changed.connect(self._update_display)
        self._vm.current_device_changed.connect(self._update_display)

    def _update_display(self) -> None:
        """Update device info display."""
        # Get current device
        best_device = self._vm.get_best_device()
        self._device_name_value.setText(best_device.name)

        # VRAM
        vram_gb = best_device.vram_total_gb
        if vram_gb > 0:
            self._vram_value.setText(f"{vram_gb:.1f} GB")
        else:
            self._vram_value.setText(self.tr("N/A"))

        # Runtime type with color
        runtime = self._vm.current_runtime.upper()
        self._runtime_value.setText(runtime)
        self._apply_runtime_color(runtime)

        # Precision (default to FP16 for GPU, FP32 for CPU)
        if self._vm.has_gpu:
            self._precision_value.setText("FP16")
        else:
            self._precision_value.setText("FP32")

    def _apply_runtime_color(self, runtime: str) -> None:
        """Apply color to runtime label based on type."""
        if runtime == "CUDA":
            color = Theme.ACCENT
        elif runtime == "XPU":
            color = Theme.SUCCESS
        else:
            color = Theme.TEXT_SECONDARY

        self._runtime_value.setStyleSheet(f"color: {color}; font-weight: bold;")


__all__ = ["DevicePanel"]