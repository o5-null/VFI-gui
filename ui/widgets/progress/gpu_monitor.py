"""HwMonitorPlaceholder — GPU monitor placeholder widget.

A placeholder widget for future GPU hardware monitoring.
Shows 'N/A' until GPUMonitorService is implemented.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QLabel
from PyQt6.QtCore import QEvent, Qt

from ui.styles.theme import Theme

if TYPE_CHECKING:
    from ui.viewmodels.task_viewmodel import TaskViewModel


class HwMonitorPlaceholder(QGroupBox):
    """GPU Monitor placeholder — currently shows 'N/A'.
    
    🔧 Future: Will display VRAM usage, GPU utilization, and temperature.
    GPUMonitorService not yet implemented — signals won't fire.
    
    Features:
        - QGroupBox titled "Hardware Monitor"
        - Labels: VRAM, GPU Util, Temp (all show N/A)
        - Grayed out appearance
        - Binds to TaskViewModel GPU signals (placeholder)
    """

    def __init__(self, vm: "TaskViewModel", parent=None):
        """Initialize HwMonitorPlaceholder.
        
        Args:
            vm: TaskViewModel for GPU data binding (signals won't fire yet)
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        self.setObjectName("hwMonitorPlaceholder")
        self._setup_ui()
        self._bind_viewmodel()
        self.retranslate_ui()

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change for i18n."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(a0)

    def retranslate_ui(self) -> None:
        """Update all user-visible text for i18n."""
        if hasattr(self, "_vram_label"):
            self.setTitle(self.tr("Hardware Monitor"))
            self._vram_label.setText(self.tr("VRAM: N/A"))
            self._gpu_util_label.setText(self.tr("GPU Util: N/A"))
            self._temp_label.setText(self.tr("Temp: N/A"))

    def _setup_ui(self) -> None:
        """Create widget structure."""
        self.setStyleSheet("QGroupBox { color: #5d5d5d; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_MD, Theme.PADDING_MD,
            Theme.PADDING_MD, Theme.PADDING_MD
        )
        layout.setSpacing(Theme.SPACING_MD)

        # VRAM label
        self._vram_label = QLabel()
        self._vram_label.setObjectName("hwMonitorVramLabel")
        self._vram_label.setStyleSheet(f"color: {Theme.TEXT_DISABLED};")
        layout.addWidget(self._vram_label)

        # GPU utilization label
        self._gpu_util_label = QLabel()
        self._gpu_util_label.setObjectName("hwMonitorGpuUtilLabel")
        self._gpu_util_label.setStyleSheet(f"color: {Theme.TEXT_DISABLED};")
        layout.addWidget(self._gpu_util_label)

        # Temperature label
        self._temp_label = QLabel()
        self._temp_label.setObjectName("hwMonitorTempLabel")
        self._temp_label.setStyleSheet(f"color: {Theme.TEXT_DISABLED};")
        layout.addWidget(self._temp_label)

        layout.addStretch()

    def _bind_viewmodel(self) -> None:
        """Connect to TaskViewModel GPU signals.
        
        Note: These signals are placeholders and won't fire until
        GPUMonitorService is implemented.
        """
        # GPU signals (placeholders - won't fire yet)
        self._vm.vram_used_changed.connect(self._on_vram_changed)
        self._vm.vram_total_changed.connect(self._on_vram_total_changed)
        self._vm.gpu_util_changed.connect(self._on_gpu_util_changed)
        self._vm.gpu_temp_changed.connect(self._on_gpu_temp_changed)

    def _on_vram_changed(self, vram: float) -> None:
        """Handle VRAM used change.
        
        Args:
            vram: VRAM used in GB
        """
        if vram > 0 and self._vm.vram_total > 0:
            self._vram_label.setText(
                self.tr("VRAM: {used:.1f} / {total:.1f} GB").format(
                    used=vram, total=self._vm.vram_total
                )
            )
            self._vram_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")

    def _on_vram_total_changed(self, total: float) -> None:
        """Handle VRAM total change.
        
        Args:
            total: Total VRAM in GB
        """
        if self._vm.vram_used > 0 and total > 0:
            self._vram_label.setText(
                self.tr("VRAM: {used:.1f} / {total:.1f} GB").format(
                    used=self._vm.vram_used, total=total
                )
            )
            self._vram_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")

    def _on_gpu_util_changed(self, util: float) -> None:
        """Handle GPU utilization change.
        
        Args:
            util: GPU utilization percentage
        """
        if util > 0:
            self._gpu_util_label.setText(self.tr("GPU Util: {util:.0f}%").format(util=util))
            self._gpu_util_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")

    def _on_gpu_temp_changed(self, temp: int) -> None:
        """Handle GPU temperature change.
        
        Args:
            temp: GPU temperature in Celsius
        """
        if temp > 0:
            self._temp_label.setText(self.tr("Temp: {temp}°C").format(temp=temp))
            self._temp_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")


__all__ = ["HwMonitorPlaceholder"]