"""Benchmark and Device Detection Dialog for VFI-gui.

Provides a dialog for:
- Viewing system and device information
- Running performance benchmarks
- Comparing device capabilities
"""

from typing import Optional, Dict, Any, List
import json

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QSplitter,
    QMessageBox,
    QCheckBox,
    QHeaderView,
    QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
from loguru import logger

from core import tr
from core.benchmark import DeviceDetector, BenchmarkRunner, BenchmarkConfig, DeviceType
from core.benchmark.device_detector import SystemInfo, DeviceInfo


class BenchmarkWorker(QThread):
    """Worker thread for running benchmarks without blocking UI."""
    
    progress = pyqtSignal(str, float)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, runner: BenchmarkRunner, config: BenchmarkConfig):
        super().__init__()
        self.runner = runner
        self.config = config
        self._is_cancelled = False
    
    def run(self):
        try:
            # Set up progress callback
            def progress_callback(message: str, progress: float):
                if not self._is_cancelled:
                    self.progress.emit(message, progress)
            
            self.runner.progress_callback = progress_callback
            result = self.runner.run(self.config)
            
            if not self._is_cancelled:
                self.result_ready.emit(result)
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            if not self._is_cancelled:
                self.error.emit(str(e))
    
    def cancel(self):
        self._is_cancelled = True
        self.runner.cancel()


class SystemInfoWidget(QWidget):
    """Widget displaying system information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # System info group
        sys_group = QGroupBox(tr("System Information"))
        sys_layout = QFormLayout(sys_group)
        sys_layout.setSpacing(8)
        
        self.os_label = QLabel(tr("Loading..."))
        self.cpu_label = QLabel(tr("Loading..."))
        self.ram_label = QLabel(tr("Loading..."))
        self.python_label = QLabel(tr("Loading..."))
        self.pytorch_label = QLabel(tr("Loading..."))
        self.cuda_version_label = QLabel(tr("Loading..."))
        
        sys_layout.addRow(tr("Operating System:"), self.os_label)
        sys_layout.addRow(tr("CPU Count:"), self.cpu_label)
        sys_layout.addRow(tr("Total RAM:"), self.ram_label)
        sys_layout.addRow(tr("Python Version:"), self.python_label)
        sys_layout.addRow(tr("PyTorch Version:"), self.pytorch_label)
        sys_layout.addRow(tr("CUDA Version:"), self.cuda_version_label)
        
        layout.addWidget(sys_group)
        
        # Refresh button
        refresh_btn = QPushButton(tr("Refresh"))
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
    
    def set_system_info(self, info: SystemInfo):
        """Update system information display."""
        self.os_label.setText(f"{info.os_name} {info.os_version}")
        self.cpu_label.setText(f"{info.cpu_count} cores")
        self.ram_label.setText(f"{info.total_ram_gb:.1f} GB")
        self.python_label.setText(info.python_version)
        self.pytorch_label.setText(info.pytorch_version or tr("Not installed"))
        self.cuda_version_label.setText(info.cuda_version or tr("N/A"))
    
    def refresh(self):
        """Refresh system information."""
        detector = DeviceDetector()
        info = detector.get_system_info()
        self.set_system_info(info)


class DeviceListWidget(QWidget):
    """Widget displaying list of available devices."""
    
    device_selected = pyqtSignal(DeviceInfo)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._devices: List[DeviceInfo] = []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header
        header = QLabel(tr("Available Devices"))
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)
        header.setFont(header_font)
        layout.addWidget(header)
        
        # Device table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            tr("Device"),
            tr("Type"),
            tr("Memory"),
            tr("Compute"),
            tr("Capabilities")
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        # Refresh button
        refresh_btn = QPushButton(tr("Refresh Device List"))
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)
    
    def set_devices(self, devices: List[DeviceInfo]):
        """Set the device list."""
        self._devices = devices
        self.table.setRowCount(len(devices))
        
        detector = DeviceDetector()
        
        for i, device in enumerate(devices):
            # Device name
            name_item = QTableWidgetItem(device.display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, device)
            self.table.setItem(i, 0, name_item)
            
            # Type
            type_item = QTableWidgetItem(device.device_type.value.upper())
            self.table.setItem(i, 1, type_item)
            
            # Memory
            memory_text = f"{device.memory_gb:.1f} GB" if device.total_memory_mb > 0 else "N/A"
            memory_item = QTableWidgetItem(memory_text)
            self.table.setItem(i, 2, memory_item)
            
            # Compute capability
            compute_text = device.compute_capability or "N/A"
            compute_item = QTableWidgetItem(compute_text)
            self.table.setItem(i, 3, compute_item)
            
            # Capabilities
            caps = detector.get_device_capabilities(device)
            caps_text = []
            if caps.get("supports_fp16"):
                caps_text.append("FP16")
            if caps.get("supports_bf16"):
                caps_text.append("BF16")
            if caps.get("supports_tensor_cores"):
                caps_text.append("Tensor")
            
            caps_item = QTableWidgetItem(", ".join(caps_text) if caps_text else "Standard")
            self.table.setItem(i, 4, caps_item)
        
        # Select first device
        if devices:
            self.table.selectRow(0)
    
    def _on_selection_changed(self):
        """Handle device selection change."""
        selected = self.table.selectedItems()
        if selected:
            row = selected[0].row()
            if 0 <= row < len(self._devices):
                self.device_selected.emit(self._devices[row])
    
    def refresh(self):
        """Refresh device list."""
        detector = DeviceDetector()
        devices = detector.get_all_devices()
        self.set_devices(devices)
    
    def get_selected_device(self) -> Optional[DeviceInfo]:
        """Get currently selected device."""
        selected = self.table.selectedItems()
        if selected:
            row = selected[0].row()
            if 0 <= row < len(self._devices):
                return self._devices[row]
        return None


class BenchmarkConfigWidget(QWidget):
    """Widget for configuring benchmark parameters."""
    
    config_changed = pyqtSignal(BenchmarkConfig)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QFormLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Test iterations
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 50)
        self.iterations_spin.setValue(10)
        self.iterations_spin.setSuffix(tr(" iterations"))
        layout.addRow(tr("Test Iterations:"), self.iterations_spin)
        
        # Warmup iterations
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 10)
        self.warmup_spin.setValue(3)
        self.warmup_spin.setSuffix(tr(" iterations"))
        layout.addRow(tr("Warmup Iterations:"), self.warmup_spin)
        
        # Data type
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["float32", "float16", "bfloat16"])
        self.dtype_combo.setCurrentText("float16")
        layout.addRow(tr("Data Type:"), self.dtype_combo)
        
        # Resolutions
        resolutions_group = QGroupBox(tr("Test Resolutions"))
        resolutions_layout = QVBoxLayout(resolutions_group)
        
        self.res_480p = QCheckBox(tr("640 x 480 (SD)"))
        self.res_480p.setChecked(True)
        resolutions_layout.addWidget(self.res_480p)
        
        self.res_720p = QCheckBox(tr("1280 x 720 (HD)"))
        self.res_720p.setChecked(True)
        resolutions_layout.addWidget(self.res_720p)
        
        self.res_1080p = QCheckBox(tr("1920 x 1080 (FHD)"))
        self.res_1080p.setChecked(True)
        resolutions_layout.addWidget(self.res_1080p)
        
        self.res_1440p = QCheckBox(tr("2560 x 1440 (QHD)"))
        self.res_1440p.setChecked(False)
        resolutions_layout.addWidget(self.res_1440p)
        
        self.res_4k = QCheckBox(tr("3840 x 2160 (4K)"))
        self.res_4k.setChecked(False)
        resolutions_layout.addWidget(self.res_4k)
        
        layout.addRow(resolutions_group)
    
    def get_config(self) -> BenchmarkConfig:
        """Get benchmark configuration from UI."""
        resolutions = []
        if self.res_480p.isChecked():
            resolutions.append((640, 480))
        if self.res_720p.isChecked():
            resolutions.append((1280, 720))
        if self.res_1080p.isChecked():
            resolutions.append((1920, 1080))
        if self.res_1440p.isChecked():
            resolutions.append((2560, 1440))
        if self.res_4k.isChecked():
            resolutions.append((3840, 2160))
        
        return BenchmarkConfig(
            test_iterations=self.iterations_spin.value(),
            warmup_iterations=self.warmup_spin.value(),
            dtype=self.dtype_combo.currentText(),
            test_resolutions=resolutions if resolutions else [(1280, 720)],
        )


class BenchmarkResultsWidget(QWidget):
    """Widget displaying benchmark results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            tr("Resolution"),
            tr("FPS"),
            tr("Avg Time"),
            tr("Min/Max"),
            tr("Memory"),
            tr("Status")
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        # Summary
        summary_group = QGroupBox(tr("Summary"))
        summary_layout = QFormLayout(summary_group)
        
        self.best_res_label = QLabel(tr("-"))
        self.best_fps_label = QLabel(tr("-"))
        self.recommended_batch_label = QLabel(tr("-"))
        
        summary_layout.addRow(tr("Best Resolution:"), self.best_res_label)
        summary_layout.addRow(tr("Best FPS:"), self.best_fps_label)
        summary_layout.addRow(tr("Recommended Batch Size:"), self.recommended_batch_label)
        
        layout.addWidget(summary_group)
        
        # Export button
        export_btn = QPushButton(tr("Export Results"))
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)
    
    def set_results(self, result):
        """Set benchmark results."""
        from core.benchmark.benchmark_runner import BenchmarkResult
        
        results = result.resolution_results
        self.table.setRowCount(len(results))
        
        for i, res in enumerate(results):
            # Resolution
            res_text = f"{res.width}x{res.height}"
            self.table.setItem(i, 0, QTableWidgetItem(res_text))
            
            # FPS
            fps_text = f"{res.fps:.1f}" if res.success else "-"
            fps_item = QTableWidgetItem(fps_text)
            if res.success and res.fps >= 30:
                fps_item.setForeground(QColor("#4CAF50"))  # Green
            elif res.success and res.fps >= 10:
                fps_item.setForeground(QColor("#FFC107"))  # Yellow
            elif res.success:
                fps_item.setForeground(QColor("#F44336"))  # Red
            self.table.setItem(i, 1, fps_item)
            
            # Avg time
            time_text = f"{res.avg_inference_time_ms:.1f}ms" if res.success else "-"
            self.table.setItem(i, 2, QTableWidgetItem(time_text))
            
            # Min/Max
            minmax_text = f"{res.min_inference_time_ms:.0f}/{res.max_inference_time_ms:.0f}ms" if res.success else "-"
            self.table.setItem(i, 3, QTableWidgetItem(minmax_text))
            
            # Memory
            memory_text = f"{res.peak_memory_mb:.0f}MB" if res.success else "-"
            self.table.setItem(i, 4, QTableWidgetItem(memory_text))
            
            # Status
            status_text = "✓ Success" if res.success else f"✗ {res.error_message}"
            self.table.setItem(i, 5, QTableWidgetItem(status_text))
        
        # Update summary
        if result.best_resolution:
            self.best_res_label.setText(f"{result.best_resolution[0]}x{result.best_resolution[1]}")
            self.best_fps_label.setText(f"{result.best_fps:.1f} FPS")
        else:
            self.best_res_label.setText(tr("None"))
            self.best_fps_label.setText(tr("-"))
        
        self.recommended_batch_label.setText(str(result.recommended_batch_size))
        
        self._current_result = result
    
    def _export_results(self):
        """Export benchmark results to JSON."""
        if not hasattr(self, '_current_result') or self._current_result is None:
            QMessageBox.warning(self, tr("No Results"), tr("No benchmark results to export."))
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            tr("Export Results"),
            "benchmark_results.json",
            tr("JSON Files (*.json)")
        )
        
        if filename:
            try:
                result = self._current_result
                data = {
                    "timestamp": result.end_time,
                    "device": result.device_info.display_name if result.device_info else "Unknown",
                    "config": {
                        "test_iterations": result.config.test_iterations,
                        "warmup_iterations": result.config.warmup_iterations,
                        "dtype": result.config.dtype,
                    },
                    "results": [
                        {
                            "resolution": f"{r.width}x{r.height}",
                            "fps": round(r.fps, 2) if r.success else None,
                            "avg_time_ms": round(r.avg_inference_time_ms, 2) if r.success else None,
                            "peak_memory_mb": round(r.peak_memory_mb, 2) if r.success else None,
                            "success": r.success,
                            "error": r.error_message,
                        }
                        for r in result.resolution_results
                    ],
                    "summary": {
                        "best_resolution": result.best_resolution,
                        "best_fps": round(result.best_fps, 2),
                        "recommended_batch_size": result.recommended_batch_size,
                    }
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                QMessageBox.information(self, tr("Export Complete"), tr("Results exported successfully."))
            except Exception as e:
                QMessageBox.critical(self, tr("Export Failed"), str(e))


class BenchmarkDialog(QDialog):
    """Dialog for device detection and benchmarking.
    
    Provides a comprehensive interface for:
    - Viewing system information
    - Detecting available devices (CUDA, XPU, CPU)
    - Running performance benchmarks
    - Viewing and exporting benchmark results
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[BenchmarkWorker] = None
        self._setup_ui()
        self._load_initial_data()
        
        self.setWindowTitle(tr("Device Detection & Benchmark"))
        self.setMinimumSize(900, 700)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # System Info Tab
        self.system_info_widget = SystemInfoWidget()
        self.tabs.addTab(self.system_info_widget, tr("System Info"))
        
        # Devices Tab
        devices_widget = QWidget()
        devices_layout = QVBoxLayout(devices_widget)
        self.device_list = DeviceListWidget()
        self.device_list.device_selected.connect(self._on_device_selected)
        devices_layout.addWidget(self.device_list)
        
        # Device details
        details_group = QGroupBox(tr("Device Details"))
        details_layout = QFormLayout(details_group)
        self.selected_device_label = QLabel(tr("Select a device to view details"))
        details_layout.addRow(tr("Selected:"), self.selected_device_label)
        devices_layout.addWidget(details_group)
        
        self.tabs.addTab(devices_widget, tr("Devices"))
        
        # Benchmark Tab
        benchmark_widget = QWidget()
        benchmark_layout = QHBoxLayout(benchmark_widget)
        
        # Left panel: configuration
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        
        self.config_widget = BenchmarkConfigWidget()
        config_layout.addWidget(self.config_widget)
        
        # Run buttons
        btn_layout = QHBoxLayout()
        
        self.quick_test_btn = QPushButton(tr("Quick Test"))
        self.quick_test_btn.clicked.connect(self._run_quick_test)
        btn_layout.addWidget(self.quick_test_btn)
        
        self.run_benchmark_btn = QPushButton(tr("Run Full Benchmark"))
        self.run_benchmark_btn.clicked.connect(self._run_benchmark)
        btn_layout.addWidget(self.run_benchmark_btn)
        
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self._cancel_benchmark)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        
        config_layout.addLayout(btn_layout)
        config_layout.addStretch()
        
        benchmark_layout.addWidget(config_panel)
        
        # Right panel: progress and results
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        # Progress bar
        self.progress_group = QGroupBox(tr("Progress"))
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel(tr("Ready"))
        progress_layout.addWidget(self.progress_label)
        
        results_layout.addWidget(self.progress_group)
        
        # Results widget
        self.results_widget = BenchmarkResultsWidget()
        results_layout.addWidget(self.results_widget)
        
        benchmark_layout.addWidget(results_panel)
        benchmark_layout.setStretchFactor(config_panel, 1)
        benchmark_layout.setStretchFactor(results_panel, 2)
        
        self.tabs.addTab(benchmark_widget, tr("Benchmark"))
        
        layout.addWidget(self.tabs)
        
        # Close button
        close_btn = QPushButton(tr("Close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def _load_initial_data(self):
        """Load initial data."""
        # Load system info
        detector = DeviceDetector()
        system_info = detector.get_system_info()
        self.system_info_widget.set_system_info(system_info)
        
        # Load devices
        devices = detector.get_all_devices()
        self.device_list.set_devices(devices)
    
    def _on_device_selected(self, device: DeviceInfo):
        """Handle device selection."""
        self.selected_device_label.setText(device.display_name)
    
    def _run_quick_test(self):
        """Run a quick benchmark test."""
        device = self.device_list.get_selected_device()
        if not device:
            QMessageBox.warning(self, tr("No Device"), tr("Please select a device first."))
            return
        
        config = BenchmarkConfig(
            warmup_iterations=1,
            test_iterations=3,
            test_resolutions=[(1280, 720), (1920, 1080)],
            device_type=device.device_type,
            device_id=device.device_id,
        )
        
        self._start_benchmark(config)
    
    def _run_benchmark(self):
        """Run full benchmark."""
        device = self.device_list.get_selected_device()
        if not device:
            QMessageBox.warning(self, tr("No Device"), tr("Please select a device first."))
            return
        
        config = self.config_widget.get_config()
        config.device_type = device.device_type
        config.device_id = device.device_id
        
        self._start_benchmark(config)
    
    def _start_benchmark(self, config: BenchmarkConfig):
        """Start benchmark with given configuration."""
        runner = BenchmarkRunner()
        self._worker = BenchmarkWorker(runner, config)
        
        self._worker.progress.connect(self._on_benchmark_progress)
        self._worker.result_ready.connect(self._on_benchmark_complete)
        self._worker.error.connect(self._on_benchmark_error)
        self._worker.finished.connect(self._on_benchmark_finished)
        
        self.run_benchmark_btn.setEnabled(False)
        self.quick_test_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(tr("Starting benchmark..."))
        
        self._worker.start()
    
    def _cancel_benchmark(self):
        """Cancel running benchmark."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.progress_label.setText(tr("Cancelling..."))
            self._worker.wait(5000)  # Wait up to 5 seconds
            if self._worker.isRunning():
                self._worker.terminate()
    
    def _on_benchmark_progress(self, message: str, progress: float):
        """Handle benchmark progress update."""
        self.progress_label.setText(message)
        self.progress_bar.setValue(int(progress))
    
    def _on_benchmark_complete(self, result):
        """Handle benchmark completion."""
        self.results_widget.set_results(result)
        self.tabs.setCurrentIndex(2)  # Switch to results tab
        QMessageBox.information(self, tr("Benchmark Complete"), 
                               tr("Benchmark completed successfully!"))
    
    def _on_benchmark_error(self, error_message: str):
        """Handle benchmark error."""
        QMessageBox.critical(self, tr("Benchmark Failed"), error_message)
    
    def _on_benchmark_finished(self):
        """Handle benchmark thread finished."""
        self.run_benchmark_btn.setEnabled(True)
        self.quick_test_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._worker = None
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                tr("Benchmark Running"),
                tr("A benchmark is currently running. Do you want to cancel it?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._cancel_benchmark()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
