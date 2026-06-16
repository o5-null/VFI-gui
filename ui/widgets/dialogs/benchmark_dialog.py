"""Benchmark dialog for VFI-gui.

A dialog for running performance benchmarks with device, model, and resolution selection.
Results are displayed in a read-only text area.

Uses QThread to run BenchmarkRunner without blocking the UI.
"""

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QLabel,
    QWidget,
    QSpinBox,
)
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal

from loguru import logger

from core.model_selection import ModelSelectionManager
from core.model_manager import CheckpointInfo, ModelTypeInfo


# Resolution preset index -> (width, height)
_RESOLUTION_MAP = {
    0: (1280, 720),
    1: (1920, 1080),
    2: (3840, 2160),
}


class _BenchmarkWorker(QThread):
    """Worker thread for running benchmark without blocking UI."""

    progress = pyqtSignal(str, float)  # (message, percent)
    finished = pyqtSignal(object)      # BenchmarkResult or Exception

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._runner = None

    def run(self):
        try:
            from core.benchmark.benchmark_runner import BenchmarkRunner

            self._runner = BenchmarkRunner(progress_callback=self._emit_progress)
            result = self._runner.run(self._config)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Benchmark worker error: {e}", exc_info=True)
            self.finished.emit(e)

    def cancel(self):
        """Request cancellation through the runner."""
        if self._runner is not None:
            self._runner.cancel()

    def _emit_progress(self, message: str, progress: float):
        self.progress.emit(message, progress)


class BenchmarkDialog(QDialog):
    """Dialog for running performance benchmarks.

    Features:
    - Device selection (auto-detect available GPUs)
    - Model selection (RIFE, FILM, AMT, etc.)
    - Resolution presets (720p, 1080p, 4K)
    - Thread count configuration
    - Progress bar for benchmark status
    - Results display area

    Actual benchmark logic is in core layer, this dialog provides UI.
    """

    def __init__(self, model_selection: ModelSelectionManager, parent=None):
        super().__init__(parent)
        self._model_selection = model_selection
        self._worker = None
        self._setup_ui()
        self._connect_signals()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    def retranslate_ui(self):
        """Update all user-visible text for i18n."""
        self.setWindowTitle(self.tr("Benchmark"))

        self._device_label.setText(self.tr("Device:"))
        self._model_type_label.setText(self.tr("Model Type:"))
        self._checkpoint_label.setText(self.tr("Checkpoint:"))
        self._resolution_label.setText(self.tr("Resolution:"))
        self._threads_label.setText(self.tr("Threads:"))
        self._threads_spin.setSpecialValueText(self.tr("Auto (default)"))
        self._run_button.setText(self.tr("Run Benchmark"))

        # Re-populate device combo
        device_idx = self._device_combo.currentIndex()
        self._device_combo.clear()
        self._populate_devices()
        self._device_combo.setCurrentIndex(device_idx)

        # Re-populate model type combo (preserving selection)
        model_type_idx = self._model_type_combo.currentIndex()
        self._populate_model_types()
        if model_type_idx >= 0 and model_type_idx < self._model_type_combo.count():
            self._model_type_combo.setCurrentIndex(model_type_idx)

        # Re-populate checkpoint combo (preserving selection)
        checkpoint_idx = self._checkpoint_combo.currentIndex()
        self._populate_checkpoints()
        if checkpoint_idx >= 0 and checkpoint_idx < self._checkpoint_combo.count():
            self._checkpoint_combo.setCurrentIndex(checkpoint_idx)

        res_idx = self._resolution_combo.currentIndex()
        self._resolution_combo.clear()
        self._resolution_combo.addItems([
            self.tr("720p (1280×720)"),
            self.tr("1080p (1920×1080)"),
            self.tr("4K (3840×2160)"),
        ])
        self._resolution_combo.setCurrentIndex(res_idx)

        self._results_text.setPlaceholderText(
            self.tr("Click 'Run Benchmark' to start performance testing")
        )

    def _setup_ui(self):
        """Create widgets and layout."""
        self.setWindowTitle(self.tr("Benchmark"))
        self.setMinimumSize(560, 480)

        layout = QVBoxLayout(self)

        # Settings section
        settings_widget = QWidget()
        form = QFormLayout(settings_widget)

        # Device selection
        self._device_label = QLabel()
        self._device_combo = QComboBox()
        self._populate_devices()
        form.addRow(self._device_label, self._device_combo)

        # Model type selection
        self._model_type_label = QLabel()
        self._model_type_combo = QComboBox()
        self._populate_model_types()
        form.addRow(self._model_type_label, self._model_type_combo)

        # Checkpoint/version selection
        self._checkpoint_label = QLabel()
        self._checkpoint_combo = QComboBox()
        self._populate_checkpoints()
        form.addRow(self._checkpoint_label, self._checkpoint_combo)

        # Resolution selection
        self._resolution_label = QLabel()
        self._resolution_combo = QComboBox()
        self._resolution_combo.addItems([
            self.tr("720p (1280×720)"),
            self.tr("1080p (1920×1080)"),
            self.tr("4K (3840×2160)"),
        ])
        form.addRow(self._resolution_label, self._resolution_combo)

        # Thread count
        self._threads_label = QLabel()
        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(0, 128)
        self._threads_spin.setValue(0)
        self._threads_spin.setSpecialValueText(self.tr("Auto (default)"))
        self._threads_spin.setToolTip(
            self.tr("Number of CPU threads for inference.\n0 = use all logical cores (default).")
        )
        form.addRow(self._threads_label, self._threads_spin)

        layout.addWidget(settings_widget)

        # Run / Cancel button
        button_layout = QHBoxLayout()
        self._run_button = QPushButton()
        self._cancel_button = QPushButton(self.tr("Cancel"))
        self._cancel_button.setVisible(False)
        button_layout.addWidget(self._run_button)
        button_layout.addWidget(self._cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        # Results area
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlaceholderText(
            self.tr("Click 'Run Benchmark' to start performance testing")
        )
        layout.addWidget(self._results_text)

        # Close button
        self._close_button = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        layout.addWidget(self._close_button)

        self.retranslate_ui()

    def _populate_devices(self):
        """Populate device combo with available devices."""
        self._device_combo.addItem(self.tr("Auto (Best Available)"))

        try:
            from core.device_manager import device_manager

            # Add GPU devices
            for device in device_manager.get_gpu_devices():
                self._device_combo.addItem(device.display_name)

            # Add CPU fallback
            self._device_combo.addItem("CPU")
        except ImportError:
            # If device_manager not available, add basic options
            self._device_combo.addItem("CUDA")
            self._device_combo.addItem("CPU")

    def _populate_model_types(self):
        """Populate model type combo with available model types from ModelSelectionManager."""
        self._model_type_combo.blockSignals(True)
        self._model_type_combo.clear()

        available_types = self._model_selection.get_available_model_types()

        if not available_types:
            # No installed models — show all types as reference
            all_types = self._model_selection.get_all_model_types()
            for type_info in all_types:
                self._model_type_combo.addItem(
                    type_info.display_name, type_info.name
                )
        else:
            for type_info in available_types:
                self._model_type_combo.addItem(
                    type_info.display_name, type_info.name
                )

        self._model_type_combo.blockSignals(False)

        # Populate checkpoints for the first (or current) model type
        self._populate_checkpoints()

    def _populate_checkpoints(self):
        """Populate checkpoint combo for the currently selected model type."""
        self._checkpoint_combo.blockSignals(True)
        self._checkpoint_combo.clear()

        model_type = self._get_current_model_type()
        if not model_type:
            self._checkpoint_combo.blockSignals(False)
            return

        checkpoints = self._model_selection.get_available_checkpoints(model_type)

        for ckpt in checkpoints:
            # Show checkpoint display name, store checkpoint name
            self._checkpoint_combo.addItem(ckpt.display_name, ckpt.name)

        self._checkpoint_combo.blockSignals(False)

    def _get_current_model_type(self) -> str:
        """Get the currently selected model type string."""
        idx = self._model_type_combo.currentIndex()
        if idx >= 0:
            return self._model_type_combo.itemData(idx) or ""
        return ""

    def _get_current_checkpoint_name(self) -> str:
        """Get the currently selected checkpoint filename."""
        idx = self._checkpoint_combo.currentIndex()
        if idx >= 0:
            return self._checkpoint_combo.itemData(idx) or ""
        return ""

    def _on_model_type_changed(self, index: int):
        """Handle model type combo selection change — refresh checkpoints."""
        self._populate_checkpoints()

    def _connect_signals(self):
        """Connect button signals."""
        self._run_button.clicked.connect(self._on_run_benchmark)
        self._cancel_button.clicked.connect(self._on_cancel)
        self._close_button.rejected.connect(self.reject)
        self._model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)

    # ====================
    # Benchmark Execution
    # ====================

    def _build_config(self):
        """Build BenchmarkConfig from current UI selections.
        
        Device is pre-resolved in the main thread to avoid calling
        torch.xpu/cuda.is_available() inside the QThread worker,
        which can fail or return wrong results.
        """
        from core.benchmark.benchmark_runner import BenchmarkConfig, BenchmarkMode
        from core.device_type import DeviceType

        # Model — from dynamic comboboxes
        model_type = self._get_current_model_type()
        checkpoint_name = self._get_current_checkpoint_name()

        if not model_type or not checkpoint_name:
            raise ValueError(self.tr("No model selected. Please install a model first."))

        # Resolution
        res_idx = self._resolution_combo.currentIndex()
        resolution = _RESOLUTION_MAP.get(res_idx, (1920, 1080))

        # Device — resolve NOW in the main thread
        device_idx = self._device_combo.currentIndex()
        resolved_device = None
        try:
            from core.device_manager import device_manager

            gpu_devices = device_manager.get_gpu_devices()
            if device_idx == 0:
                # Auto — get best device now
                resolved_device = device_manager.get_best_device()
            elif device_idx - 1 < len(gpu_devices):
                resolved_device = gpu_devices[device_idx - 1]
            else:
                # CPU option is last
                resolved_device = device_manager.get_devices_by_type(DeviceType.CPU)[0]
        except (ImportError, IndexError):
            pass

        return BenchmarkConfig(
            mode=BenchmarkMode.SINGLE,
            warmup_iterations=3,
            sequence_repetitions=3,
            test_resolutions=[resolution],
            model_type=model_type,
            checkpoint_name=checkpoint_name,
            num_threads=self._threads_spin.value(),
            resolved_device=resolved_device,
        )

    def _on_run_benchmark(self):
        """Handle run benchmark button click."""
        try:
            config = self._build_config()
        except Exception as e:
            logger.error(f"Failed to build benchmark config: {e}")
            self._results_text.setText(self.tr("Configuration error: {}").format(e))
            return

        # Update UI state
        self._set_running(True)
        self._progress_bar.setValue(0)
        self._results_text.clear()

        # Create and start worker
        self._worker = _BenchmarkWorker(config, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_cancel(self):
        """Cancel running benchmark gracefully."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._results_text.append(self.tr("Cancelling..."))

    def _on_progress(self, message: str, percent: float):
        """Handle progress update from worker."""
        self._progress_bar.setValue(int(percent))
        self._results_text.append(f"[{percent:5.1f}%] {message}")

    def _on_finished(self, result):
        """Handle benchmark completion."""
        self._set_running(False)

        if isinstance(result, Exception):
            self._progress_bar.setValue(0)
            self._results_text.append(
                self.tr("\nBenchmark failed: {}").format(result)
            )
            return

        # Display results
        self._progress_bar.setValue(100)
        self._results_text.append(self._format_results(result))

    def _set_running(self, running: bool):
        """Toggle UI between running/idle states."""
        self._run_button.setEnabled(not running)
        self._cancel_button.setVisible(running)
        self._device_combo.setEnabled(not running)
        self._model_type_combo.setEnabled(not running)
        self._checkpoint_combo.setEnabled(not running)
        self._resolution_combo.setEnabled(not running)
        self._threads_spin.setEnabled(not running)

    def _format_results(self, result) -> str:
        """Format BenchmarkResult into readable text."""
        lines = []
        lines.append("=" * 50)
        lines.append(self.tr("Benchmark Results"))
        lines.append("=" * 50)

        # Device info
        if result.device_info:
            lines.append(
                self.tr("Device: {}").format(result.device_info.display_name)
            )

        # Config info
        cfg = result.config
        lines.append(self.tr("Model: {} ({})").format(cfg.model_type, cfg.checkpoint_name))
        if cfg.num_threads > 0:
            lines.append(self.tr("Threads: {}").format(cfg.num_threads))

        lines.append("")

        # Resolution results
        for res in result.resolution_results:
            if res.success:
                lines.append(
                    self.tr("  {}×{}: {:.1f} FPS").format(
                        res.width, res.height, res.fps
                    )
                )
                lines.append(
                    self.tr("    Avg: {:.1f}ms | Min: {:.1f}ms | Max: {:.1f}ms").format(
                        res.avg_inference_time_ms,
                        res.min_inference_time_ms,
                        res.max_inference_time_ms,
                    )
                )
                lines.append(
                    self.tr("    First frame: {:.1f}ms | Peak memory: {:.0f}MB").format(
                        res.first_frame_ms, res.peak_memory_mb
                    )
                )
            else:
                lines.append(
                    self.tr("  {}×{}: FAILED - {}").format(
                        res.width, res.height, res.error_message
                    )
                )

        # Summary
        lines.append("")
        if result.best_fps > 0:
            lines.append(
                self.tr("Best: {:.1f} FPS at {}").format(
                    result.best_fps, result.best_resolution
                )
            )
        lines.append(
            self.tr("Duration: {:.1f}s").format(result.total_duration)
        )

        return "\n".join(lines)

    def get_num_threads(self) -> int:
        """Get the number of threads selected by user.

        Returns:
            0 for auto (PyTorch default), or specific thread count
        """
        return self._threads_spin.value()

    def reject(self):
        """Override reject to clean up worker on close."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        super().reject()
