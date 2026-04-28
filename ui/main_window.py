"""VFI-gui Main Window."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMenuBar,
    QMenu,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QApplication,
    QTabWidget,
    QDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QCloseEvent
from loguru import logger

from core import (
    Config,
    Processor,
    VideoProcessor,  # Backward compatibility
    QueueManager,
    ModelManager,
    tr,
    get_i18n,
    BackendType,
    BackendConfig,
    ProcessingConfig,
)
from core.model_selection import ModelSelectionManager
from core.task_orchestrator import TaskOrchestrator
from ui.controllers.processing_controller import ProcessingController
from ui.widgets.video_input import VideoInputWidget
from ui.widgets.pipeline_config import PipelineConfigWidget
from ui.widgets.progress_panel import ProgressPanel
from ui.widgets.batch_queue import BatchQueueWidget
from ui.widgets.model_manager_panel import ModelManagerPanel
from ui.widgets.dependency_panel import DependencyPanel
from ui.widgets.benchmark_dialog import BenchmarkDialog


class MainWindow(QMainWindow):
    """Main application window for VFI-gui."""

    # Signals
    video_selected = pyqtSignal(str)
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()
    processing_cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()

        logger.info("Initializing MainWindow")

        # Initialize components - use global config provider for consistency
        from core import get_config
        self.config = get_config()
        self.processor: Optional[Processor] = None  # Deprecated - kept for backward compat
        self.queue_manager = QueueManager()
        # Create shared model manager (single source of truth)
        self.model_manager = ModelManager(self.config)

        # NEW: Create TaskOrchestrator and ProcessingController
        self._orchestrator = TaskOrchestrator(self.config)
        self._controller = ProcessingController(self._orchestrator, self)
        
        # Create shared model selection manager using the same model manager
        self.model_selection_manager = ModelSelectionManager(
            self.config, 
            model_manager=self.model_manager
        )

        # Setup UI
        self._setup_window()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_statusbar()
        self._setup_connections()

        # Load saved settings
        self._load_settings()
        
        # Apply saved language preference
        self._apply_language_preference()
        
        logger.info("MainWindow initialized")

    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle(tr("VFI-gui - Video Frame Interpolation"))
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Restore window geometry if saved
        geometry = self.config.get("window.geometry")
        if geometry:
            self.restoreGeometry(bytes.fromhex(geometry))

    def _setup_menubar(self):
        """Create application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu(tr("&File"))

        open_action = QAction(tr("&Open Video..."), self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_video)
        file_menu.addAction(open_action)

        open_folder_action = QAction(tr("Open &Folder..."), self)
        open_folder_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_folder_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        add_to_queue_action = QAction(tr("&Add to Queue"), self)
        add_to_queue_action.setShortcut(QKeySequence("Ctrl+Shift+A"))
        add_to_queue_action.triggered.connect(self._on_add_to_queue)
        file_menu.addAction(add_to_queue_action)

        file_menu.addSeparator()

        save_settings_action = QAction(tr("&Save Settings"), self)
        save_settings_action.setShortcut(QKeySequence.StandardKey.Save)
        save_settings_action.triggered.connect(self._on_save_settings)
        file_menu.addAction(save_settings_action)

        file_menu.addSeparator()

        exit_action = QAction(tr("E&xit"), self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu(tr("&Edit"))

        clear_queue_action = QAction(tr("&Clear Queue"), self)
        clear_queue_action.triggered.connect(self._on_clear_queue)
        edit_menu.addAction(clear_queue_action)

        # Language menu
        language_menu = menubar.addMenu(tr("&Language"))
        self._setup_language_menu(language_menu)

        # Tools menu
        tools_menu = menubar.addMenu(tr("&Tools"))

        models_action = QAction(tr("&Model Manager..."), self)
        models_action.triggered.connect(self._on_model_manager)
        tools_menu.addAction(models_action)

        tools_menu.addSeparator()

        benchmark_action = QAction(tr("&Device Detection & Benchmark..."), self)
        benchmark_action.triggered.connect(self._on_benchmark)
        tools_menu.addAction(benchmark_action)

        tools_menu.addSeparator()

        proxy_action = QAction(tr("&Proxy Settings..."), self)
        proxy_action.triggered.connect(self._on_proxy_settings)
        tools_menu.addAction(proxy_action)

        performance_action = QAction(tr("&Performance Settings..."), self)
        performance_action.triggered.connect(self._on_performance_settings)
        tools_menu.addAction(performance_action)

        # Help menu
        help_menu = menubar.addMenu(tr("&Help"))

        about_action = QAction(tr("&About"), self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_language_menu(self, menu: QMenu):
        """Setup language selection menu."""
        from PyQt6.QtGui import QActionGroup
        i18n = get_i18n()
        current_lang = i18n.get_current_language()
        
        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)
        
        for lang_code, lang_name in i18n.SUPPORTED_LANGUAGES.items():
            action = QAction(lang_name, self)
            action.setData(lang_code)
            action.setCheckable(True)
            action.setChecked(lang_code == current_lang)
            action.triggered.connect(lambda checked, code=lang_code: self._on_language_changed(code))
            lang_group.addAction(action)
            menu.addAction(action)

    def _setup_toolbar(self):
        """Create application toolbar."""
        toolbar = QToolBar(tr("Main Toolbar"))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Open action
        open_action = QAction(tr("Open"), self)
        open_action.setStatusTip(tr("Open a video file"))
        open_action.triggered.connect(self._on_open_video)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # Start action
        self.start_action = QAction(tr("Start"), self)
        self.start_action.setStatusTip(tr("Start processing"))
        self.start_action.triggered.connect(self._on_start_processing)
        toolbar.addAction(self.start_action)

        # Pause action
        self.pause_action = QAction(tr("Pause"), self)
        self.pause_action.setStatusTip(tr("Pause processing"))
        self.pause_action.setEnabled(False)
        self.pause_action.triggered.connect(self._on_pause_processing)
        toolbar.addAction(self.pause_action)

        # Cancel action
        self.cancel_action = QAction(tr("Cancel"), self)
        self.cancel_action.setStatusTip(tr("Cancel processing"))
        self.cancel_action.setEnabled(False)
        self.cancel_action.triggered.connect(self._on_cancel_processing)
        toolbar.addAction(self.cancel_action)

        toolbar.addSeparator()

        # Add to queue
        add_queue_action = QAction(tr("Add to Queue"), self)
        add_queue_action.setStatusTip(tr("Add current video to batch queue"))
        add_queue_action.triggered.connect(self._on_add_to_queue)
        toolbar.addAction(add_queue_action)

    def _setup_central_widget(self):
        """Create the main central widget layout with tab structure."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3d3d3d;
            }
        """)

        # Create tabs
        self._create_processing_tab()
        self._create_models_tab()
        self._create_dependencies_tab()

        main_layout.addWidget(self.tab_widget)

    def _create_processing_tab(self):
        """Create the main processing tab."""
        processing_widget = QWidget()
        processing_layout = QVBoxLayout(processing_widget)
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_layout.setSpacing(0)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Video input and pipeline config
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # Video input widget
        self.video_input = VideoInputWidget()
        left_layout.addWidget(self.video_input)

        # Pipeline configuration widget (using shared model selection manager)
        self.pipeline_config = PipelineConfigWidget(self.config, self.model_selection_manager)
        left_layout.addWidget(self.pipeline_config)

        # Right panel: Progress and queue
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # Progress panel
        self.progress_panel = ProgressPanel()
        right_layout.addWidget(self.progress_panel)

        # Batch queue widget
        self.batch_queue = BatchQueueWidget(self.queue_manager)
        right_layout.addWidget(self.batch_queue)

        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)

        # Set initial sizes (55% left, 45% right)
        main_splitter.setSizes([700, 500])

        processing_layout.addWidget(main_splitter)

        # Add tab
        self.tab_widget.addTab(processing_widget, tr("Processing"))

    def _create_models_tab(self):
        """Create the models management tab."""
        models_dir = self.config.get("paths.models_dir", str(Path(__file__).parent.parent / "models"))
        # Pass shared model manager to ensure consistency
        self.model_manager_panel = ModelManagerPanel(
            self.config, 
            models_dir, 
            model_manager=self.model_manager
        )

        # Connect signals
        self.model_manager_panel.model_downloaded.connect(self._on_model_downloaded)
        self.model_manager_panel.model_deleted.connect(self._on_model_deleted)
        
        # Connect to shared model selection manager for automatic refresh
        self.model_manager_panel.model_downloaded.connect(
            lambda mt, ckpt: self.model_selection_manager.refresh()
        )
        self.model_manager_panel.model_deleted.connect(
            lambda mt, ckpt: self.model_selection_manager.refresh()
        )

        self.tab_widget.addTab(self.model_manager_panel, tr("Models"))

    def _create_dependencies_tab(self):
        """Create the dependencies management tab."""
        self.dependency_panel = DependencyPanel(self.config)
        self.tab_widget.addTab(self.dependency_panel, tr("Dependencies"))

    def _setup_statusbar(self):
        """Create application status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage(tr("Ready"))

    def _setup_connections(self):
        """Connect signals and slots."""
        # Video input signals
        self.video_input.video_selected.connect(self._on_video_selected)
        self.video_input.input_type_changed.connect(self._on_input_type_changed)

        # Pipeline config signals
        self.pipeline_config.config_changed.connect(self._on_config_changed)

        # Progress panel signals
        self.progress_panel.cancel_requested.connect(self._on_cancel_processing)

        # Batch queue signals
        self.batch_queue.start_batch_requested.connect(self._on_start_batch)
        self.batch_queue.item_selected.connect(self._on_queue_item_selected)

        # Language change signal
        get_i18n().language_changed.connect(self._on_language_changed_signal)

        # NEW: Connect ProcessingController signals
        self._controller.progress_updated.connect(self.progress_panel.update_progress)
        self._controller.processing_finished.connect(self._on_processing_finished_new)
        self._controller.error_occurred.connect(self._on_processing_error_new)
        self._controller.processing_cancelled.connect(self._on_processing_cancelled_new)
        self._controller.state_changed.connect(self._on_state_changed_new)

    def _load_settings(self):
        """Load saved settings from config."""
        # Load pipeline config into UI
        pipeline_config = self.config.get_pipeline_config()
        self.pipeline_config.load_config(pipeline_config)

    def _save_settings(self):
        """Save current settings to config."""
        # Save pipeline config from UI
        pipeline_config = self.pipeline_config.get_config()
        self.config.set_pipeline_config(pipeline_config)

        # Save window geometry
        self.config.ui.set("window_geometry", self.saveGeometry().toHex().data().decode())

        # Persist to file
        self.config.save()
    
    def _apply_language_preference(self):
        """Apply saved language preference from config."""
        saved_lang = self.config.get_language()
        if saved_lang:
            get_i18n().set_language(saved_lang)
    
    def _retranslate_ui(self):
        """Retranslate all UI elements after language change."""
        self.setWindowTitle(tr("VFI-gui - Video Frame Interpolation"))
        self.statusbar.showMessage(tr("Ready"))
        
        # Rebuild menu bar
        self.menuBar().clear()
        self._setup_menubar()
        
        # Update tab titles
        self.tab_widget.setTabText(0, tr("Processing"))
        self.tab_widget.setTabText(1, tr("Models"))
        self.tab_widget.setTabText(2, tr("Dependencies"))
        
        # Refresh widgets
        self.video_input.retranslate_ui()
        self.pipeline_config.retranslate_ui()
        self.progress_panel.retranslate_ui()
        self.batch_queue.retranslate_ui()
        self.model_manager_panel.retranslate_ui()
        self.dependency_panel.retranslate_ui()

    # === Action handlers ===

    def _on_open_video(self):
        """Handle open video action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select Video File"),
            "",
            tr("Video Files (*.mp4 *.mkv *.avi *.webm *.mov *.flv);;All Files (*)"),
        )
        if file_path:
            self.video_input.set_video_path(file_path)

    def _on_open_folder(self):
        """Handle open folder action for batch processing."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            tr("Select Folder with Videos"),
        )
        if folder_path:
            # Add all video files from folder to queue
            video_extensions = [".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv"]
            folder = Path(folder_path)
            video_files = [
                str(f)
                for f in folder.iterdir()
                if f.suffix.lower() in video_extensions
            ]
            if video_files:
                for video_file in video_files:
                    self.queue_manager.add_item(video_file)
                self.batch_queue.refresh()
                self.statusbar.showMessage(tr("Added {} videos to queue").format(len(video_files)))
            else:
                QMessageBox.information(
                    self,
                    tr("No Videos Found"),
                    tr("No video files found in the selected folder."),
                )

    def _on_video_selected(self, file_path: str):
        """Handle video selection."""
        self._current_video = file_path
        self.statusbar.showMessage(tr("Selected: {}").format(Path(file_path).name))
        self.video_selected.emit(file_path)

    def _on_input_type_changed(self, is_image_sequence: bool):
        """Handle input type change (video file vs image sequence).
        
        Automatically switches output mode to match input type.
        
        Args:
            is_image_sequence: True if input is image sequence, False if video file.
        """
        # Get the codec settings widget from pipeline config
        codec_settings = self.pipeline_config.codec_settings
        
        # Switch output mode to match input type
        if is_image_sequence:
            # Set output mode to "images"
            for i in range(codec_settings.output_mode_combo.count()):
                if codec_settings.output_mode_combo.itemData(i) == "images":
                    codec_settings.output_mode_combo.setCurrentIndex(i)
                    break
        else:
            # Set output mode to "video"
            for i in range(codec_settings.output_mode_combo.count()):
                if codec_settings.output_mode_combo.itemData(i) == "video":
                    codec_settings.output_mode_combo.setCurrentIndex(i)
                    break

    def _on_config_changed(self):
        """Handle pipeline configuration changes."""
        self._save_settings()

    def _on_add_to_queue(self):
        """Add current video to batch queue."""
        video_path = self.video_input.get_video_path()
        if video_path:
            config = self.pipeline_config.get_config()
            self.queue_manager.add_item(video_path, config)
            self.batch_queue.refresh()
            self.statusbar.showMessage(tr("Added to queue"))

    def _on_start_processing(self):
        """Start processing the current video."""
        video_path = self.video_input.get_video_path()
        if not video_path:
            QMessageBox.warning(self, tr("No Video"), tr("Please select a video file first."))
            return

        self._start_processing(video_path)

    def _start_processing(self, video_path: str, custom_config: Optional[Dict[str, Any]] = None):
        """Start processing a video file using TaskOrchestrator via ProcessingController."""
        logger.info(f"Starting processing: {video_path}")

        # Get pipeline config (now passed directly to orchestrator)
        pipeline_config = custom_config or self.pipeline_config.get_config()

        # Use ProcessingController to start processing
        # This delegates to TaskOrchestrator which handles backend + IO
        task_id = self._controller.start_processing(video_path, pipeline_config)

        if task_id:
            # Update UI state
            self._set_processing_state(True)
            self.processing_started.emit()
            self.statusbar.showMessage(tr("Processing started..."))
        else:
            logger.error(f"Failed to start processing: {video_path}")

    def _on_pause_processing(self):
        """Pause/resume processing using ProcessingController."""
        state = self._controller.get_state()
        if state == "paused":
            self._controller.resume_processing()
            self.pause_action.setText(tr("Pause"))
            self.statusbar.showMessage(tr("Processing resumed"))
        elif state == "running":
            self._controller.pause_processing()
            self.pause_action.setText(tr("Resume"))
            self.statusbar.showMessage(tr("Processing paused"))

    def _on_cancel_processing(self):
        """Cancel processing using ProcessingController."""
        if self._controller.is_processing():
            reply = QMessageBox.question(
                self,
                tr("Cancel Processing"),
                tr("Are you sure you want to cancel processing?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.statusbar.showMessage(tr("Cancelling processing..."))
                self._controller.cancel_processing()

    def _on_processing_finished(self, output_path: str):
        """Handle processing completion."""
        logger.info(f"Processing finished: {output_path}")
        self._set_processing_state(False)
        self.processing_finished.emit()
        self.statusbar.showMessage(tr("Processing complete: {}").format(output_path))

        QMessageBox.information(
            self,
            tr("Processing Complete"),
            tr("Output saved to:\n{}").format(output_path),
        )

        # Check if more items in queue
        if self.queue_manager.has_pending():
            self._process_next_in_queue()

    def _on_processing_error(self, error_message: str):
        """Handle processing error (legacy handler for Processor signals)."""
        logger.error(f"Processing error: {error_message}")
        self._set_processing_state(False)
        self.statusbar.showMessage(tr("Processing failed"))

        QMessageBox.critical(
            self,
            tr("Processing Error"),
            tr("An error occurred during processing:\n\n{}").format(error_message),
        )

    # ====================
    # NEW: Handlers for ProcessingController signals
    # ====================

    def _on_processing_finished_new(self, success: bool, message: str):
        """Handle processing completion from TaskOrchestrator."""
        if success:
            logger.info(f"Processing finished: {message}")
            self._set_processing_state(False)
            self.processing_finished.emit()
            self.statusbar.showMessage(tr("Processing complete: {}").format(message))

            QMessageBox.information(
                self,
                tr("Processing Complete"),
                tr("Output saved to:\n{}").format(message),
            )
        else:
            # Error case - handled by _on_processing_error_new
            pass

    def _on_processing_error_new(self, error_message: str):
        """Handle processing error from TaskOrchestrator."""
        logger.error(f"Processing error: {error_message}")
        self._set_processing_state(False)
        self.statusbar.showMessage(tr("Processing failed"))

        QMessageBox.critical(
            self,
            tr("Processing Error"),
            tr("An error occurred during processing:\n\n{}").format(error_message),
        )

    def _on_processing_cancelled_new(self):
        """Handle processing cancelled from TaskOrchestrator."""
        logger.info("Processing cancelled")
        self._set_processing_state(False)
        self.processing_cancelled.emit()
        self.statusbar.showMessage(tr("Processing cancelled"))

    def _on_state_changed_new(self, state: str):
        """Handle orchestrator state change."""
        if state == "paused":
            self.pause_action.setText(tr("Resume"))
        else:
            self.pause_action.setText(tr("Pause"))

    # ====================

    def _set_processing_state(self, processing: bool):
        """Update UI state based on processing status."""
        self.start_action.setEnabled(not processing)
        self.pause_action.setEnabled(processing)
        self.cancel_action.setEnabled(processing)
        self.video_input.setEnabled(not processing)
        self.pipeline_config.setEnabled(not processing)

    def _on_start_batch(self):
        """Start batch processing."""
        if self.queue_manager.has_pending():
            self._process_next_in_queue()

    def _process_next_in_queue(self):
        """Process the next item in the queue."""
        item = self.queue_manager.get_next_pending()
        if item:
            self._start_processing(item.video_path, item.config)

    def _on_queue_item_selected(self, index: int):
        """Handle queue item selection."""
        item = self.queue_manager.get_item(index)
        if item:
            self.video_input.set_video_path(item.video_path)
            self.pipeline_config.load_config(item.config)

    def _on_clear_queue(self):
        """Clear the batch queue."""
        reply = QMessageBox.question(
            self,
            tr("Clear Queue"),
            tr("Are you sure you want to clear the queue?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.queue_manager.clear()
            self.batch_queue.refresh()

    def _on_model_manager(self):
        """Switch to model manager tab."""
        self.tab_widget.setCurrentIndex(1)  # Switch to Models tab

    def _on_proxy_settings(self):
        """Open proxy settings dialog."""
        from ui.widgets.proxy_settings_dialog import ProxySettingsDialog
        dialog = ProxySettingsDialog(self)
        dialog.exec()

    def _on_performance_settings(self):
        """Open performance settings dialog."""
        from ui.widgets.performance_settings_dialog import PerformanceSettingsDialog
        dialog = PerformanceSettingsDialog(self, self.settings_controller)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Apply performance settings to backend config
            settings = dialog.get_settings()
            self._apply_performance_settings(settings)
            self.statusbar.showMessage(tr("Performance settings applied"))

    def _apply_performance_settings(self, settings: dict):
        """Apply performance settings to backend configuration.

        DEPRECATED: This method is deprecated. Performance settings should now
        be passed via pipeline_config to TaskOrchestrator instead.

        Args:
            settings: Performance settings dictionary

        Note:
            The new architecture passes these settings through the pipeline config
            to TaskOrchestrator rather than modifying a shared backend_config.
        """
        import warnings
        warnings.warn(
            "_apply_performance_settings() is deprecated. "
            "Performance settings should be set via the pipeline config UI.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning("Performance settings are now passed via pipeline config to TaskOrchestrator")

    def _on_model_downloaded(self, model_type: str, ckpt_name: str):
        """Handle model download completion."""
        logger.info(f"Model downloaded: {model_type}/{ckpt_name}")
        self.model_manager_panel.refresh()

    def _on_model_deleted(self, model_type: str, ckpt_name: str):
        """Handle model deletion."""
        logger.info(f"Model deleted: {model_type}/{ckpt_name}")
        self.model_manager_panel.refresh()

    def _on_save_settings(self):
        """Save current settings."""
        self._save_settings()
        self.statusbar.showMessage(tr("Settings saved"))

    def _on_benchmark(self):
        """Show benchmark and device detection dialog."""
        dialog = BenchmarkDialog(self)
        dialog.exec()

    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            tr("About VFI-gui"),
            """<h3>VFI-gui</h3>
            <p>{}</p>
            <p>Version 0.1.0</p>
            <p>{}</p>
            <p>{}</p>""".format(
                tr("Video Frame Interpolation GUI"),
                tr("A PyQt6 desktop application for VSGAN-tensorrt-docker video processing workflow."),
                tr("Supports RIFE interpolation, ESRGAN/CUGAN upscaling, and scene detection."),
            ),
        )
    
    def _on_language_changed(self, language_code: str):
        """Handle language selection change."""
        if get_i18n().set_language(language_code):
            self.config.set_language(language_code)
            self.config.save()
            self._retranslate_ui()
    
    def _on_language_changed_signal(self, language_code: str):
        """Handle language change signal from i18n manager."""
        self._retranslate_ui()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event."""
        # Check if processing is in progress using ProcessingController
        if self._controller.is_processing():
            reply = QMessageBox.question(
                self,
                tr("Processing in Progress"),
                tr("Processing is still running. Cancel and exit?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            # Cancel via controller
            self._controller.cancel_processing()

        # Shutdown orchestrator gracefully
        self._orchestrator.shutdown()

        self._save_settings()
        event.accept()
