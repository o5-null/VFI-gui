"""ConfigPage - Configuration view for VFI-gui.

Displayed when no task is running. Contains:
- VideoDropZone
- InterpolationGroup
- SceneDetectGroup
- OutputGroup
- QueueListPreview
- DevicePanel

Layout uses QSplitter for left/right sections.
"""

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QMessageBox,
)

from ui.styles import Theme
from ui.widgets.video_drop_zone import VideoDropZone
from ui.widgets.pipeline.interpolation_group import InterpolationGroup
from ui.widgets.pipeline.scene_detect_group import SceneDetectGroup
from ui.widgets.pipeline.output_group import OutputGroup
from ui.widgets.queue_list_preview import QueueListPreview
from ui.widgets.device_panel import DevicePanel

if TYPE_CHECKING:
    from ui.viewmodels import ViewModelContainer
    from ui.controllers import ControllerContainer


class ConfigPage(QWidget):
    """Configuration view — displayed when task is not running.
    
    Layout:
        ┌──────────────────────────────────────────────────────────────┐
        │  [▶ Start Processing]  [+ Add to Queue]  [Settings]          │
        ├──────────────────────────┬───────────────────────────────────┤
        │  LEFT (stretch=2):       │  RIGHT (stretch=1):               │
        │  ┌────────────────────┐  │  ┌─────────────────────────────┐ │
        │  │  VideoDropZone      │  │  │  QueueListPreview           │ │
        │  ├────────────────────┤  │  ├─────────────────────────────┤ │
        │  │  InterpolationGroup │  │  │  DevicePanel               │ │
        │  ├────────────────────┤  │  └─────────────────────────────┘ │
        │  │  SceneDetectGroup   │  │                                   │
        │  ├────────────────────┤  │                                   │
        │  │  OutputGroup        │  │                                   │
        │  └────────────────────┘  │                                   │
        └──────────────────────────┴───────────────────────────────────┘
    
    Uses QSplitter (horizontal) for left/right split.
    All user-visible text uses self.tr() for i18n.
    """
    
    def __init__(
        self,
        vms: "ViewModelContainer",
        ctrls: "ControllerContainer",
        parent=None,
    ):
        """Initialize ConfigPage.
        
        Args:
            vms: ViewModelContainer with pipeline, task, queue, device, codec
            ctrls: ControllerContainer with processing, queue, settings
            parent: Parent widget
        """
        super().__init__(parent)
        self._vms = vms
        self._ctrls = ctrls
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Setup UI layout."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
        )
        main_layout.setSpacing(Theme.SPACING_MD)
        
        # Top action buttons
        self._setup_action_buttons(main_layout)
        
        # Horizontal splitter for left/right sections
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        
        # Left section (stretch=2)
        left_widget = self._create_left_section()
        splitter.addWidget(left_widget)
        
        # Right section (stretch=1)
        right_widget = self._create_right_section()
        splitter.addWidget(right_widget)
        
        # Set stretch factors
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter, 1)
        
    def _setup_action_buttons(self, layout: QVBoxLayout) -> None:
        """Setup action button row at top."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(Theme.SPACING_MD)
        
        # Start Processing button
        self._start_btn = QPushButton()
        self._start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.ACCENT};
                color: {Theme.TEXT_PRIMARY};
                border: none;
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Theme.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {Theme.ACCENT_PRESSED};
            }}
            QPushButton:disabled {{
                background-color: {Theme.ACCENT_DISABLED};
                color: {Theme.TEXT_DISABLED};
            }}
        """)
        self._start_btn.clicked.connect(self._on_start_clicked)
        button_layout.addWidget(self._start_btn)
        
        # Add to Queue button
        self._add_queue_btn = QPushButton()
        self._add_queue_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                border-color: {Theme.BORDER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_PRESSED};
            }}
        """)
        self._add_queue_btn.clicked.connect(self._on_add_queue_clicked)
        button_layout.addWidget(self._add_queue_btn)
        
        # Settings button
        self._settings_btn = QPushButton()
        self._settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                border-color: {Theme.BORDER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_PRESSED};
            }}
        """)
        self._settings_btn.clicked.connect(self._on_settings_clicked)
        button_layout.addWidget(self._settings_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def _create_left_section(self) -> QWidget:
        """Create left section with real config widgets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Theme.SPACING_MD)
        
        # VideoDropZone
        self._video_drop_zone = VideoDropZone(self._vms.pipeline, self)
        layout.addWidget(self._video_drop_zone)
        
        # InterpolationGroup
        self._interp_group = InterpolationGroup(self._vms.pipeline, self)
        layout.addWidget(self._interp_group)
        
        # SceneDetectGroup
        self._scene_detect_group = SceneDetectGroup(self._vms.pipeline, self)
        layout.addWidget(self._scene_detect_group)
        
        # OutputGroup (needs PipelineVM + CodecVM)
        self._output_group = OutputGroup(self._vms.pipeline, self._vms.codec, self)
        layout.addWidget(self._output_group)
        
        layout.addStretch()
        return widget
        
    def _create_right_section(self) -> QWidget:
        """Create right section with queue preview and device panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Theme.SPACING_MD)
        
        # QueueListPreview
        self._queue_preview = QueueListPreview(self._vms.queue, self._ctrls.queue, self)
        layout.addWidget(self._queue_preview)
        
        # DevicePanel
        self._device_panel = DevicePanel(self._vms.device, self)
        layout.addWidget(self._device_panel)
        
        layout.addStretch()
        return widget
        
    def _connect_signals(self) -> None:
        """Connect ViewModel signals."""
        # Device info is now handled by real DevicePanel widget
        pass
        
    # ====================
    # Button Handlers
    # ====================
    
    def _on_start_clicked(self) -> None:
        """Handle Start Processing button click.
        
        Validates config, persists, starts processing.
        """
        # Validate configuration
        errors = self._vms.pipeline.validate()
        
        if errors:
            QMessageBox.warning(
                self,
                self.tr("Validation Error"),
                "\n".join(errors),
            )
            return
        
        # Persist configuration
        self._vms.pipeline.persist()
        
        # Get video path
        video_path = self._vms.pipeline.video_path
        
        if not video_path:
            QMessageBox.warning(
                self,
                self.tr("No Video Selected"),
                self.tr("Please select a video file first."),
            )
            return
        
        # Get pipeline config
        pipeline_config = self._vms.pipeline.to_pipeline_config()
        
        # Start processing
        self._ctrls.processing.start_task(video_path, pipeline_config)
        
    def _on_add_queue_clicked(self) -> None:
        """Handle Add to Queue button click."""
        # Validate configuration
        errors = self._vms.pipeline.validate()
        
        if errors:
            QMessageBox.warning(
                self,
                self.tr("Validation Error"),
                "\n".join(errors),
            )
            return
        
        # Get video path
        video_path = self._vms.pipeline.video_path
        
        if not video_path:
            QMessageBox.warning(
                self,
                self.tr("No Video Selected"),
                self.tr("Please select a video file first."),
            )
            return
        
        # Get pipeline config
        pipeline_config = self._vms.pipeline.to_pipeline_config()
        
        # Add to queue
        self._ctrls.queue.add_to_queue(video_path, pipeline_config)
        
    def _on_settings_clicked(self) -> None:
        """Handle Settings button click."""
        from ui.widgets.dialogs import SettingsDialog
        from ui.app import get_app
        
        # SettingsDialog requires ConfigFacade
        app = get_app()
        dialog = SettingsDialog(app.config, self)
        dialog.exec()
        
    # ====================
    # i18n Support
    # ====================
    
    def changeEvent(self, a0: QEvent | None) -> None:  # noqa: ARG002
        """Handle change events (language change)."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)
        
    def _retranslate_ui(self) -> None:
        """Retranslate UI text for current locale."""
        self._start_btn.setText(self.tr("▶ Start Processing"))
        self._add_queue_btn.setText(self.tr("+ Add to Queue"))
        self._settings_btn.setText(self.tr("Settings"))


__all__ = ["ConfigPage"]