"""MainWindow - Main application window for VFI-gui.

qBittorrent-style layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Menu Bar:  文件(F)  编辑(E)  视图(V)  工具(T)  帮助(H)        │
    ├──────────────────────────────────────────────────────────────────┤
    │  Toolbar: [打开] [添加] [开始] [暂停] [停止] [删除] [设置]     │
    ├──────────┬───────────────────────────────────────────────────────┤
    │ 状态     │  TaskTableView                                       │
    │ 全部(0)  │  ┌───────────────────────────────────────────────┐   │
    │ 处理中(0)│  │ 名称  │ 状态 │ 进度 │ FPS │ 剩余时间          │   │
    │ 等待中(0)│  │       │      │      │     │                   │   │
    │ 已完成(0)│  └───────────────────────────────────────────────┘   │
    │ 失败(0)  ├───────────────────────────────────────────────────────┤
    │ 已取消(0)│  TaskDetailsTabs                                      │
    │──────────│  [通用] [进度] [日志] [GPU]                          │
    │ 分类     │                                                       │
    │ 全部(0)  │                                                       │
    │ 未分类(0)│                                                       │
    │──────────│                                                       │
    │ 标签     │                                                       │
    │ 全部(0)  │                                                       │
    │ 无标签(0)│                                                       │
    ├──────────┴───────────────────────────────────────────────────────┤
    │  状态栏: 就绪 │ DHH:MM │ ↑0.0 kB/s │ NVIDIA RTX 4090           │
    └──────────────────────────────────────────────────────────────────┘

No page switching — single unified view. ConfigPage becomes a dialog.
"""

from typing import TYPE_CHECKING

from loguru import logger

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QStatusBar, QLabel,
    QMessageBox, QFileDialog, QDialog,
)

from ui.styles import Theme, IconManager
from ui.widgets.sidebar.status_filter_panel import StatusFilterPanel
from ui.widgets.task_list.task_table_view import TaskTableView
from ui.widgets.details.task_details_tabs import TaskDetailsTabs

if TYPE_CHECKING:
    from ui.app import VFIApp


class MainWindow(QMainWindow):
    """Main window — qBittorrent-style unified layout.

    Single view: sidebar + table + details tabs.
    No page switching. ConfigPage opened as dialog when needed.
    """

    def __init__(self, app: "VFIApp"):
        super().__init__()
        self._app = app

        logger.debug("MainWindow initializing...")

        self._setup_window()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central()
        self._setup_statusbar()
        self._connect_signals()
        self._load_settings()

        logger.debug("MainWindow initialized")

    # ====================
    # Window Setup
    # ====================

    def _setup_window(self) -> None:
        self.setWindowTitle(self.tr("VFI-gui"))
        self.setMinimumSize(1100, 750)

    # ====================
    # Menu Bar
    # ====================

    def _setup_menubar(self) -> None:
        menubar = self.menuBar()

        # File menu
        self._file_menu = menubar.addMenu(self.tr("文件(&F)"))
        self._add_menu_action(self._file_menu, self.tr("打开视频..."), self._on_open_video, "Ctrl+O")
        self._add_menu_action(self._file_menu, self.tr("打开文件夹..."), self._on_open_folder, "Ctrl+Shift+O")
        self._file_menu.addSeparator()
        self._add_menu_action(self._file_menu, self.tr("退出"), self.close, "Ctrl+Q")

        # Edit menu
        self._edit_menu = menubar.addMenu(self.tr("编辑(&E)"))
        self._add_menu_action(self._edit_menu, self.tr("设置..."), self._on_settings, "Alt+O")

        # View menu
        self._view_menu = menubar.addMenu(self.tr("视图(&V)"))
        self._add_menu_action(self._view_menu, self.tr("切换工具栏"), self._toggle_toolbar)
        self._add_menu_action(self._view_menu, self.tr("切换侧边栏"), self._toggle_sidebar)
        self._add_menu_action(self._view_menu, self.tr("切换详情面板"), self._toggle_details)

# Tools menu
        self._tools_menu = menubar.addMenu(self.tr("工具(&T)"))
        self._add_menu_action(self._tools_menu, self.tr("开始处理"), self._on_start, "Ctrl+S")
        self._add_menu_action(self._tools_menu, self.tr("添加到队列"), self._on_add_queue, "Ctrl+Shift+A")
        self._tools_menu.addSeparator()
        self._add_menu_action(self._tools_menu, self.tr("模型管理器"), self._on_model_manager)
        self._add_menu_action(self._tools_menu, self.tr("基准测试"), self._on_benchmark)

        # Help menu
        self._help_menu = menubar.addMenu(self.tr("帮助(&H)"))
        self._add_menu_action(self._help_menu, self.tr("关于 VFI-gui"), self._on_about)

    def _add_menu_action(self, menu, text: str, slot, shortcut: str = "") -> None:
        """Add action to menu with optional shortcut (type-safe)."""
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        menu.addAction(action)

    # ====================
    # Toolbar
    # ====================

    def _setup_toolbar(self) -> None:
        toolbar = QToolBar(self.tr("主工具栏"))
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {Theme.BG_TERTIARY};
                border: none;
                border-bottom: 1px solid {Theme.BORDER};
                padding: 2px 4px;
                spacing: 4px;
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: {Theme.RADIUS_MD}px;
                padding: 4px 8px;
                color: {Theme.TEXT_PRIMARY};
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QToolButton:hover {{
                background-color: {Theme.BG_HOVER};
                border: 1px solid {Theme.BORDER};
            }}
            QToolButton:pressed {{
                background-color: {Theme.BG_PRESSED};
            }}
        """)

        # Action buttons
        self._toolbar_actions = {}

        # Open
        open_action = QAction(IconManager.get("folder-open"), self.tr("打开"), self)
        open_action.triggered.connect(self._on_open_video)
        toolbar.addAction(open_action)
        self._toolbar_actions["open"] = open_action

        # Add to queue
        add_action = QAction(IconManager.get("add"), self.tr("添加"), self)
        add_action.triggered.connect(self._on_add_queue)
        toolbar.addAction(add_action)
        self._toolbar_actions["add"] = add_action

        toolbar.addSeparator()

        # Start
        start_action = QAction(IconManager.get("play"), self.tr("开始"), self)
        start_action.triggered.connect(self._on_start)
        toolbar.addAction(start_action)
        self._toolbar_actions["start"] = start_action

        # Pause
        pause_action = QAction(IconManager.get("pause"), self.tr("暂停"), self)
        pause_action.triggered.connect(self._on_pause)
        toolbar.addAction(pause_action)
        self._toolbar_actions["pause"] = pause_action

        # Stop
        stop_action = QAction(IconManager.get("stop"), self.tr("停止"), self)
        stop_action.triggered.connect(self._on_stop)
        toolbar.addAction(stop_action)
        self._toolbar_actions["stop"] = stop_action

        toolbar.addSeparator()

        # Delete
        delete_action = QAction(IconManager.get("delete"), self.tr("删除"), self)
        delete_action.triggered.connect(self._on_delete)
        toolbar.addAction(delete_action)
        self._toolbar_actions["delete"] = delete_action

        toolbar.addSeparator()

        # Settings
        settings_action = QAction(IconManager.get("settings"), self.tr("设置"), self)
        settings_action.triggered.connect(self._on_settings)
        toolbar.addAction(settings_action)
        self._toolbar_actions["settings"] = settings_action

        self.addToolBar(toolbar)
        self._toolbar = toolbar

    # ====================
    # Central Layout
    # ====================

    def _setup_central(self) -> None:
        """Setup qBittorrent-style 3-zone layout."""
        vms = self._app.viewmodels

        # Main horizontal splitter: sidebar | (table + details)
        self._h_splitter = QSplitter()
        self._h_splitter.setOrientation(Qt.Orientation.Horizontal)

        # Left: StatusFilterPanel
        self._sidebar = StatusFilterPanel(vms.queue, self)
        self._h_splitter.addWidget(self._sidebar)

        # Right: vertical splitter for table + details
        self._v_splitter = QSplitter()
        self._v_splitter.setOrientation(Qt.Orientation.Vertical)

        # Top: TaskTableView
        self._task_table = TaskTableView(vms.queue, self)
        self._v_splitter.addWidget(self._task_table)

        # Bottom: TaskDetailsTabs
        self._details_tabs = TaskDetailsTabs(vms, self)
        self._v_splitter.addWidget(self._details_tabs)

        # Set vertical splitter sizes (table:details = 3:1)
        self._v_splitter.setStretchFactor(0, 3)
        self._v_splitter.setStretchFactor(1, 1)
        self._v_splitter.setSizes([450, 150])

        self._h_splitter.addWidget(self._v_splitter)

        # Set horizontal splitter sizes (sidebar:main = 1:5)
        self._h_splitter.setStretchFactor(0, 0)
        self._h_splitter.setStretchFactor(1, 1)
        self._h_splitter.setSizes([200, 800])

        self.setCentralWidget(self._h_splitter)

    # ====================
    # Status Bar
    # ====================

    def _setup_statusbar(self) -> None:
        statusbar = QStatusBar()
        statusbar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {Theme.BG_TERTIARY};
                border-top: 1px solid {Theme.BORDER};
                color: {Theme.TEXT_SECONDARY};
                font-size: {Theme.FONT_SIZE_SM};
                padding: 2px;
            }}
        """)

        # State label
        self._state_label = QLabel(self.tr("就绪"))
        statusbar.addWidget(self._state_label)

        # Separator
        sep1 = QLabel(" │ ")
        sep1.setStyleSheet(f"color: {Theme.BORDER};")
        statusbar.addWidget(sep1)

        # Speed / FPS label
        self._speed_label = QLabel("0.0 fps")
        statusbar.addWidget(self._speed_label)

        # Separator
        sep2 = QLabel(" │ ")
        sep2.setStyleSheet(f"color: {Theme.BORDER};")
        statusbar.addWidget(sep2)

        # Queue count
        self._queue_label = QLabel("0 / 0")
        statusbar.addWidget(self._queue_label)

        # Device info (right side)
        best_device = self._app.viewmodels.device.get_best_device()
        self._device_label = QLabel(best_device.name if best_device else "CPU")
        statusbar.addPermanentWidget(self._device_label)

        self.setStatusBar(statusbar)

    # ====================
    # Signal Connections
    # ====================

    def _connect_signals(self) -> None:
        vms = self._app.viewmodels

        # Task state → status bar + toolbar button states
        vms.task.state_changed.connect(self._on_task_state_changed)

        # Device changes → status bar
        vms.device.current_device_changed.connect(self._on_device_changed)

        # Sidebar filter → task table filter
        self._sidebar.filter_changed.connect(self._task_table.set_filter)

        # Task table selection → details tabs
        self._task_table.selection_changed.connect(self._on_task_selected)

        # Queue counts → status bar
        vms.queue.total_count_changed.connect(self._on_queue_count_changed)
        vms.queue.completed_count_changed.connect(self._on_queue_count_changed)

        # FPS → status bar
        vms.task.fps_changed.connect(self._on_fps_changed)

    # ====================
    # Signal Handlers
    # ====================

    def _on_task_state_changed(self, state: str) -> None:
        """Handle task state change."""
        logger.debug(f"Task state changed: {state}")

        state_map = {
            "idle": self.tr("就绪"),
            "loading": self.tr("加载中..."),
            "processing": self.tr("处理中"),
            "paused": self.tr("已暂停"),
            "completed": self.tr("已完成"),
            "failed": self.tr("失败"),
            "cancelled": self.tr("已取消"),
            "cancelling": self.tr("取消中..."),
        }

        state_text = state_map.get(state, state)
        self._state_label.setText(state_text)

        # Update toolbar button states
        is_running = state in ("loading", "processing", "paused")
        self._toolbar_actions["start"].setEnabled(not is_running)
        self._toolbar_actions["pause"].setEnabled(is_running)
        self._toolbar_actions["stop"].setEnabled(is_running)

        # Update pause button text
        if state == "paused":
            self._toolbar_actions["pause"].setText(self.tr("继续"))
        else:
            self._toolbar_actions["pause"].setText(self.tr("暂停"))

    def _on_device_changed(self, device_name: str) -> None:
        self._device_label.setText(device_name)

    def _on_task_selected(self, item_index: int) -> None:
        """Handle task selection in table."""
        if item_index >= 0:
            item = self._app.viewmodels.queue.item_at(item_index)
            self._details_tabs.update_item(item)
        else:
            self._details_tabs.update_item(None)

    def _on_queue_count_changed(self, _=None) -> None:
        """Update queue count in status bar."""
        vm = self._app.viewmodels.queue
        completed = vm.completed_count
        total = vm.total_count
        self._queue_label.setText(f"{completed}/{total}")

    def _on_fps_changed(self, fps: float) -> None:
        """Update FPS in status bar."""
        self._speed_label.setText(f"{fps:.1f} fps")

    # ====================
    # Menu/Toolbar Handlers
    # ====================

    def _on_open_video(self) -> None:
        logger.debug("Open Video clicked")
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("选择视频"),
            "",
            self.tr("视频文件 (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv)"),
        )
        if video_path:
            logger.info(f"Video selected: {video_path}")
            self._app.viewmodels.pipeline.set_video_path(video_path)

    def _on_open_folder(self) -> None:
        logger.debug("Open Folder clicked")
        folder_path = QFileDialog.getExistingDirectory(self, self.tr("选择文件夹"))
        if folder_path:
            logger.info(f"Folder selected: {folder_path}")

    def _on_start(self) -> None:
        """Start processing — open AddTaskDialog, then start if confirmed."""
        from ui.widgets.dialogs.add_task_dialog import AddTaskDialog

        dialog = AddTaskDialog(self._app.viewmodels, self._app.controllers, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.action == "start":
            video_path = self._app.viewmodels.pipeline.video_path
            pipeline_config = self._app.viewmodels.pipeline.to_pipeline_config()
            self._app.controllers.processing.start_task(video_path, pipeline_config)

    def _on_add_queue(self) -> None:
        """Add task to queue — open AddTaskDialog, then add if confirmed."""
        from ui.widgets.dialogs.add_task_dialog import AddTaskDialog

        dialog = AddTaskDialog(self._app.viewmodels, self._app.controllers, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            video_path = self._app.viewmodels.pipeline.video_path
            pipeline_config = self._app.viewmodels.pipeline.to_pipeline_config()
            self._app.controllers.queue.add_to_queue(video_path, pipeline_config)

            # If user chose "start", also start processing
            if dialog.action == "start":
                self._app.controllers.processing.start_task(video_path, pipeline_config)

    def _on_pause(self) -> None:
        """Toggle pause/resume."""
        state = self._app.viewmodels.task.state
        if state == "paused":
            self._app.controllers.processing.resume_task()
        elif state in ("loading", "processing"):
            self._app.controllers.processing.pause_task()

    def _on_stop(self) -> None:
        """Stop current task."""
        self._app.controllers.processing.cancel_task()

    def _on_delete(self) -> None:
        """Delete selected task from queue."""
        item = self._task_table.get_selected_item()
        if item:
            self._app.controllers.queue.remove_item(item.index)

    def _on_settings(self) -> None:
        from ui.widgets.dialogs import SettingsDialog
        dialog = SettingsDialog(self._app.config, self)
        dialog.exec()

    def _on_about(self) -> None:
        from ui.widgets.dialogs import AboutDialog
        dialog = AboutDialog(self)
        dialog.exec()

    def _on_benchmark(self) -> None:
        from ui.widgets.dialogs import BenchmarkDialog
        dialog = BenchmarkDialog(self._app.model_selection, self)
        dialog.exec()

    def _on_model_manager(self) -> None:
        """Open Model Manager dialog."""
        from ui.widgets.dialogs import ModelManagerDialog
        dialog = ModelManagerDialog(self._app.model_manager, self)
        dialog.exec()

    def _toggle_toolbar(self) -> None:
        self._toolbar.setVisible(not self._toolbar.isVisible())

    def _toggle_sidebar(self) -> None:
        self._sidebar.setVisible(not self._sidebar.isVisible())

    def _toggle_details(self) -> None:
        self._details_tabs.setVisible(not self._details_tabs.isVisible())

    # ====================
    # Settings Persistence
    # ====================

    def _load_settings(self) -> None:
        geometry = self._app.controllers.settings.load_window_geometry()
        if geometry:
            self.restoreGeometry(geometry)

        state = self._app.controllers.settings.load_window_state()
        if state:
            self.restoreState(state)

    # ====================
    # Window Events
    # ====================

    def closeEvent(self, a0) -> None:
        logger.info("MainWindow closing...")

        geometry_data = self.saveGeometry().data()
        self._app.controllers.settings.save_window_geometry(geometry_data)

        state_data = self.saveState().data()
        self._app.controllers.settings.save_window_state(state_data)

        self._app.shutdown()
        super().closeEvent(a0)

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)

    def _retranslate_ui(self) -> None:
        """Retranslate UI text for current locale."""
        self.setWindowTitle(self.tr("VFI-gui"))
        self._toolbar.setWindowTitle(self.tr("主工具栏"))
        self._state_label.setText(self.tr("就绪"))

        # Toolbar actions
        if hasattr(self, "_toolbar_actions"):
            self._toolbar_actions["open"].setText(self.tr("打开"))
            self._toolbar_actions["add"].setText(self.tr("添加"))
            self._toolbar_actions["start"].setText(self.tr("开始"))
            self._toolbar_actions["pause"].setText(self.tr("暂停"))
            self._toolbar_actions["stop"].setText(self.tr("停止"))
            self._toolbar_actions["delete"].setText(self.tr("删除"))
            self._toolbar_actions["settings"].setText(self.tr("设置"))


__all__ = ["MainWindow"]
