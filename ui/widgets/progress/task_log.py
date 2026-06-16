"""TaskLog — real-time processing log viewer with auto-scroll.

A read-only log viewer for task processing messages.
Binds to TaskViewModel.log_entry_added for new log entries.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QPlainTextEdit, QPushButton
from PyQt6.QtCore import QEvent, Qt

from ui.styles.theme import Theme

if TYPE_CHECKING:
    from ui.viewmodels.task_viewmodel import TaskViewModel


class TaskLog(QWidget):
    """Real-time processing log viewer with auto-scroll.
    
    Binds to TaskViewModel.log_entry_added for new log entries.
    
    Features:
        - QGroupBox titled "Log"
        - QPlainTextEdit (read-only) for log display
        - Auto-scrolls to bottom on new entries
        - Log format: "[HH:MM:SS] message"
        - Color-coded log levels (info/warning/error) via HTML
        - Clear button
        - Max ~200 lines (truncates oldest)
    """

    MAX_LOG_LINES = 200

    def __init__(self, vm: "TaskViewModel", parent=None):
        """Initialize TaskLog.
        
        Args:
            vm: TaskViewModel for log data binding
            parent: Parent widget
        """
        super().__init__(parent)
        self._vm = vm
        self._log_lines: list[str] = []
        self.setObjectName("taskLogWidget")
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
        if hasattr(self, "_group_box"):
            self._group_box.setTitle(self.tr("Log"))
            self._clear_btn.setText(self.tr("Clear"))

    def _setup_ui(self) -> None:
        """Create widget structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Theme.SPACING_SM)

        # Group box
        self._group_box = QGroupBox(self)
        self._group_box.setObjectName("taskLogGroupBox")
        group_layout = QVBoxLayout(self._group_box)
        group_layout.setContentsMargins(
            Theme.PADDING_MD, Theme.PADDING_MD,
            Theme.PADDING_MD, Theme.PADDING_MD
        )
        group_layout.setSpacing(Theme.SPACING_SM)

        # Log text edit (read-only)
        self._log_text = QPlainTextEdit()
        self._log_text.setObjectName("taskLogText")
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(self.MAX_LOG_LINES)
        group_layout.addWidget(self._log_text)

        # Clear button
        self._clear_btn = QPushButton()
        self._clear_btn.setObjectName("taskLogClearBtn")
        self._clear_btn.clicked.connect(self._clear_log)
        group_layout.addWidget(self._clear_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self._group_box)

    def _bind_viewmodel(self) -> None:
        """Connect to TaskViewModel signals."""
        self._vm.log_entry_added.connect(self._on_log_entry_added)

    def _on_log_entry_added(self, level: str, message: str) -> None:
        """Handle new log entry from ViewModel.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self._get_level_color(level)

        # Format as HTML for color support
        formatted_line = f'<span style="color: {color};">[{timestamp}] [{level.upper()}] {message}</span>'
        self._log_lines.append(formatted_line)

        # Truncate if exceeds max lines
        if len(self._log_lines) > self.MAX_LOG_LINES:
            self._log_lines = self._log_lines[-self.MAX_LOG_LINES:]

        # Update display
        self._update_log_display()

        # Auto-scroll to bottom
        self._log_text.ensureCursorVisible()
        cursor = self._log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._log_text.setTextCursor(cursor)

    def _get_level_color(self, level: str) -> str:
        """Get color for log level.
        
        Args:
            level: Log level
            
        Returns:
            Hex color string
        """
        level_colors = {
            "info": Theme.LOG_INFO,
            "warning": Theme.LOG_WARNING,
            "error": Theme.LOG_ERROR,
            "success": Theme.LOG_SUCCESS,
            "debug": Theme.TEXT_SECONDARY,
        }
        return level_colors.get(level.lower(), Theme.LOG_INFO)

    def _update_log_display(self) -> None:
        """Update log text display with all lines."""
        # Use appendHtml for each line for proper HTML rendering in QPlainTextEdit
        self._log_text.clear()
        for line in self._log_lines:
            self._log_text.appendHtml(line)

    def _clear_log(self) -> None:
        """Clear all log entries."""
        self._log_lines.clear()
        self._log_text.clear()


__all__ = ["TaskLog"]