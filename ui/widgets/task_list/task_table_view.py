"""TaskTableView - Table view for task list (qBittorrent-style).

Displays tasks in a QTableView with columns:
- Status icon + name
- Progress
- FPS
- Size
- ETA
- Added time

Uses QAbstractTableModel for MVVM binding with QueueViewModel.
"""

from typing import TYPE_CHECKING, List

from PyQt6.QtCore import (
    QEvent, Qt, QAbstractTableModel, QModelIndex, QSize,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHeaderView, QTableView,
    QAbstractItemView, QStyledItemDelegate, QStyleOptionViewItem,
    QApplication, QStyle,
)

from ui.styles.theme import Theme
from ui.styles.icons import IconManager

if TYPE_CHECKING:
    from ui.viewmodels.queue_viewmodel import QueueViewModel, QueueItemVO


class TaskTableModel(QAbstractTableModel):
    """Table model for task list data binding."""

    COLUMNS = ["name", "status", "progress", "fps", "eta"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: List["QueueItemVO"] = []
        self._column_labels = {
            "name": "名称",
            "status": "状态",
            "progress": "进度",
            "fps": "FPS",
            "eta": "剩余时间",
        }

    def set_items(self, items: List["QueueItemVO"]):
        """Update model data."""
        self.beginResetModel()
        self._items = items
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._items)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._items):
            return None

        item = self._items[index.row()]
        col = self.COLUMNS[index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            if col == "name":
                return item.video_name
            elif col == "status":
                status_map = {
                    "queued": "等待中", "pending": "等待中",
                    "processing": "处理中", "completed": "已完成",
                    "failed": "失败", "cancelled": "已取消",
                }
                return status_map.get(item.status, item.status)
            elif col == "progress":
                return f"{int(item.progress * 100)}%"
            elif col == "fps":
                return f"{item.fps:.1f}" if item.fps > 0 else "-"
            elif col == "eta":
                return "-"

        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == "status":
                color_map = {
                    "queued": Theme.STATUS_QUEUED,
                    "pending": Theme.STATUS_QUEUED,
                    "processing": Theme.STATUS_PROCESSING,
                    "completed": Theme.STATUS_COMPLETED,
                    "failed": Theme.STATUS_FAILED,
                    "cancelled": Theme.STATUS_CANCELLED,
                }
                color = color_map.get(item.status, Theme.TEXT_SECONDARY)
                return QColor(color)

        elif role == Qt.ItemDataRole.UserRole:
            return item.index

        return None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            col = self.COLUMNS[section]
            return self._column_labels.get(col, col)
        return None

    def item_at_row(self, row: int) -> "QueueItemVO | None":
        """Get QueueItemVO at row."""
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def set_column_labels(self, labels: dict):
        """Update column labels for i18n."""
        self._column_labels.update(labels)
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(self.COLUMNS) - 1)


class ProgressDelegate(QStyledItemDelegate):
    """Custom delegate that renders progress as a colored bar in the progress column."""

    COLUMNS = TaskTableModel.COLUMNS

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex):
        col = self.COLUMNS[index.column()]
        if col == "progress":
            item = index.model().item_at_row(index.row())  # type: ignore[union-attr]
            if item is None:
                return

            progress = item.progress
            painter.save()

            bg_rect = option.rect.adjusted(2, 2, -2, -2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(Theme.PROGRESS_BG))
            painter.drawRoundedRect(bg_rect, 2, 2)

            if progress > 0:
                fill_width = int(bg_rect.width() * progress)
                fill_rect = bg_rect.adjusted(0, 0, -(bg_rect.width() - fill_width), 0)

                if item.status == "completed":
                    bar_color = QColor(Theme.STATUS_COMPLETED)
                elif item.status == "failed":
                    bar_color = QColor(Theme.STATUS_FAILED)
                else:
                    bar_color = QColor(Theme.PROGRESS_FILL)

                painter.setBrush(bar_color)
                painter.drawRoundedRect(fill_rect, 2, 2)

            text = f"{int(progress * 100)}%"
            painter.setPen(QColor(Theme.PROGRESS_TEXT))
            painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, text)

            painter.restore()
        else:
            super().paint(painter, option, index)


class TaskTableView(QWidget):
    """Table view for task list (qBittorrent-style).

    Features:
        - QTableView with custom model
        - Progress bar rendering in progress column
        - Status color coding
        - Row selection with context menu support
        - Filter support from StatusFilterPanel
        - Double-click to view task details

    Signals:
        selection_changed(int): Emitted when selected row changes (-1 if none)
    """

    selection_changed = pyqtSignal(int)

    def __init__(self, vm: "QueueViewModel", parent=None):
        super().__init__(parent)
        self._vm = vm
        self._current_filter = "all"
        self._model = TaskTableModel(self)
        self._setup_ui()
        self._bind_viewmodel()
        self._retranslate_ui()

    def _setup_ui(self):
        """Setup the table view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setItemDelegateForColumn(
            self._model.COLUMNS.index("progress"),
            ProgressDelegate(self._table),
        )

        self._table.setStyleSheet(f"""
            QTableView {{
                background-color: {Theme.BG_PRIMARY};
                alternate-background-color: {Theme.BG_SECONDARY};
                border: none;
                gridline-color: {Theme.BORDER};
                selection-background-color: {Theme.ACCENT};
                selection-color: {Theme.TEXT_PRIMARY};
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QTableView::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {Theme.BORDER};
            }}
            QTableView::item:selected {{
                background-color: {Theme.ACCENT};
            }}
            QHeaderView::section {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                padding: 6px 8px;
                border: none;
                border-right: 1px solid {Theme.BORDER};
                border-bottom: 1px solid {Theme.BORDER};
                font-weight: bold;
                font-size: {Theme.FONT_SIZE_MD};
            }}
        """)

        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(2, 120)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        layout.addWidget(self._table)

    def _bind_viewmodel(self):
        """Bind to QueueViewModel signals."""
        if self._vm:
            self._vm.queue_changed.connect(self._refresh)
            self._vm.item_added.connect(self._refresh)
            self._vm.item_removed.connect(self._refresh)
            self._vm.item_status_changed.connect(self._refresh)
            self._refresh()

    def _refresh(self):
        """Refresh table data from ViewModel."""
        items = self._vm.items() if self._vm else []

        if self._current_filter != "all":
            items = [i for i in items if i.status == self._current_filter
                     or (self._current_filter == "queued" and i.status == "pending")]

        self._model.set_items(items)

    def _on_selection_changed(self, selected, deselected):
        """Handle row selection change."""
        indexes = selected.indexes()
        if indexes:
            row = indexes[0].row()
            item = self._model.item_at_row(row)
            if item:
                self.selection_changed.emit(item.index)
                return
        self.selection_changed.emit(-1)

    def set_filter(self, filter_value: str):
        """Set filter from StatusFilterPanel."""
        self._current_filter = filter_value
        self._refresh()

    def get_selected_item(self) -> "QueueItemVO | None":
        """Get currently selected item."""
        indexes = self._table.selectionModel().selectedRows()
        if indexes:
            return self._model.item_at_row(indexes[0].row())
        return None

    def _retranslate_ui(self):
        """Update UI text for i18n."""
        self._model.set_column_labels({
            "name": self.tr("名称"),
            "status": self.tr("状态"),
            "progress": self.tr("进度"),
            "fps": "FPS",
            "eta": self.tr("剩余时间"),
        })

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(a0)


__all__ = ["TaskTableView", "TaskTableModel", "ProgressDelegate"]
