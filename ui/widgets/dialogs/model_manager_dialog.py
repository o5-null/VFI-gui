"""Model Manager Dialog for VFI-gui.

A dialog for managing model checkpoints:
- View installed models and their status
- Download missing models with progress
- Delete installed models
- i18n support via tr() and retranslate_ui()
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QFrame,
    QMessageBox,
    QSizePolicy,
)
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal, QSize

from ui.styles.theme import Theme
from ui.styles.icons import IconManager

if TYPE_CHECKING:
    from core.model_manager import ModelManager, ModelTypeInfo, CheckpointInfo


class DownloadWorker(QThread):
    """Background worker for downloading model checkpoints.

    Signals:
        progress: Emitted with progress percentage (0-100)
        finished: Emitted when download completes successfully
        error: Emitted with error message if download fails
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        download_urls: list[str],
        dest_path: Path,
        parent=None,
    ):
        super().__init__(parent)
        self._urls = download_urls
        self._dest_path = dest_path
        self._cancelled = False

    def run(self):
        """Download checkpoint from URLs (tries each URL until success)."""
        import requests

        for url in self._urls:
            if self._cancelled:
                return

            try:
                # Create parent directory
                self._dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Stream download with progress
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(self._dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if self._cancelled:
                            # Clean up partial file
                            f.close()
                            if self._dest_path.exists():
                                self._dest_path.unlink()
                            return

                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            self.progress.emit(percent)

                self.finished.emit()
                return

            except Exception as e:
                # Try next URL
                continue

        # All URLs failed
        self.error.emit(self.tr("Download failed: all URLs unavailable"))

    def cancel(self):
        """Cancel the download."""
        self._cancelled = True


class CheckpointItem(QFrame):
    """Widget representing a single checkpoint with status and action button.

    Shows:
    - Checkpoint name
    - Size (MB)
    - Status icon
    - Action button (download/delete)
    """

    def __init__(
        self,
        checkpoint: "CheckpointInfo",
        model_manager: "ModelManager",
        parent=None,
    ):
        super().__init__(parent)
        self._checkpoint = checkpoint
        self._model_manager = model_manager
        self._download_worker: Optional[DownloadWorker] = None
        self._setup_ui()
        self._apply_style()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(event)

    def _setup_ui(self):
        """Create widgets and layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_MD,
            Theme.PADDING_SM,
            Theme.PADDING_MD,
            Theme.PADDING_SM,
        )
        layout.setSpacing(Theme.SPACING_MD)

        # Status icon
        self._status_icon = QLabel()
        self._status_icon.setFixedSize(16, 16)
        layout.addWidget(self._status_icon)

        # Checkpoint name
        self._name_label = QLabel(self._checkpoint.name)
        self._name_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(self._name_label)

        # Size label
        self._size_label = QLabel()
        if self._checkpoint.is_installed:
            self._size_label.setText(f"{self._checkpoint.size_mb:.1f} MB")
        layout.addWidget(self._size_label)

        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(100)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # Action button
        self._action_btn = QPushButton()
        self._action_btn.setFixedSize(80, 28)
        layout.addWidget(self._action_btn)

        # Update UI based on status
        self._update_ui()

    def _apply_style(self):
        """Apply stylesheet using Theme constants."""
        self.setStyleSheet(f"""
            CheckpointItem {{
                background-color: {Theme.BG_TERTIARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
            }}
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-family: {Theme.FONT_FAMILY};
                font-size: {Theme.FONT_SIZE_MD};
            }}
            QProgressBar {{
                background-color: {Theme.PROGRESS_BG};
                border: none;
                border-radius: {Theme.RADIUS_SM}px;
                height: 16px;
            }}
            QProgressBar::chunk {{
                background-color: {Theme.PROGRESS_FILL};
                border-radius: {Theme.RADIUS_SM}px;
            }}
            QPushButton {{
                background-color: {Theme.ACCENT};
                color: {Theme.TEXT_PRIMARY};
                border: none;
                border-radius: {Theme.RADIUS_MD}px;
                font-family: {Theme.FONT_FAMILY};
                font-size: {Theme.FONT_SIZE_SM};
                padding: {Theme.PADDING_SM}px {Theme.PADDING_MD}px;
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

    def _safe_disconnect(self):
        """Disconnect clicked signal safely (ignores if not connected)."""
        try:
            self._action_btn.clicked.disconnect()
        except TypeError:
            pass

    def _update_ui(self):
        """Update widgets based on checkpoint status."""
        if self._checkpoint.is_installed:
            # Installed: show delete button
            icon = IconManager.get("success", Theme.SUCCESS, QSize(16, 16))
            self._status_icon.setPixmap(icon.pixmap(16, 16))
            self._action_btn.setIcon(IconManager.get("delete", Theme.TEXT_PRIMARY, QSize(16, 16)))
            self._action_btn.setText(self.tr("Delete"))
            self._safe_disconnect()
            self._action_btn.clicked.connect(self._on_delete_clicked)
        else:
            # Not installed: show download button
            icon = IconManager.get("download", Theme.ACCENT, QSize(16, 16))
            self._status_icon.setPixmap(icon.pixmap(16, 16))
            self._action_btn.setIcon(IconManager.get("download", Theme.TEXT_PRIMARY, QSize(16, 16)))
            self._action_btn.setText(self.tr("Download"))
            self._safe_disconnect()
            self._action_btn.clicked.connect(self._on_download_clicked)

    def _retranslate_ui(self):
        """Update button text for i18n."""
        if self._checkpoint.is_installed:
            self._action_btn.setText(self.tr("Delete"))
        else:
            self._action_btn.setText(self.tr("Download"))

    def _on_download_clicked(self):
        """Start download."""
        from ui.app import get_app

        app = get_app()
        if app is None:
            return

        # Get download URLs
        urls = self._checkpoint.download_urls
        if not urls:
            QMessageBox.warning(
                self,
                self.tr("Download Error"),
                self.tr("No download URLs available for this checkpoint."),
            )
            return

        # Determine destination path
        models_dir = Path(self._model_manager._models_dir)
        dest_path = models_dir / self._checkpoint.model_type / self._checkpoint.name

        # Show progress bar, hide size label
        self._size_label.setVisible(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._action_btn.setText(self.tr("Cancel"))
        self._safe_disconnect()
        self._action_btn.clicked.connect(self._on_cancel_download)

        # Start download worker
        self._download_worker = DownloadWorker(urls, dest_path, self)
        self._download_worker.progress.connect(self._progress_bar.setValue)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.start()

    def _on_cancel_download(self):
        """Cancel ongoing download."""
        if self._download_worker:
            self._download_worker.cancel()
            self._download_worker.wait()
            self._download_worker = None

        # Reset UI
        self._progress_bar.setVisible(False)
        self._size_label.setVisible(True)
        self._update_ui()

    def _on_download_finished(self):
        """Handle download completion."""
        self._download_worker = None

        # Refresh model manager
        self._model_manager.refresh()

        # Update checkpoint info
        from core.model_manager import ModelStatus
        self._checkpoint.status = ModelStatus.INSTALLED

        # Get updated size
        models_dir = Path(self._model_manager._models_dir)
        ckpt_path = models_dir / self._checkpoint.model_type / self._checkpoint.name
        if ckpt_path.exists():
            self._checkpoint.size_mb = ckpt_path.stat().st_size / (1024 * 1024)

        # Reset UI
        self._progress_bar.setVisible(False)
        self._size_label.setVisible(True)
        self._size_label.setText(f"{self._checkpoint.size_mb:.1f} MB")
        self._update_ui()

        # Notify parent dialog
        parent = self.parent()
        while parent and not isinstance(parent, ModelManagerDialog):
            parent = parent.parent()
        if parent:
            parent._refresh_summary()

    def _on_download_error(self, error_msg: str):
        """Handle download error."""
        self._download_worker = None

        QMessageBox.warning(self, self.tr("Download Error"), error_msg)

        # Reset UI
        self._progress_bar.setVisible(False)
        self._size_label.setVisible(True)
        self._update_ui()

    def _on_delete_clicked(self):
        """Delete installed checkpoint with confirmation."""
        # Confirmation dialog
        result = QMessageBox.question(
            self,
            self.tr("Delete Model"),
            self.tr("Are you sure you want to delete '{name}'?\nSize: {size:.1f} MB")
            .format(name=self._checkpoint.name, size=self._checkpoint.size_mb),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result != QMessageBox.StandardButton.Yes:
            return

        # Delete the file
        if self._checkpoint.path:
            try:
                ckpt_path = Path(self._checkpoint.path)
                if ckpt_path.exists():
                    ckpt_path.unlink()

                # Refresh model manager
                self._model_manager.refresh()

                # Update checkpoint status
                from core.model_manager import ModelStatus
                self._checkpoint.status = ModelStatus.DOWNLOADABLE
                self._checkpoint.size_mb = 0.0

                # Update UI
                self._size_label.setText("")
                self._update_ui()

                # Notify parent dialog
                parent = self.parent()
                while parent and not isinstance(parent, ModelManagerDialog):
                    parent = parent.parent()
                if parent:
                    parent._refresh_summary()
                    parent._installed_tab.refresh()

            except Exception as e:
                QMessageBox.warning(
                    self,
                    self.tr("Delete Error"),
                    self.tr("Failed to delete model: {error}").format(error=str(e)),
                )


class ModelTypeCard(QFrame):
    """Widget representing a model type with its checkpoints.

    Shows:
    - Model type name and description
    - Installed/total count
    - List of checkpoints (CheckpointItem widgets)
    """

    def __init__(
        self,
        model_type: "ModelTypeInfo",
        model_manager: "ModelManager",
        parent=None,
    ):
        super().__init__(parent)
        self._model_type = model_type
        self._model_manager = model_manager
        self._checkpoint_items: list[CheckpointItem] = []
        self._setup_ui()
        self._apply_style()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(event)

    def _setup_ui(self):
        """Create widgets and layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_LG,
            Theme.PADDING_MD,
            Theme.PADDING_LG,
            Theme.PADDING_MD,
        )
        layout.setSpacing(Theme.SPACING_SM)

        # Header: name + info + count
        header_layout = QHBoxLayout()
        header_layout.setSpacing(Theme.SPACING_MD)

        # Model type name
        self._name_label = QLabel(self._model_type.display_name)
        self._name_label.setStyleSheet(f"""
            font-family: {Theme.FONT_FAMILY};
            font-size: {Theme.FONT_SIZE_LG};
            font-weight: bold;
            color: {Theme.TEXT_PRIMARY};
        """)
        header_layout.addWidget(self._name_label)

        # Info (short feature summary)
        if self._model_type.info:
            self._info_label = QLabel(self._model_type.info)
            self._info_label.setStyleSheet(f"""
                font-family: {Theme.FONT_FAMILY};
                font-size: {Theme.FONT_SIZE_SM};
                color: {Theme.TEXT_SECONDARY};
            """)
            header_layout.addWidget(self._info_label)

        header_layout.addStretch()

        # Installed count
        self._count_label = QLabel()
        installed = self._model_type.installed_count
        total = self._model_type.total_count
        self._count_label.setText(f"{installed}/{total}")
        self._count_label.setStyleSheet(f"""
            font-family: {Theme.FONT_FAMILY};
            font-size: {Theme.FONT_SIZE_MD};
            color: {Theme.ACCENT};
        """)
        header_layout.addWidget(self._count_label)

        layout.addLayout(header_layout)

        # Description
        if self._model_type.description:
            self._desc_label = QLabel(self._model_type.description)
            self._desc_label.setWordWrap(True)
            self._desc_label.setStyleSheet(f"""
                font-family: {Theme.FONT_FAMILY};
                font-size: {Theme.FONT_SIZE_SM};
                color: {Theme.TEXT_SECONDARY};
            """)
            layout.addWidget(self._desc_label)

        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"""
            background-color: {Theme.BORDER};
            max-height: 1px;
        """)
        layout.addWidget(separator)

        # Checkpoints list
        self._checkpoints_layout = QVBoxLayout()
        self._checkpoints_layout.setSpacing(Theme.SPACING_SM)
        layout.addLayout(self._checkpoints_layout)

        # Create checkpoint items
        for checkpoint in self._model_type.checkpoints:
            item = CheckpointItem(checkpoint, self._model_manager, self)
            self._checkpoint_items.append(item)
            self._checkpoints_layout.addWidget(item)

    def _apply_style(self):
        """Apply stylesheet using Theme constants."""
        self.setStyleSheet(f"""
            ModelTypeCard {{
                background-color: {Theme.BG_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_LG}px;
            }}
        """)

    def _retranslate_ui(self):
        """Update checkpoint items for i18n."""
        for item in self._checkpoint_items:
            item._retranslate_ui()

    def refresh_checkpoints(self):
        """Refresh checkpoint items after changes."""
        # Clear existing items
        for item in self._checkpoint_items:
            item.deleteLater()
        self._checkpoint_items.clear()

        # Re-scan checkpoints for this model type
        updated_type = self._model_manager.get_model_types().get(self._model_type.name)
        if updated_type:
            self._model_type = updated_type

            # Update count label
            installed = self._model_type.installed_count
            total = self._model_type.total_count
            self._count_label.setText(f"{installed}/{total}")

            # Recreate checkpoint items
            for checkpoint in self._model_type.checkpoints:
                item = CheckpointItem(checkpoint, self._model_manager, self)
                self._checkpoint_items.append(item)
                self._checkpoints_layout.addWidget(item)


class InstalledTab(QWidget):
    """Tab showing installed model checkpoints."""

    def __init__(self, model_manager: "ModelManager", parent=None):
        super().__init__(parent)
        self._model_manager = model_manager
        self._model_cards: list[ModelTypeCard] = []
        self._setup_ui()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(event)

    def _setup_ui(self):
        """Create widgets and layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
        )
        layout.setSpacing(Theme.SPACING_MD)

        # Scroll area for model cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Theme.BG_PRIMARY};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {Theme.SCROLLBAR_BG};
                width: 12px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Theme.SCROLLBAR_HANDLE};
                border-radius: {Theme.RADIUS_SM}px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Theme.SCROLLBAR_HANDLE_HOVER};
            }}
        """)
        layout.addWidget(scroll)

        # Container for model cards
        container = QWidget()
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(Theme.SPACING_MD)
        self._container_layout.addStretch()
        scroll.setWidget(container)

        # Populate with installed models
        self._populate_installed()

    def _populate_installed(self):
        """Populate with model types that have installed checkpoints."""
        model_types = self._model_manager.get_model_types()

        # Sort by sort_order
        sorted_types = sorted(
            model_types.values(),
            key=lambda x: x.sort_order,
        )

        for model_type in sorted_types:
            if model_type.installed_count > 0:
                card = ModelTypeCard(model_type, self._model_manager, self)
                self._model_cards.append(card)
                self._container_layout.insertWidget(
                    self._container_layout.count() - 1,
                    card,
                )

    def _retranslate_ui(self):
        """Update model cards for i18n."""
        for card in self._model_cards:
            card._retranslate_ui()

    def refresh(self):
        """Refresh the installed models list."""
        # Clear existing cards
        for card in self._model_cards:
            card.deleteLater()
        self._model_cards.clear()

        # Repopulate
        self._populate_installed()


class AvailableTab(QWidget):
    """Tab showing available (not installed) model checkpoints."""

    def __init__(self, model_manager: "ModelManager", parent=None):
        super().__init__(parent)
        self._model_manager = model_manager
        self._model_cards: list[ModelTypeCard] = []
        self._setup_ui()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(event)

    def _setup_ui(self):
        """Create widgets and layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
            Theme.PADDING_MD,
        )
        layout.setSpacing(Theme.SPACING_MD)

        # Scroll area for model cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Theme.BG_PRIMARY};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {Theme.SCROLLBAR_BG};
                width: 12px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Theme.SCROLLBAR_HANDLE};
                border-radius: {Theme.RADIUS_SM}px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Theme.SCROLLBAR_HANDLE_HOVER};
            }}
        """)
        layout.addWidget(scroll)

        # Container for model cards
        container = QWidget()
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(Theme.SPACING_MD)
        self._container_layout.addStretch()
        scroll.setWidget(container)

        # Populate with available models
        self._populate_available()

    def _populate_available(self):
        """Populate with all model types (showing missing checkpoints)."""
        model_types = self._model_manager.get_model_types()

        # Sort by sort_order
        sorted_types = sorted(
            model_types.values(),
            key=lambda x: x.sort_order,
        )

        for model_type in sorted_types:
            # Show all model types in Available tab
            # (CheckpointItem will show appropriate action for each checkpoint)
            card = ModelTypeCard(model_type, self._model_manager, self)
            self._model_cards.append(card)
            self._container_layout.insertWidget(
                self._container_layout.count() - 1,
                card,
            )

    def _retranslate_ui(self):
        """Update model cards for i18n."""
        for card in self._model_cards:
            card._retranslate_ui()

    def refresh(self):
        """Refresh the available models list."""
        # Refresh each card's checkpoints
        for card in self._model_cards:
            card.refresh_checkpoints()


class ModelManagerDialog(QDialog):
    """Dialog for managing model checkpoints.

    Features:
    - Two tabs: Installed Models and Available Models
    - Download models with progress indicator
    - Delete installed models with confirmation
    - Refresh button to rescan models
    - Total installed count and size summary
    - i18n support (tr() + retranslate_ui)
    """

    def __init__(self, model_manager: "ModelManager", parent=None):
        super().__init__(parent)
        self._model_manager = model_manager
        self._setup_ui()
        self._connect_signals()
        self._apply_style()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    def retranslate_ui(self):
        """Update all user-visible text for i18n."""
        self.setWindowTitle(self.tr("Model Manager"))

        # Tab names
        self._tab_widget.setTabText(0, self.tr("Installed Models"))
        self._tab_widget.setTabText(1, self.tr("Available Models"))

        # Buttons
        self._refresh_btn.setText(self.tr("Refresh"))
        close_btn = self._button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_btn:
            close_btn.setText(self.tr("Close"))

        # Summary
        self._update_summary()

        # Refresh tab contents
        self._installed_tab._retranslate_ui()
        self._available_tab._retranslate_ui()

    def _setup_ui(self):
        """Create widgets and layout."""
        self.setWindowTitle(self.tr("Model Manager"))
        self.setMinimumSize(500, 400)
        self.resize(600, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
            Theme.PADDING_LG,
        )
        layout.setSpacing(Theme.SPACING_MD)

        # Header: title + refresh button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(Theme.SPACING_MD)

        title_label = QLabel(self.tr("Model Manager"))
        title_label.setStyleSheet(f"""
            font-family: {Theme.FONT_FAMILY};
            font-size: {Theme.FONT_SIZE_XL};
            font-weight: bold;
            color: {Theme.TEXT_PRIMARY};
        """)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self._refresh_btn = QPushButton()
        self._refresh_btn.setIcon(IconManager.get("refresh", Theme.TEXT_PRIMARY, QSize(16, 16)))
        self._refresh_btn.setText(self.tr("Refresh"))
        self._refresh_btn.setFixedHeight(28)
        header_layout.addWidget(self._refresh_btn)

        layout.addLayout(header_layout)

        # Tab widget
        self._tab_widget = QTabWidget()
        self._tab_widget.setStyleSheet(f"""
            QTabWidget {{
                background-color: {Theme.BG_PRIMARY};
            }}
            QTabWidget::pane {{
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_LG}px;
                background-color: {Theme.BG_PRIMARY};
            }}
            QTabBar {{
                background-color: {Theme.BG_SECONDARY};
            }}
            QTabBar::tab {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                padding: {Theme.PADDING_MD}px {Theme.PADDING_LG}px;
                margin-right: {Theme.SPACING_SM}px;
            }}
            QTabBar::tab:selected {{
                background-color: {Theme.ACCENT};
                color: {Theme.TEXT_PRIMARY};
            }}
            QTabBar::tab:hover {{
                background-color: {Theme.BG_HOVER};
            }}
        """)
        layout.addWidget(self._tab_widget)

        # Create tabs
        self._installed_tab = InstalledTab(self._model_manager, self)
        self._available_tab = AvailableTab(self._model_manager, self)
        self._tab_widget.addTab(self._installed_tab, self.tr("Installed Models"))
        self._tab_widget.addTab(self._available_tab, self.tr("Available Models"))

        # Summary section
        self._summary_label = QLabel()
        self._summary_label.setStyleSheet(f"""
            font-family: {Theme.FONT_FAMILY};
            font-size: {Theme.FONT_SIZE_MD};
            color: {Theme.TEXT_SECONDARY};
        """)
        layout.addWidget(self._summary_label)
        self._update_summary()

        # Dialog buttons
        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        layout.addWidget(self._button_box)

        self.retranslate_ui()

    def _apply_style(self):
        """Apply stylesheet using Theme constants."""
        self.setStyleSheet(f"""
            ModelManagerDialog {{
                background-color: {Theme.BG_PRIMARY};
            }}
            QPushButton {{
                background-color: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD}px;
                font-family: {Theme.FONT_FAMILY};
                font-size: {Theme.FONT_SIZE_MD};
                padding: {Theme.PADDING_SM}px {Theme.PADDING_MD}px;
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_HOVER};
                border-color: {Theme.BORDER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_PRESSED};
            }}
        """)

    def _connect_signals(self):
        """Connect button signals."""
        self._button_box.rejected.connect(self.close)
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)

    def _update_summary(self):
        """Update the summary label."""
        installed = self._model_manager.get_installed_checkpoints()
        total_size = self._model_manager.get_total_size_mb()
        count = len(installed)

        self._summary_label.setText(
            self.tr("Total: {count} models installed, {size:.1f} MB")
            .format(count=count, size=total_size)
        )

    def _refresh_summary(self):
        """Refresh the summary after changes."""
        self._update_summary()

    def _on_refresh_clicked(self):
        """Refresh model lists."""
        self._model_manager.refresh()
        self._installed_tab.refresh()
        self._available_tab.refresh()
        self._update_summary()

    def _on_delete_clicked(self, checkpoint: "CheckpointInfo"):
        """Handle delete request (delegates to CheckpointItem)."""
        # This is called from CheckpointItem's delete button
        # Confirmation dialog and deletion logic is in CheckpointItem
        pass