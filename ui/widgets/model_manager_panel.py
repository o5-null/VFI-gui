"""Model Manager Panel for VFI-gui.

A full-page panel for managing VFI models with extended functionality.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QLabel,
    QProgressBar,
    QGroupBox,
    QFrame,
    QSplitter,
    QMessageBox,
    QMenu,
    QHeaderView,
    QComboBox,
    QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QAction

from core import tr
from core.models import (
    ModelManager,
    ModelTypeInfo,
    CheckpointInfo,
    ModelStatus,
)

if TYPE_CHECKING:
    from core.config import Config


class DownloadWorker(QThread):
    """Worker thread for downloading models."""

    progress = pyqtSignal(int, str)  # progress%, message
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, urls: List[str], target_path: Path, parent=None):
        super().__init__(parent)
        self.urls = urls
        self.target_path = target_path
        self._cancelled = False

    def run(self):
        """Download the model."""
        from core.network import download_with_retry, ProxyConfig

        self.target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.progress.emit(0, tr("Starting download..."))

            def progress_hook(block_num, block_size, total_size):
                """Progress callback."""
                if self._cancelled:
                    raise KeyboardInterrupt()

                if total_size > 0:
                    progress = int((block_num * block_size / total_size) * 100)
                    progress = min(progress, 100)
                    self.progress.emit(progress, tr("Downloading... {}%").format(progress))

            # Download with proxy support
            download_with_retry(
                self.urls,
                self.target_path,
                progress_callback=progress_hook,
            )

            self.finished.emit(True, tr("Downloaded: {}").format(self.target_path.name))

        except KeyboardInterrupt:
            self.finished.emit(False, tr("Download cancelled"))
        except Exception as e:
            self.finished.emit(False, tr("Download failed: {}").format(str(e)))

    def cancel(self):
        """Cancel the download."""
        self._cancelled = True


class ModelManagerPanel(QWidget):
    """Full-page panel for model management.

    Features:
    - Tree view of all model types and checkpoints
    - Status indicators (installed/downloadable)
    - Download progress
    - Delete installed models
    - Import external models
    - Model directory configuration
    """

    # Signals
    model_downloaded = pyqtSignal(str, str)  # model_type, ckpt_name
    model_deleted = pyqtSignal(str, str)  # model_type, ckpt_name

    def __init__(
        self,
        config: "Config",
        models_dir: Optional[str] = None,
        model_manager: Optional[ModelManager] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._config = config
        models_dir = models_dir or str(config.vapoursynth.get_models_dir())
        
        # Use provided model manager or create one
        if model_manager is not None:
            self._model_manager = model_manager
        else:
            from core.config_provider import get_config
            self._model_manager = ModelManager(get_config())
        
        self._download_worker: Optional[DownloadWorker] = None

        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel(tr("Model Manager"))
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Model directory
        dir_label = QLabel(tr("Models Directory:"))
        header_layout.addWidget(dir_label)

        self.dir_edit = QLineEdit()
        self.dir_edit.setText(str(self._model_manager._ckpts_dir))
        self.dir_edit.setReadOnly(True)
        self.dir_edit.setMinimumWidth(200)
        header_layout.addWidget(self.dir_edit)

        self.browse_btn = QPushButton(tr("Browse..."))
        self.browse_btn.clicked.connect(self._on_browse_dir)
        header_layout.addWidget(self.browse_btn)

        # Refresh button
        self.refresh_btn = QPushButton(tr("Refresh"))
        self.refresh_btn.setFixedWidth(80)
        self.refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(self.refresh_btn)

        layout.addLayout(header_layout)

        # Summary info
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #808080; font-size: 11px;")
        layout.addWidget(self.summary_label)

        # Model tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([tr("Model"), tr("Status"), tr("Size")])
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        # Set column widths
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 120)
        header.resizeSection(2, 100)

        layout.addWidget(self.tree, 1)

        # Progress section (hidden by default)
        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #808080; font-size: 11px;")
        progress_layout.addWidget(self.progress_label)

        cancel_layout = QHBoxLayout()
        cancel_layout.addStretch()
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.clicked.connect(self._cancel_download)
        cancel_layout.addWidget(self.cancel_btn)
        progress_layout.addLayout(cancel_layout)

        self.progress_frame.hide()
        layout.addWidget(self.progress_frame)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.download_all_btn = QPushButton(tr("Download All Missing"))
        self.download_all_btn.clicked.connect(self._download_all_missing)
        btn_layout.addWidget(self.download_all_btn)

        self.import_btn = QPushButton(tr("Import Model..."))
        self.import_btn.clicked.connect(self._on_import_model)
        btn_layout.addWidget(self.import_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

    def retranslate_ui(self):
        """Retranslate UI elements after language change."""
        # Update tree headers
        self.tree.setHeaderLabels([tr("Model"), tr("Status"), tr("Size")])

        # Update buttons
        self.refresh_btn.setText(tr("Refresh"))
        self.cancel_btn.setText(tr("Cancel"))
        self.download_all_btn.setText(tr("Download All Missing"))
        self.import_btn.setText(tr("Import Model..."))
        self.browse_btn.setText(tr("Browse..."))

        # Refresh the tree content
        self.refresh()

    def refresh(self):
        """Refresh the model list."""
        self.tree.clear()

        # Get current selected model from config
        current_model_type = self._config.get("pipeline.interpolation.model_type", "rife").lower()
        current_model_version = self._config.get("pipeline.interpolation.model_version", "")

        # Scan models
        self._model_manager.refresh()
        models = self._model_manager.get_model_types()
        unknown = self._model_manager.scan_unknown_checkpoints()

        total_installed = 0
        total_size = 0.0

        # Add model types
        for model_type, type_info in sorted(models.items()):
            type_item = QTreeWidgetItem()
            type_item.setText(0, f"{type_info.display_name} ({type_info.installed_count}/{type_info.total_count})")
            type_item.setData(0, Qt.ItemDataRole.UserRole, ("type", model_type))

            # Highlight if this is the current model type
            is_current_type = model_type.lower() == current_model_type
            if is_current_type:
                type_item.setBackground(0, Qt.GlobalColor.darkBlue)
                type_item.setForeground(0, Qt.GlobalColor.white)

            # Set icon based on status
            if type_info.installed_count == type_info.total_count:
                type_item.setText(1, tr("Complete"))
                type_item.setForeground(1, Qt.GlobalColor.darkGreen)
            elif type_info.installed_count > 0:
                type_item.setText(1, tr("Partial"))
                type_item.setForeground(1, Qt.GlobalColor.darkYellow)
            else:
                type_item.setText(1, tr("Empty"))
                type_item.setForeground(1, Qt.GlobalColor.gray)

            # Add checkpoints
            for ckpt_info in type_info.checkpoints:
                ckpt_item = QTreeWidgetItem()
                ckpt_item.setText(0, ckpt_info.name)
                ckpt_item.setData(0, Qt.ItemDataRole.UserRole, ("ckpt", model_type, ckpt_info.name))

                if ckpt_info.is_installed:
                    ckpt_item.setText(1, tr("Installed"))
                    ckpt_item.setForeground(1, Qt.GlobalColor.darkGreen)
                    ckpt_item.setText(2, f"{ckpt_info.size_mb:.1f} MB")
                    total_installed += 1
                    total_size += ckpt_info.size_mb

                    # Highlight if this is the currently selected model
                    if is_current_type and self._is_current_checkpoint(model_type, ckpt_info.name, current_model_version):
                        ckpt_item.setBackground(0, Qt.GlobalColor.darkGreen)
                        ckpt_item.setForeground(0, Qt.GlobalColor.white)
                        ckpt_item.setText(0, f"★ {ckpt_info.name}")  # Add star indicator
                else:
                    ckpt_item.setText(1, tr("Available"))
                    ckpt_item.setForeground(1, Qt.GlobalColor.gray)

                type_item.addChild(ckpt_item)

            self.tree.addTopLevelItem(type_item)
            type_item.setExpanded(type_info.installed_count > 0 or is_current_type)

        # Add unknown checkpoints
        if unknown:
            unknown_item = QTreeWidgetItem()
            unknown_item.setText(0, tr("Other Models"))
            unknown_item.setText(1, f"{len(unknown)}")
            unknown_item.setForeground(1, Qt.GlobalColor.gray)
            unknown_item.setData(0, Qt.ItemDataRole.UserRole, ("unknown",))

            for ckpt_info in unknown:
                ckpt_item = QTreeWidgetItem()
                ckpt_item.setText(0, ckpt_info.name)
                ckpt_item.setText(1, tr("Installed"))
                ckpt_item.setForeground(1, Qt.GlobalColor.darkGreen)
                ckpt_item.setText(2, f"{ckpt_info.size_mb:.1f} MB")
                ckpt_item.setData(0, Qt.ItemDataRole.UserRole, ("unknown_ckpt", ckpt_info))
                unknown_item.addChild(ckpt_item)

                total_installed += 1
                total_size += ckpt_info.size_mb

            self.tree.addTopLevelItem(unknown_item)

        # Update summary
        self.summary_label.setText(
            tr("Total: {} models installed, {:.1f} MB").format(total_installed, total_size)
        )

        # Update directory
        self.dir_edit.setText(str(self._model_manager._ckpts_dir))

    def _is_current_checkpoint(self, model_type: str, ckpt_name: str, current_version: str) -> bool:
        """Check if this checkpoint is the currently selected one.

        Args:
            model_type: Model type name
            ckpt_name: Checkpoint filename
            current_version: Current version from config

        Returns:
            True if this is the current checkpoint
        """
        # Map checkpoint names to version strings
        version_map = {
            "rife": {
                "rife47.pth": "4.7",
                "rife49.pth": "4.9",
                "rife417.pth": "4.17",
                "rife426.pth": "4.26",
            },
            "film": {
                "film_net_fp32.pt": "fp32",
            },
            "ifrnet": {
                "IFRNet_S_Vimeo90K.pth": "S_Vimeo90K",
                "IFRNet_L_Vimeo90K.pth": "L_Vimeo90K",
            },
            "amt": {
                "amt-s.pth": "s",
                "amt-g.pth": "g",
            },
        }
        
        if model_type in version_map:
            ckpt_version = version_map[model_type].get(ckpt_name, "")
            if current_version and ckpt_version == current_version:
                return True
            # If no version specified, check if this is the default
            if not current_version:
                defaults = {"rife": "4.22", "film": "fp32", "ifrnet": "L_Vimeo90K", "amt": "s"}
                if model_type in defaults and ckpt_version == defaults[model_type]:
                    return True
        
        return False

    def _on_browse_dir(self):
        """Browse for models directory."""
        current_dir = self.dir_edit.text()
        new_dir = QFileDialog.getExistingDirectory(
            self,
            tr("Select Models Directory"),
            current_dir
        )

        if new_dir:
            self._model_manager.set_models_dir(new_dir)
            if self._config:
                self._config.set("paths.models_dir", new_dir)
                self._config.save()
            self.refresh()

    def _on_import_model(self):
        """Import an external model file."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            tr("Select Model Files"),
            "",
            tr("Model Files (*.onnx *.engine *.pth *.pt *.bin);;All Files (*)")
        )

        if not file_paths:
            return

        # Import each file
        imported = 0
        for file_path in file_paths:
            path = Path(file_path)
            target_dir = self._model_manager._ckpts_dir / "imported"
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / path.name

            if target_path.exists():
                reply = QMessageBox.question(
                    self,
                    tr("File Exists"),
                    tr("{} already exists. Overwrite?").format(path.name),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    continue

            try:
                import shutil
                shutil.copy2(path, target_path)
                imported += 1
            except Exception as e:
                QMessageBox.warning(
                    self,
                    tr("Import Failed"),
                    tr("Failed to import {}: {}").format(path.name, str(e))
                )

        if imported > 0:
            self.refresh()
            QMessageBox.information(
                self,
                tr("Import Complete"),
                tr("Imported {} model(s) to imported/ directory").format(imported)
            )

    def _show_context_menu(self, pos):
        """Show context menu for tree items."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        menu = QMenu(self)

        if data[0] == "ckpt":
            model_type, ckpt_name = data[1], data[2]
            ckpt_info = self._get_checkpoint_info(model_type, ckpt_name)

            if ckpt_info:
                if ckpt_info.is_installed:
                    # Delete action
                    delete_action = QAction(tr("Delete"), self)
                    delete_action.triggered.connect(
                        lambda: self._delete_checkpoint(model_type, ckpt_name)
                    )
                    menu.addAction(delete_action)

                    # Open folder action
                    open_folder_action = QAction(tr("Open Folder"), self)
                    open_folder_action.triggered.connect(
                        lambda: self._open_checkpoint_folder(ckpt_info.path)
                    )
                    menu.addAction(open_folder_action)
                else:
                    # Download action
                    download_action = QAction(tr("Download"), self)
                    download_action.triggered.connect(
                        lambda: self._download_checkpoint(ckpt_info)
                    )
                    menu.addAction(download_action)

        elif data[0] == "unknown_ckpt":
            ckpt_info = data[1]
            delete_action = QAction(tr("Delete"), self)
            delete_action.triggered.connect(
                lambda: self._delete_unknown_checkpoint(ckpt_info)
            )
            menu.addAction(delete_action)

            open_folder_action = QAction(tr("Open Folder"), self)
            open_folder_action.triggered.connect(
                lambda: self._open_checkpoint_folder(ckpt_info.path)
            )
            menu.addAction(open_folder_action)

        elif data[0] == "type":
            model_type = data[1]
            type_folder_path = self._model_manager._ckpts_dir / model_type
            if type_folder_path.exists():
                open_folder_action = QAction(tr("Open Folder"), self)
                open_folder_action.triggered.connect(
                    lambda: self._open_folder(type_folder_path)
                )
                menu.addAction(open_folder_action)

        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _get_checkpoint_info(self, model_type: str, ckpt_name: str) -> Optional[CheckpointInfo]:
        """Get checkpoint info from model manager."""
        return self._model_manager.get_checkpoint_info(model_type, ckpt_name)

    def _download_checkpoint(self, ckpt_info: CheckpointInfo):
        """Download a checkpoint."""
        if not ckpt_info.download_urls:
            QMessageBox.warning(
                self,
                tr("No Download URL"),
                tr("No download URL available for this checkpoint."),
            )
            return

        target_path = self._model_manager._ckpts_dir / ckpt_info.model_type / ckpt_info.name

        # Start download
        self._start_download(ckpt_info.download_urls, target_path, ckpt_info.model_type, ckpt_info.name)

    def _start_download(self, urls: List[str], target_path: Path, model_type: str, ckpt_name: str):
        """Start download worker."""
        self.progress_frame.show()
        self.tree.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.download_all_btn.setEnabled(False)

        self._download_worker = DownloadWorker(urls, target_path, self)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(
            lambda success, msg: self._on_download_finished(success, msg, model_type, ckpt_name)
        )
        self._download_worker.start()

    def _on_download_progress(self, progress: int, message: str):
        """Handle download progress."""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)

    def _on_download_finished(self, success: bool, message: str, model_type: str, ckpt_name: str):
        """Handle download completion."""
        self.progress_frame.hide()
        self.tree.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.download_all_btn.setEnabled(True)

        if success:
            self.model_downloaded.emit(model_type, ckpt_name)
            self.refresh()
        else:
            QMessageBox.warning(self, tr("Download Failed"), message)

        self._download_worker = None

    def _cancel_download(self):
        """Cancel ongoing download."""
        if self._download_worker:
            self._download_worker.cancel()

    def _delete_checkpoint(self, model_type: str, ckpt_name: str):
        """Delete an installed checkpoint."""
        reply = QMessageBox.question(
            self,
            tr("Delete Model"),
            tr("Are you sure you want to delete {}?").format(ckpt_name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            ckpt_path = self._model_manager._ckpts_dir / model_type / ckpt_name
            if ckpt_path.exists():
                ckpt_path.unlink()
                self.model_deleted.emit(model_type, ckpt_name)
                self.refresh()

    def _delete_unknown_checkpoint(self, ckpt_info: CheckpointInfo):
        """Delete an unknown checkpoint."""
        if ckpt_info.path and ckpt_info.path.exists():
            reply = QMessageBox.question(
                self,
                tr("Delete Model"),
                tr("Are you sure you want to delete {}?").format(ckpt_info.name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                ckpt_info.path.unlink()
                self.refresh()

    def _open_checkpoint_folder(self, path: Optional[Path]):
        """Open the checkpoint folder in file manager."""
        if path and path.exists():
            import subprocess
            import sys

            if sys.platform == "win32":
                subprocess.run(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", str(path)])
            else:
                subprocess.run(["xdg-open", str(path.parent)])

    def _open_folder(self, path: Optional[Path]):
        """Open the folder in file manager."""
        if path and path.exists():
            import subprocess
            import sys

            if sys.platform == "win32":
                subprocess.run(["explorer", str(path)])
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])

    def _download_all_missing(self):
        """Download all missing checkpoints."""
        missing = self._model_manager.get_missing_checkpoints()

        if not missing:
            QMessageBox.information(
                self,
                tr("No Missing Models"),
                tr("All models are already installed."),
            )
            return

        # Confirm download
        reply = QMessageBox.question(
            self,
            tr("Download All Missing"),
            tr("Download {} missing models?").format(len(missing)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Download first one, rest will be queued
        self._download_queue = list(missing)
        self._download_next_in_queue()

    def _download_next_in_queue(self):
        """Download next model in queue."""
        if not hasattr(self, "_download_queue") or not self._download_queue:
            return

        ckpt_info = self._download_queue.pop(0)
        target_path = self._model_manager._ckpts_dir / ckpt_info.model_type / ckpt_info.name

        self._start_download(
            ckpt_info.download_urls,
            target_path,
            ckpt_info.model_type,
            ckpt_info.name
        )

    def get_installed_models(self) -> Dict[str, List[str]]:
        """Get dictionary of installed models by type.

        Returns:
            Dict mapping model type to list of checkpoint names.
        """
        result: Dict[str, List[str]] = {}

        for ckpt in self._model_manager.get_installed_checkpoints():
            if ckpt.model_type not in result:
                result[ckpt.model_type] = []
            result[ckpt.model_type].append(ckpt.name)

        return result
