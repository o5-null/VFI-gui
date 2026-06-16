"""Download Worker for VFI-gui.

Background worker for downloading model files with progress reporting.
"""

from typing import List
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from core import tr


class DownloadWorker(QThread):
    """Worker thread for downloading models.

    Provides progress updates and cancellation support.
    """

    # Signals
    progress = pyqtSignal(int, str)  # progress%, message
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, urls: List[str], target_path: Path, parent=None):
        super().__init__(parent)
        self.urls = urls
        self.target_path = target_path
        self._cancelled = False

    def run(self):
        """Download the model."""
        from core.network import download_with_retry

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

            # Download with retry support
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