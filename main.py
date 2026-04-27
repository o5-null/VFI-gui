#!/usr/bin/env python3
"""
VFI-gui - Video Frame Interpolation GUI
A PyQt6 desktop application for VSGAN-tensorrt-docker video processing workflow.
"""

import sys

# Initialize path manager FIRST - this sets up all application paths
from core.paths import paths

# Setup runtime environment (detect GPU, select cuda/xpu/cpu)
from core.runtime_manager import runtime_manager

# Auto-detect and activate runtime before any other imports
runtime_manager.auto_select_runtime()

# Setup VapourSynth portable paths before any imports
paths.setup_python_paths()

# Initialize logger early
from core.logger import setup_logger, logger
setup_logger()
logger.info("VFI-gui starting...")

# Log current device info
current = runtime_manager.current_runtime
if current:
    for rt_info in runtime_manager.get_available_runtimes():
        if rt_info.runtime_type == current:
            if rt_info.gpu_name:
                logger.info(f"Using device: {rt_info.gpu_name} ({current.value.upper()}, x{rt_info.gpu_count})")
            else:
                logger.info(f"Using device: {current.value.upper()}")
            break
    else:
        logger.info(f"Using device: {current.value.upper()}")
else:
    logger.info("Using device: Unknown")

# Ensure all directories exist
paths.ensure_dirs()

# Initialize i18n system before UI imports
from core.i18n import init_i18n, get_i18n, tr
init_i18n(str(paths.locales_dir))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    """Application entry point."""
    logger.info("Initializing application...")
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("VFI-gui")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("VFI")
    
    logger.debug("Setting up dark theme")
    # Set dark theme
    from ui.styles.stylesheet import DARK_THEME
    app.setStyleSheet(DARK_THEME)
    
    logger.info("Creating main window")
    window = MainWindow()
    window.show()
    
    logger.info("Application ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
