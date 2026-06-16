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
    from ui.styles import apply_dark_theme
    apply_dark_theme(app)
    
    # Install Qt translator for self.tr() support (bridges gettext → Qt)
    logger.debug("Installing Qt translator")
    from core.i18n_qt import install_translator
    current_language = get_i18n().get_current_language()
    translator = install_translator(current_language)
    if translator:
        logger.info(f"Qt translator installed for: {current_language}")
    
    # Create VFIApp singleton (must happen after QApplication for QTimer)
    logger.info("Initializing VFIApp...")
    from ui.app import get_app
    vfi = get_app()
    
    # Start device polling
    vfi.start_device_polling()
    
    logger.info("Creating main window")
    window = MainWindow(vfi)
    window.show()
    
    logger.info("Application ready")
    ret = app.exec()
    
    # VFIApp shutdown is handled by MainWindow.closeEvent
    logger.info("VFI-gui exiting")
    sys.exit(ret)


if __name__ == "__main__":
    main()
