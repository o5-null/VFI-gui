"""Unified path management for VFI-gui.

This module provides centralized path management for all application directories.
All path-related operations should use this module instead of scattered Path() calls.

Usage:
    from core.paths import paths
    
    # Get application directories
    models_dir = paths.models_dir
    temp_dir = paths.temp_dir
    output_dir = paths.output_dir
    
    # Ensure directories exist
    paths.ensure_dirs()
"""

import os
from pathlib import Path
from typing import Optional
from loguru import logger


class PathManager:
    """Centralized path manager for VFI-gui application.
    
    Manages all application directories with consistent path resolution.
    Supports both portable (app-relative) and user-configured paths.
    """
    
    def __init__(self):
        """Initialize path manager with default paths."""
        # Application directory (where main.py is located)
        self._app_dir: Path = Path(__file__).parent.parent.resolve()
        
        # User config directory (now in app directory for portability)
        self._config_dir: Path = self._app_dir / "config"
        
        # Default paths (relative to app directory)
        self._models_dir: Optional[Path] = None
        self._temp_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._logs_dir: Optional[Path] = None
        self._locales_dir: Optional[Path] = None
        
        # VapourSynth portable paths
        self._vs_portable_dir: Optional[Path] = None
        
        # Initialize defaults
        self._init_defaults()
    
    def _init_defaults(self):
        """Initialize default paths."""
        # Core directories
        self._models_dir = self._app_dir / "models"
        self._temp_dir = self._app_dir / "temp"
        self._output_dir = self._app_dir / "output"
        self._logs_dir = self._app_dir / "logs"
        self._locales_dir = self._app_dir / "locales"
        
        # VapourSynth portable
        self._vs_portable_dir = self._app_dir / "plugin" / "vapoursynth-portable"

        # Runtime directory (virtual environments)
        self._runtime_dir: Optional[Path] = self._app_dir.parent / "runtime"

    # ==================== Properties ====================
    
    @property
    def app_dir(self) -> Path:
        """Application root directory."""
        return self._app_dir
    
    @property
    def config_dir(self) -> Path:
        """User configuration directory."""
        return self._config_dir
    
    @property
    def models_dir(self) -> Path:
        """Models directory (TensorRT engines, ONNX, PyTorch checkpoints)."""
        return self._models_dir or self._app_dir / "models"
    
    @models_dir.setter
    def models_dir(self, path: str | Path):
        """Set models directory."""
        self._models_dir = Path(path).resolve()
    
    @property
    def temp_dir(self) -> Path:
        """Temporary files directory."""
        return self._temp_dir or self._app_dir / "temp"
    
    @temp_dir.setter
    def temp_dir(self, path: str | Path):
        """Set temp directory."""
        self._temp_dir = Path(path).resolve()
    
    @property
    def output_dir(self) -> Path:
        """Output files directory."""
        return self._output_dir or self._app_dir / "output"
    
    @output_dir.setter
    def output_dir(self, path: str | Path):
        """Set output directory."""
        self._output_dir = Path(path).resolve()
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        return self._logs_dir or self._app_dir / "logs"
    
    @property
    def locales_dir(self) -> Path:
        """Localization files directory."""
        return self._locales_dir or self._app_dir / "locales"
    
    @property
    def vs_portable_dir(self) -> Path:
        """VapourSynth portable directory."""
        return self._vs_portable_dir or self._app_dir / "plugin" / "vapoursynth-portable"
    
    @property
    def vs_site_packages(self) -> Path:
        """VapourSynth site-packages directory."""
        return self.vs_portable_dir / "Lib" / "site-packages"
    
    @property
    def vs_scripts_dir(self) -> Path:
        """VapourSynth scripts directory."""
        return self.vs_portable_dir / "vs-scripts"
    
    @property
    def vsgan_src_dir(self) -> Path:
        """VSGAN source directory."""
        return self._app_dir / "core" / "vsgan" / "src"

    @property
    def runtime_dir(self) -> Path:
        """Runtime directory containing virtual environments (cuda/, xpu/)."""
        return self._runtime_dir or self._app_dir.parent / "runtime"
    
    # ==================== Path Checks ====================
    
    @property
    def has_vs_portable(self) -> bool:
        """Check if VapourSynth portable is available."""
        return self.vs_portable_dir.exists()
    
    @property
    def has_models(self) -> bool:
        """Check if models directory exists and has content."""
        return self.models_dir.exists() and any(self.models_dir.iterdir())
    
    # ==================== Directory Management ====================
    
    def ensure_dir(self, path: Path) -> Path:
        """Ensure a directory exists.
        
        Args:
            path: Directory path to ensure
            
        Returns:
            The ensured path
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def ensure_dirs(self):
        """Ensure all core directories exist."""
        dirs = [
            self.models_dir,
            self.temp_dir,
            self.output_dir,
            self.logs_dir,
            self.config_dir,
        ]
        
        for dir_path in dirs:
            self.ensure_dir(dir_path)
            logger.debug(f"Ensured directory: {dir_path}")
    
    # ==================== Configuration ====================
    
    def configure(
        self,
        models_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """Configure custom paths.
        
        Args:
            models_dir: Custom models directory
            temp_dir: Custom temp directory
            output_dir: Custom output directory
        """
        if models_dir:
            self._models_dir = Path(models_dir).resolve()
            logger.info(f"Models directory set to: {self._models_dir}")
        
        if temp_dir:
            self._temp_dir = Path(temp_dir).resolve()
            logger.info(f"Temp directory set to: {self._temp_dir}")
        
        if output_dir:
            self._output_dir = Path(output_dir).resolve()
            logger.info(f"Output directory set to: {self._output_dir}")
        
        # Ensure configured directories exist
        self.ensure_dirs()
    
    def load_from_config(self, config: dict):
        """Load paths from configuration dict.
        
        Args:
            config: Configuration dictionary with 'paths' key
        """
        paths_config = config.get("paths", {})
        
        if "models_dir" in paths_config:
            self._models_dir = Path(paths_config["models_dir"]).resolve()
        
        if "output_dir" in paths_config:
            self._output_dir = Path(paths_config["output_dir"]).resolve()
        
        self.ensure_dirs()
    
    def to_config(self) -> dict:
        """Export current paths to configuration dict.
        
        Returns:
            Dictionary with paths configuration
        """
        return {
            "paths": {
                "models_dir": str(self.models_dir),
                "temp_dir": str(self.temp_dir),
                "output_dir": str(self.output_dir),
            }
        }
    
    # ==================== Python Path Setup ====================
    
    def setup_python_paths(self):
        """Setup Python paths for VapourSynth portable.
        
        Adds VapourSynth site-packages, vs-scripts, and VSGAN src
        to sys.path if they exist.
        """
        import sys
        
        paths_to_add = [
            self.vs_site_packages,
            self.vs_scripts_dir,
            self.vsgan_src_dir,
        ]
        
        for path in paths_to_add:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
                logger.debug(f"Added to sys.path: {path}")
    
    # ==================== Utility Methods ====================
    
    def get_checkpoint_path(self, model_type: str, checkpoint_name: str) -> Path:
        """Get path to a model checkpoint.
        
        Args:
            model_type: Model type (e.g., "rife", "film")
            checkpoint_name: Checkpoint filename (e.g., "rife49.pth")
            
        Returns:
            Full path to the checkpoint
        """
        return self.models_dir / model_type / checkpoint_name
    
    def get_engine_path(self, engine_name: str) -> Path:
        """Get path to a TensorRT engine.
        
        Args:
            engine_name: Engine filename
            
        Returns:
            Full path to the engine
        """
        return self.models_dir / engine_name
    
    def get_temp_file(self, filename: str) -> Path:
        """Get path to a temporary file.
        
        Args:
            filename: Temporary filename
            
        Returns:
            Full path in temp directory
        """
        return self.temp_dir / filename
    
    def get_output_file(self, input_path: str | Path, suffix: str = "") -> Path:
        """Generate output file path from input.
        
        Args:
            input_path: Input video path
            suffix: Optional suffix to add to filename
            
        Returns:
            Output file path
        """
        input_path = Path(input_path)
        name = input_path.stem
        if suffix:
            name = f"{name}_{suffix}"
        return self.output_dir / f"{name}.mp4"
    
    def cleanup_temp(self):
        """Clean up temporary files."""
        import shutil
        
        if self.temp_dir.exists():
            for item in self.temp_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"Failed to clean {item}: {e}")
    
    def __repr__(self) -> str:
        return (
            f"PathManager("
            f"app={self.app_dir}, "
            f"models={self.models_dir}, "
            f"temp={self.temp_dir}, "
            f"output={self.output_dir})"
        )


# Global singleton instance
paths = PathManager()


# Convenience functions for backward compatibility
def get_app_dir() -> Path:
    """Get application directory."""
    return paths.app_dir


def get_models_dir() -> Path:
    """Get models directory."""
    return paths.models_dir


def get_temp_dir() -> Path:
    """Get temp directory."""
    return paths.temp_dir


def get_output_dir() -> Path:
    """Get output directory."""
    return paths.output_dir


def configure_paths(
    models_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Configure application paths."""
    paths.configure(models_dir=models_dir, temp_dir=temp_dir, output_dir=output_dir)
