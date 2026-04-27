"""Dependency management for VFI-gui.

Handles detection and management of external dependencies like FFmpeg.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from loguru import logger


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    installed: bool
    version: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None


class FFmpegManager:
    """FFmpeg detection and management."""

    # Common FFmpeg binary names
    FFMPEG_NAMES = ["ffmpeg", "ffmpeg.exe"]
    FFPROBE_NAMES = ["ffprobe", "ffprobe.exe"]

    def __init__(self, custom_path: Optional[str] = None):
        """Initialize FFmpeg manager.

        Args:
            custom_path: Optional custom path to FFmpeg directory or executable.
        """
        self.custom_path = custom_path
        self._ffmpeg_path: Optional[str] = None
        self._ffprobe_path: Optional[str] = None
        self._version: Optional[str] = None

    def detect(self) -> DependencyInfo:
        """Detect FFmpeg installation.

        Returns:
            DependencyInfo with detection results.
        """
        # Try custom path first
        if self.custom_path:
            result = self._check_custom_path()
            if result.installed:
                return result

        # Try system PATH
        result = self._check_system_path()
        return result

    def _check_custom_path(self) -> DependencyInfo:
        """Check custom path for FFmpeg."""
        if not self.custom_path:
            return DependencyInfo(
                name="FFmpeg",
                installed=False,
                error="No custom path specified"
            )

        custom = Path(self.custom_path)

        # If it's a directory, look for ffmpeg inside
        if custom.is_dir():
            for name in self.FFMPEG_NAMES:
                ffmpeg_candidate = custom / name
                if ffmpeg_candidate.exists():
                    self._ffmpeg_path = str(ffmpeg_candidate)
                    break

            for name in self.FFPROBE_NAMES:
                ffprobe_candidate = custom / name
                if ffprobe_candidate.exists():
                    self._ffprobe_path = str(ffprobe_candidate)
                    break
        else:
            # Assume it's the ffmpeg executable path
            if custom.exists():
                self._ffmpeg_path = str(custom)
                # Try to find ffprobe in same directory
                ffprobe_path = custom.parent / self.FFPROBE_NAMES[0]
                if ffprobe_path.exists():
                    self._ffprobe_path = str(ffprobe_path)

        if self._ffmpeg_path:
            version = self._get_version(self._ffmpeg_path)
            if version:
                self._version = version
                return DependencyInfo(
                    name="FFmpeg",
                    installed=True,
                    version=version,
                    path=self._ffmpeg_path
                )
            else:
                return DependencyInfo(
                    name="FFmpeg",
                    installed=False,
                    path=self._ffmpeg_path,
                    error="FFmpeg found but failed to get version"
                )

        return DependencyInfo(
            name="FFmpeg",
            installed=False,
            error=f"FFmpeg not found in custom path: {self.custom_path}"
        )

    def _check_system_path(self) -> DependencyInfo:
        """Check system PATH for FFmpeg."""
        ffmpeg_path = shutil.which("ffmpeg")

        if ffmpeg_path:
            self._ffmpeg_path = ffmpeg_path
            self._ffprobe_path = shutil.which("ffprobe")

            version = self._get_version(ffmpeg_path)
            if version:
                self._version = version
                return DependencyInfo(
                    name="FFmpeg",
                    installed=True,
                    version=version,
                    path=ffmpeg_path
                )
            else:
                return DependencyInfo(
                    name="FFmpeg",
                    installed=False,
                    path=ffmpeg_path,
                    error="FFmpeg found but failed to get version"
                )

        return DependencyInfo(
            name="FFmpeg",
            installed=False,
            error="FFmpeg not found in system PATH"
        )

    def _get_version(self, ffmpeg_path: str) -> Optional[str]:
        """Get FFmpeg version string.

        Args:
            ffmpeg_path: Path to ffmpeg executable.

        Returns:
            Version string or None if failed.
        """
        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )

            if result.returncode == 0:
                # Parse version from output like "ffmpeg version 6.0"
                match = re.search(r"ffmpeg version\s+([^\s]+)", result.stdout)
                if match:
                    return match.group(1)

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to get FFmpeg version: {e}")

        return None

    def get_codecs(self) -> Tuple[List[str], List[str]]:
        """Get available encoders and decoders.

        Returns:
            Tuple of (encoders, decoders) lists.
        """
        if not self._ffmpeg_path:
            return [], []

        encoders = self._get_codec_list("-encoders")
        decoders = self._get_codec_list("-decoders")

        return encoders, decoders

    def _get_codec_list(self, flag: str) -> List[str]:
        """Get list of codecs from FFmpeg.

        Args:
            flag: FFmpeg flag (-encoders or -decoders).

        Returns:
            List of codec names.
        """
        if not self._ffmpeg_path:
            return []

        try:
            result = subprocess.run(
                [self._ffmpeg_path, flag],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )

            if result.returncode == 0:
                # Parse codec names from output
                codecs = []
                for line in result.stdout.split("\n"):
                    # Lines like " V..... libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)"
                    match = re.match(r"\s+[A-Z.]+\s+(\S+)", line)
                    if match:
                        codecs.append(match.group(1))
                return codecs

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to get codec list: {e}")

        return []

    def check_hardware_acceleration(self) -> dict:
        """Check available hardware acceleration methods.

        Returns:
            Dictionary with hardware acceleration availability.
        """
        if not self._ffmpeg_path:
            return {}

        hw_accels = {
            "nvenc": False,  # NVIDIA
            "qsv": False,    # Intel Quick Sync
            "vaapi": False,  # VAAPI (Linux)
            "videotoolbox": False,  # macOS
            "cuda": False,
            "amf": False,    # AMD
        }

        try:
            result = subprocess.run(
                [self._ffmpeg_path, "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )

            if result.returncode == 0:
                output = result.stdout.lower()
                hw_accels["nvenc"] = "cuda" in output or "nvenc" in output
                hw_accels["qsv"] = "qsv" in output
                hw_accels["vaapi"] = "vaapi" in output
                hw_accels["videotoolbox"] = "videotoolbox" in output
                hw_accels["cuda"] = "cuda" in output
                hw_accels["amf"] = "amf" in output

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to check hardware acceleration: {e}")

        return hw_accels

    @property
    def ffmpeg_path(self) -> Optional[str]:
        """Get FFmpeg executable path."""
        return self._ffmpeg_path

    @property
    def ffprobe_path(self) -> Optional[str]:
        """Get FFprobe executable path."""
        return self._ffprobe_path

    @property
    def version(self) -> Optional[str]:
        """Get FFmpeg version."""
        return self._version

    def is_available(self) -> bool:
        """Check if FFmpeg is available."""
        return self._ffmpeg_path is not None


class DependencyManager:
    """Manager for all external dependencies."""

    def __init__(self, config=None):
        """Initialize dependency manager.

        Args:
            config: Optional Config instance for path settings.
        """
        self.config = config
        self.ffmpeg = FFmpegManager()

    def check_all(self) -> dict:
        """Check all dependencies.

        Returns:
            Dictionary mapping dependency names to DependencyInfo.
        """
        results = {}

        # Check FFmpeg
        custom_ffmpeg = None
        if self.config:
            custom_ffmpeg = self.config.get("paths.ffmpeg_dir")
        self.ffmpeg = FFmpegManager(custom_ffmpeg)
        results["ffmpeg"] = self.ffmpeg.detect()

        return results

    def get_ffmpeg_info(self) -> DependencyInfo:
        """Get FFmpeg information."""
        return self.ffmpeg.detect()

    def set_ffmpeg_path(self, path: str) -> bool:
        """Set custom FFmpeg path.

        Args:
            path: Path to FFmpeg directory or executable.

        Returns:
            True if FFmpeg was found at the path.
        """
        self.ffmpeg = FFmpegManager(path)
        info = self.ffmpeg.detect()

        if info.installed and self.config:
            # Save to config
            path_obj = Path(path)
            if path_obj.is_file():
                ffmpeg_dir = str(path_obj.parent)
            else:
                ffmpeg_dir = path
            self.config.set("paths.ffmpeg_dir", ffmpeg_dir)
            self.config.save()

        return info.installed
