"""Codec Manager for VFI-gui.

Provides centralized management of video encoding/decoding settings:
- Codec configuration management
- FFmpeg command building
- Hardware acceleration detection
- Preset management
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class CodecType(Enum):
    """Codec type enumeration."""
    HARDWARE = "hardware"
    SOFTWARE = "software"


class CodecVendor(Enum):
    """Codec vendor enumeration."""
    NVIDIA = "nvidia"
    INTEL = "intel"
    AMD = "amd"
    APPLE = "apple"
    SOFTWARE = "software"


@dataclass
class CodecInfo:
    """Information about a codec."""
    id: str
    name: str
    type: CodecType
    vendor: CodecVendor
    description: str = ""
    supports_crf: bool = False
    supports_cq: bool = False
    supports_bitrate: bool = True
    pixel_formats: List[str] = field(default_factory=list)
    presets: List[str] = field(default_factory=list)
    preset_names: Dict[str, str] = field(default_factory=dict)
    default_preset: str = ""
    quality_range: Tuple[int, int] = (0, 51)
    quality_default: int = 22


@dataclass
class CodecConfig:
    """Codec configuration settings."""
    codec: str = "hevc_nvenc"
    rate_control: str = "cq"  # cq, crf, vbr, cbr, cqp
    quality: int = 22
    bitrate: int = 8000  # kbps
    max_bitrate: Optional[int] = None
    preset: str = "p4"
    pixel_format: str = "auto"
    profile: str = "auto"
    level: str = "auto"
    gop_size: int = 0  # 0 = auto
    keyint: int = 0  # 0 = auto
    bframes: int = 3
    multipass: bool = False
    audio_copy: bool = True
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    custom_params: str = ""
    # Output mode
    output_mode: str = "video"  # "video" or "images"
    image_format: str = "png"  # png, jpg, tiff, exr
    image_quality: int = 95  # for jpg (1-100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "codec": self.codec,
            "rate_control": self.rate_control,
            "quality": self.quality,
            "bitrate": self.bitrate,
            "max_bitrate": self.max_bitrate,
            "preset": self.preset,
            "pixel_format": self.pixel_format,
            "profile": self.profile,
            "level": self.level,
            "gop_size": self.gop_size,
            "keyint": self.keyint,
            "bframes": self.bframes,
            "multipass": self.multipass,
            "audio_copy": self.audio_copy,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
            "custom_params": self.custom_params,
            "output_mode": self.output_mode,
            "image_format": self.image_format,
            "image_quality": self.image_quality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodecConfig":
        """Create from dictionary."""
        return cls(
            codec=data.get("codec", "hevc_nvenc"),
            rate_control=data.get("rate_control", "cq"),
            quality=data.get("quality", 22),
            bitrate=data.get("bitrate", 8000),
            max_bitrate=data.get("max_bitrate"),
            preset=data.get("preset", "p4"),
            pixel_format=data.get("pixel_format", "auto"),
            profile=data.get("profile", "auto"),
            level=data.get("level", "auto"),
            gop_size=data.get("gop_size", 0),
            keyint=data.get("keyint", 0),
            bframes=data.get("bframes", 3),
            multipass=data.get("multipass", False),
            audio_copy=data.get("audio_copy", True),
            audio_codec=data.get("audio_codec", "aac"),
            audio_bitrate=data.get("audio_bitrate", "192k"),
            custom_params=data.get("custom_params", ""),
            output_mode=data.get("output_mode", "video"),
            image_format=data.get("image_format", "png"),
            image_quality=data.get("image_quality", 95),
        )


# Codec definitions
CODEC_DEFINITIONS: Dict[str, CodecInfo] = {
    # NVIDIA NVENC codecs
    "hevc_nvenc": CodecInfo(
        id="hevc_nvenc",
        name="HEVC (H.265) NVENC",
        type=CodecType.HARDWARE,
        vendor=CodecVendor.NVIDIA,
        description="NVIDIA hardware H.265 encoder (requires NVIDIA GPU)",
        supports_cq=True,
        supports_bitrate=True,
        pixel_formats=["yuv420p", "yuv444p", "p010le"],
        presets=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        preset_names={
            "p1": "P1 - Fastest (lowest quality)",
            "p2": "P2 - Faster (lower quality)",
            "p3": "P3 - Fast (low quality)",
            "p4": "P4 - Medium (balanced)",
            "p5": "P5 - Slow (good quality)",
            "p6": "P6 - Slower (better quality)",
            "p7": "P7 - Slowest (best quality)",
        },
        default_preset="p4",
        quality_range=(0, 51),
        quality_default=22,
    ),
    "av1_nvenc": CodecInfo(
        id="av1_nvenc",
        name="AV1 NVENC",
        type=CodecType.HARDWARE,
        vendor=CodecVendor.NVIDIA,
        description="NVIDIA hardware AV1 encoder (requires RTX 40 series)",
        supports_cq=True,
        supports_bitrate=True,
        pixel_formats=["yuv420p", "p010le"],
        presets=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        preset_names={
            "p1": "P1 - Fastest",
            "p2": "P2 - Faster",
            "p3": "P3 - Fast",
            "p4": "P4 - Medium",
            "p5": "P5 - Slow",
            "p6": "P6 - Slower",
            "p7": "P7 - Slowest",
        },
        default_preset="p4",
        quality_range=(0, 51),
        quality_default=22,
    ),
    # Software codecs
    "libx265": CodecInfo(
        id="libx265",
        name="HEVC (H.265)",
        type=CodecType.SOFTWARE,
        vendor=CodecVendor.SOFTWARE,
        description="Software H.265 encoder (CPU-based, high quality)",
        supports_crf=True,
        supports_bitrate=True,
        pixel_formats=["yuv420p", "yuv422p", "yuv444p", "p010le"],
        presets=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        default_preset="medium",
        quality_range=(0, 51),
        quality_default=22,
    ),
    "libx264": CodecInfo(
        id="libx264",
        name="H.264/AVC",
        type=CodecType.SOFTWARE,
        vendor=CodecVendor.SOFTWARE,
        description="Software H.264 encoder (CPU-based, widely compatible)",
        supports_crf=True,
        supports_bitrate=True,
        pixel_formats=["yuv420p", "yuv422p", "yuv444p"],
        presets=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        default_preset="medium",
        quality_range=(0, 51),
        quality_default=23,
    ),
    "libaom-av1": CodecInfo(
        id="libaom-av1",
        name="AV1 (AOM)",
        type=CodecType.SOFTWARE,
        vendor=CodecVendor.SOFTWARE,
        description="AOMedia AV1 encoder (slow but excellent compression)",
        supports_crf=True,
        supports_bitrate=True,
        pixel_formats=["yuv420p", "yuv444p", "p010le"],
        presets=["0", "1", "2", "3", "4", "5", "6"],
        preset_names={
            "0": "0 - Best quality (slowest)",
            "1": "1 - Better quality",
            "2": "2 - Good quality",
            "3": "3 - Balanced",
            "4": "4 - Faster",
            "5": "5 - Much faster",
            "6": "6 - Fastest",
        },
        default_preset="4",
        quality_range=(0, 63),
        quality_default=30,
    ),
    "libvpx-vp9": CodecInfo(
        id="libvpx-vp9",
        name="VP9",
        type=CodecType.SOFTWARE,
        vendor=CodecVendor.SOFTWARE,
        description="Google VP9 encoder (good web compatibility)",
        supports_crf=True,
        supports_bitrate=True,
        pixel_formats=["yuv420p", "yuva420p", "yuv422p", "yuv440p", "yuv444p"],
        presets=["0", "1", "2", "3", "4", "5"],
        default_preset="1",
        quality_range=(0, 63),
        quality_default=31,
    ),
}


class CodecManager:
    """Manager for video encoding/decoding settings."""

    def __init__(self):
        """Initialize codec manager."""
        self._config = CodecConfig()
        self._available_encoders: List[str] = []
        self._hardware_accels: Dict[str, bool] = {}

    def get_codec_info(self, codec_id: str) -> Optional[CodecInfo]:
        """Get codec information by ID.

        Args:
            codec_id: Codec identifier

        Returns:
            CodecInfo or None if not found
        """
        return CODEC_DEFINITIONS.get(codec_id)

    def get_all_codecs(self) -> Dict[str, CodecInfo]:
        """Get all available codec definitions.

        Returns:
            Dictionary of codec ID to CodecInfo
        """
        return CODEC_DEFINITIONS.copy()

    def get_hardware_codecs(self) -> Dict[str, CodecInfo]:
        """Get hardware-accelerated codecs.

        Returns:
            Dictionary of hardware codec ID to CodecInfo
        """
        return {
            k: v for k, v in CODEC_DEFINITIONS.items()
            if v.type == CodecType.HARDWARE
        }

    def get_software_codecs(self) -> Dict[str, CodecInfo]:
        """Get software codecs.

        Returns:
            Dictionary of software codec ID to CodecInfo
        """
        return {
            k: v for k, v in CODEC_DEFINITIONS.items()
            if v.type == CodecType.SOFTWARE
        }

    def set_config(self, config: CodecConfig) -> None:
        """Set codec configuration.

        Args:
            config: CodecConfig instance
        """
        self._config = config
        logger.debug(f"Codec config set: {config.codec}")

    def set_config_from_dict(self, data: Dict[str, Any]) -> None:
        """Set codec configuration from dictionary.

        Args:
            data: Configuration dictionary
        """
        self._config = CodecConfig.from_dict(data)
        logger.debug(f"Codec config set from dict: {self._config.codec}")

    def get_config(self) -> CodecConfig:
        """Get current codec configuration.

        Returns:
            Current CodecConfig instance
        """
        return self._config

    def get_config_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.to_dict()

    def build_ffmpeg_encode_args(
        self,
        width: int,
        height: int,
        fps: float,
        input_source: str = "pipe:",
        output_path: str = "",
        include_audio: bool = False,
        audio_source: Optional[str] = None,
    ) -> List[str]:
        """Build FFmpeg encoding command arguments.

        Args:
            width: Video width
            height: Video height
            fps: Frame rate
            input_source: Input source (default: pipe:)
            output_path: Output file path
            include_audio: Whether to include audio
            audio_source: Audio source file path

        Returns:
            List of FFmpeg command arguments
        """
        config = self._config
        codec_info = self.get_codec_info(config.codec)

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", input_source,
        ]

        # Add audio input if needed
        if include_audio and audio_source:
            cmd.extend(["-i", audio_source])

        # Build codec-specific arguments
        cmd.extend(self._build_codec_args(config, codec_info))

        # Add pixel format if specified
        if config.pixel_format != "auto":
            cmd.extend(["-pix_fmt", config.pixel_format])

        # Add profile if specified
        if config.profile != "auto" and codec_info:
            if config.codec in ["hevc_nvenc", "libx265"]:
                cmd.extend(["-profile:v", config.profile])
            elif config.codec in ["libx264"]:
                cmd.extend(["-profile:v", config.profile])

        # Add level if specified
        if config.level != "auto":
            cmd.extend(["-level:v", config.level])

        # Add GOP size if specified
        if config.gop_size > 0:
            cmd.extend(["-g", str(config.gop_size)])

        # Add keyint if specified
        if config.keyint > 0:
            cmd.extend(["-keyint_min", str(config.keyint)])

        # Add B-frames
        if config.bframes > 0:
            cmd.extend(["-bf", str(config.bframes)])

        # Add multi-pass for VBR
        if config.multipass and config.rate_control == "vbr":
            if config.codec in ["hevc_nvenc", "av1_nvenc"]:
                cmd.extend(["-multipass", "1"])

        # Add audio settings
        if include_audio and audio_source:
            if config.audio_copy:
                cmd.extend(["-c:a", "copy"])
            else:
                cmd.extend([
                    "-c:a", config.audio_codec,
                    "-b:a", config.audio_bitrate,
                ])
            cmd.extend(["-map", "0:v", "-map", "1:a?"])

        # Add custom parameters
        if config.custom_params:
            # Parse custom params (simple space-split, respecting quotes)
            import shlex
            custom_parts = shlex.split(config.custom_params)
            cmd.extend(custom_parts)

        # Add output path
        if output_path:
            cmd.append(output_path)

        logger.debug(f"FFmpeg args: {' '.join(cmd)}")
        return cmd

    def _build_codec_args(
        self,
        config: CodecConfig,
        codec_info: Optional[CodecInfo],
    ) -> List[str]:
        """Build codec-specific FFmpeg arguments.

        Args:
            config: Codec configuration
            codec_info: Codec information

        Returns:
            List of codec-specific arguments
        """
        args = []
        codec = config.codec

        if codec == "libx265":
            args.extend([
                "-c:v", "libx265",
                "-crf", str(config.quality),
                "-preset", config.preset,
            ])
        elif codec == "libx264":
            args.extend([
                "-c:v", "libx264",
                "-crf", str(config.quality),
                "-preset", config.preset,
            ])
        elif codec == "hevc_nvenc":
            if config.rate_control == "cq":
                args.extend([
                    "-c:v", "hevc_nvenc",
                    "-rc:v", "vbr",
                    "-cq:v", str(config.quality),
                    "-preset:v", config.preset,
                ])
            elif config.rate_control == "vbr":
                args.extend([
                    "-c:v", "hevc_nvenc",
                    "-rc:v", "vbr",
                    "-b:v", f"{config.bitrate}k",
                    "-preset:v", config.preset,
                ])
                if config.max_bitrate:
                    args.extend(["-maxrate:v", f"{config.max_bitrate}k"])
            elif config.rate_control == "cbr":
                args.extend([
                    "-c:v", "hevc_nvenc",
                    "-rc:v", "cbr",
                    "-b:v", f"{config.bitrate}k",
                    "-preset:v", config.preset,
                ])
            else:
                args.extend([
                    "-c:v", "hevc_nvenc",
                    "-cq:v", str(config.quality),
                    "-preset:v", config.preset,
                ])
        elif codec == "av1_nvenc":
            if config.rate_control == "cq":
                args.extend([
                    "-c:v", "av1_nvenc",
                    "-rc:v", "vbr",
                    "-cq:v", str(config.quality),
                    "-preset:v", config.preset,
                ])
            elif config.rate_control == "vbr":
                args.extend([
                    "-c:v", "av1_nvenc",
                    "-rc:v", "vbr",
                    "-b:v", f"{config.bitrate}k",
                    "-preset:v", config.preset,
                ])
            else:
                args.extend([
                    "-c:v", "av1_nvenc",
                    "-cq:v", str(config.quality),
                    "-preset:v", config.preset,
                ])
        elif codec == "libaom-av1":
            args.extend([
                "-c:v", "libaom-av1",
                "-crf", str(config.quality),
                "-cpu-used", config.preset,
                "-strict", "experimental",
            ])
        elif codec == "libvpx-vp9":
            args.extend([
                "-c:v", "libvpx-vp9",
                "-crf", str(config.quality),
                "-speed", config.preset,
            ])
        else:
            # Generic fallback
            args.extend(["-c:v", codec])
            if codec_info and codec_info.supports_crf:
                args.extend(["-crf", str(config.quality)])

        return args

    def build_image_sequence_args(
        self,
        output_dir: str,
        image_format: str = "png",
        quality: int = 95,
        filename_pattern: str = "frame_%06d",
    ) -> List[str]:
        """Build FFmpeg arguments for image sequence output.

        Args:
            output_dir: Output directory path
            image_format: Image format (png, jpg, tiff, exr)
            quality: JPEG quality (1-100, only for jpg)
            filename_pattern: Filename pattern with frame number placeholder

        Returns:
            List of FFmpeg arguments for image output
        """
        # Determine extension and codec
        format_map = {
            "png": ("png", []),
            "jpg": ("mjpeg", ["-q:v", str(quality)]),
            "jpeg": ("mjpeg", ["-q:v", str(quality)]),
            "tiff": ("tiff", []),
            "exr": ("exr", []),
        }

        ext, extra_args = format_map.get(image_format, ("png", []))

        # Build output path
        output_path = str(Path(output_dir) / f"{filename_pattern}.{ext}")

        args = ["-c:v", ext] + extra_args + [output_path]

        return args

    def is_image_output(self) -> bool:
        """Check if current config is set to image output mode.

        Returns:
            True if output mode is images
        """
        return self._config.output_mode == "images"

    def detect_hardware_encoders(self) -> Dict[str, bool]:
        """Detect available hardware encoders.

        Returns:
            Dictionary mapping encoder name to availability
        """
        import subprocess

        result = {
            "nvenc": False,
            "qsv": False,
            "amf": False,
            "vaapi": False,
            "videotoolbox": False,
        }

        try:
            proc = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = proc.stdout

            result["nvenc"] = "nvenc" in output.lower()
            result["qsv"] = "qsv" in output.lower()
            result["amf"] = "amf" in output.lower()
            result["vaapi"] = "vaapi" in output.lower()
            result["videotoolbox"] = "videotoolbox" in output.lower()

        except Exception as e:
            logger.warning(f"Failed to detect hardware encoders: {e}")

        self._hardware_accels = result
        return result

    def get_recommended_codec(self) -> str:
        """Get recommended codec based on available hardware.

        Returns:
            Recommended codec ID
        """
        hw = self.detect_hardware_encoders()

        if hw.get("nvenc"):
            return "hevc_nvenc"
        elif hw.get("qsv"):
            # Could add QSV codecs here
            return "libx265"
        else:
            return "libx265"

    def validate_config(self, config: Optional[CodecConfig] = None) -> List[str]:
        """Validate codec configuration.

        Args:
            config: Configuration to validate (uses current if None)

        Returns:
            List of validation error messages (empty if valid)
        """
        if config is None:
            config = self._config

        errors = []

        # Check codec exists
        if config.codec not in CODEC_DEFINITIONS:
            errors.append(f"Unknown codec: {config.codec}")

        codec_info = self.get_codec_info(config.codec)
        if not codec_info:
            return errors

        # Validate quality range
        q_min, q_max = codec_info.quality_range
        if not (q_min <= config.quality <= q_max):
            errors.append(f"Quality {config.quality} out of range [{q_min}, {q_max}]")

        # Validate preset
        if config.preset and config.preset not in codec_info.presets:
            errors.append(f"Invalid preset '{config.preset}' for codec {config.codec}")

        # Validate rate control
        valid_rc = ["cq", "crf", "vbr", "cbr", "cqp"]
        if config.rate_control not in valid_rc:
            errors.append(f"Invalid rate control mode: {config.rate_control}")

        # Check hardware codec availability
        if codec_info.type == CodecType.HARDWARE:
            hw = self.detect_hardware_encoders()
            if codec_info.vendor == CodecVendor.NVIDIA and not hw.get("nvenc"):
                errors.append(f"NVIDIA encoder not available for {config.codec}")

        return errors


# Global codec manager instance
_codec_manager: Optional[CodecManager] = None


def get_codec_manager() -> CodecManager:
    """Get the global codec manager instance.

    Returns:
        CodecManager instance
    """
    global _codec_manager
    if _codec_manager is None:
        _codec_manager = CodecManager()
    return _codec_manager
