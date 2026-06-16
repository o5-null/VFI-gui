"""Application-level constants for VFI-gui.

This module defines constants that are used across multiple modules:
- Model definitions (scene detection, interpolation)
- Processing parameters
- UI-related constants
"""

from typing import Dict


# ====================
# Interpolation Constants
# ====================

# Available interpolation multipliers
INTERP_MULTIPLIERS: list[int] = [2, 4, 8]


# ====================
# Scene Detection Models
# ====================

# Scene detection model definitions
# Maps model ID to display name
SCENE_DETECT_MODELS: Dict[int, str] = {
    0: "EfficientFormerV2-S0 (224px)",
    1: "EfficientFormerV2-S0 + RIFE46 Flow",
    2: "EfficientNetV2-B0 (256px)",
    3: "EfficientNetV2-B0 + RIFE46 Flow",
    4: "SwinV2-Small (256px)",
    5: "SwinV2-Small + RIFE46 Flow",
    6: "EfficientNetV2-B0 (48x27)",
    7: "DaViT-Small (256px) - 30k",
    8: "DaViT-Small (256px) - 40k",
    9: "MaxViTV2-Nano (256px) - 20k",
    10: "MaxViTV2-Nano (256px) - 30k",
    11: "MaxViTV2-Base (224px)",
    12: "MobileViTV2 + RIFE422 + Sobel (Recommended)",
    13: "AutoShot (5 images)",
    14: "Shift-LPIPS-Alex",
    15: "Shift-LPIPS-VGG",
    16: "DISTS",
}

# Default scene detection model
DEFAULT_SCENE_DETECT_MODEL: int = 12


# ====================
# Upscaling Constants
# ====================

# Tile size range for upscaling
UPSCALE_TILE_SIZE_MIN: int = 0
UPSCALE_TILE_SIZE_MAX: int = 4096

# Overlap range
UPSCALE_OVERLAP_MIN: int = 0
UPSCALE_OVERLAP_MAX: int = 512

# GPU streams range
UPSCALE_GPU_STREAMS_MIN: int = 1
UPSCALE_GPU_STREAMS_MAX: int = 16
DEFAULT_GPU_STREAMS: int = 3


# ====================
# Scene Detection Threshold
# ====================

SCENE_THRESHOLD_MIN: float = 0.0
SCENE_THRESHOLD_MAX: float = 1.0
SCENE_THRESHOLD_DEFAULT: float = 0.5
SCENE_THRESHOLD_STEP: float = 0.05


# ====================
# Interpolation Scale
# ====================

INTERP_SCALE_MIN: float = 0.1
INTERP_SCALE_MAX: float = 2.0
INTERP_SCALE_DEFAULT: float = 1.0
INTERP_SCALE_STEP: float = 0.1