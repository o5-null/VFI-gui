"""
VSGAN core modules for video processing.
Includes RIFE interpolation, scene detection, deduplication, and utilities.
"""

from .config import (
    get_models_dir,
    get_engine_cache_dir,
    get_temp_dir,
    configure,
    reset,
)
from .scene_detect import scene_detect
from .rife_trt import rife_trt, get_model_variables
from .dedup import processInfo, get_dedup_frames, ranges
from .utils import GetPlane, scale, cround, FastLineDarkenMOD
from .download import check_and_download, check_and_download_film

__all__ = [
    # Config
    "get_models_dir",
    "get_engine_cache_dir", 
    "get_temp_dir",
    "configure",
    "reset",
    # Processing
    "scene_detect",
    "rife_trt",
    "get_model_variables",
    # Dedup
    "processInfo",
    "get_dedup_frames",
    "ranges",
    # Utils
    "GetPlane",
    "scale",
    "cround",
    "FastLineDarkenMOD",
    # Download
    "check_and_download",
    "check_and_download_film",
]
