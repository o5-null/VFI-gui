"""Preprocessing module for VFI-gui.

This module provides scene detection and duplicate frame detection
for optimizing the interpolation pipeline.

Modules:
    scene_detect: Scene change detection
    dup_detect: Duplicate frame detection
    pipeline: Combined preprocessing pipeline
"""

from core.preprocess.dup_detect import DuplicateDetector
from core.preprocess.pipeline import PreprocessPipeline
from core.preprocess.scene_detect import (
    NeuralSceneDetector,
    PlaneStatsSceneDetector,
    SceneDetectorBase,
    SceneDetectorFactory,
    VapourSynthSceneDetector,
)

__all__ = [
    # Duplicate detection
    "DuplicateDetector",
    # Scene detection
    "SceneDetectorBase",
    "PlaneStatsSceneDetector",
    "NeuralSceneDetector",
    "VapourSynthSceneDetector",
    "SceneDetectorFactory",
    # Pipeline
    "PreprocessPipeline",
]
