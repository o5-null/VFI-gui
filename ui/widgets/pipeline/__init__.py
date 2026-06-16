"""Pipeline configuration group widgets.

This module exports all pipeline configuration group widgets:
    - InterpolationGroup: Frame interpolation settings
    - UpscalingGroup: Video upscaling settings
    - SceneDetectGroup: Scene detection settings
    - OutputGroup: Output codec and encoding settings
"""

from ui.widgets.pipeline.interpolation_group import InterpolationGroup
from ui.widgets.pipeline.upscaling_group import UpscalingGroup
from ui.widgets.pipeline.scene_detect_group import SceneDetectGroup
from ui.widgets.pipeline.output_group import OutputGroup

__all__ = [
    "InterpolationGroup",
    "UpscalingGroup",
    "SceneDetectGroup",
    "OutputGroup",
]