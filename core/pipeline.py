"""Pipeline builder for VapourSynth processing stages."""

from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path


# Type alias for a pipeline stage
StageFunc = Callable[[Any], Any]
Stage = Tuple[str, StageFunc]


class PipelineBuilder:
    """Builder class for constructing VapourSynth processing pipelines."""

    def __init__(self, models_dir: str = "models"):
        self._models_dir = Path(models_dir)
        self._stages: List[Stage] = []
        self._clip = None

    def set_models_dir(self, path: str) -> "PipelineBuilder":
        """Set the models directory path."""
        self._models_dir = Path(path)
        return self

    def add_interpolation(
        self,
        model: str = "4.22",
        multi: int = 2,
        scale: float = 1.0,
        scene_change: bool = False,
        trt: bool = True,
    ) -> "PipelineBuilder":
        """Add RIFE interpolation stage."""
        def stage(clip):
            from vsrife import rife
            return rife(
                clip,
                model=model,
                multi=multi,
                scale=scale,
                sc=scene_change,
                trt=trt,
                trt_static_shape=True,
                trt_optimization_level=5,
            )

        self._stages.append(("interpolation", stage))
        return self

    def add_upscaling(
        self,
        engine_path: str,
        num_streams: int = 3,
        tile_size: int = 0,
        overlap: int = 0,
        device_id: int = 0,
    ) -> "PipelineBuilder":
        """Add TensorRT upscaling stage."""
        def stage(clip):
            import vapoursynth as vs
            core = vs.core

            # Convert to RGBH for TensorRT
            clip = core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")

            # Apply model
            if tile_size > 0:
                clip = core.trt.Model(
                    clip,
                    engine_path=engine_path,
                    tilesize=[tile_size, tile_size],
                    overlap=[overlap, overlap],
                    device_id=device_id,
                    num_streams=num_streams,
                )
            else:
                clip = core.trt.Model(
                    clip,
                    engine_path=engine_path,
                    device_id=device_id,
                    num_streams=num_streams,
                )

            return clip

        self._stages.append(("upscaling", stage))
        return self

    def add_scene_detection(
        self,
        model: int = 12,
        threshold: float = 0.5,
        fp16: bool = True,
        models_dir: Optional[str] = None,
    ) -> "PipelineBuilder":
        """Add scene detection stage."""
        def stage(clip):
            import sys
            import os

            # Add models path
            models_path = models_dir or str(self._models_dir)
            if models_path not in sys.path:
                sys.path.insert(0, models_path)

            from src.scene_detect import scene_detect
            return scene_detect(
                clip,
                thresh=threshold,
                model=model,
                fp16=fp16,
            )

        self._stages.append(("scene_detection", stage))
        return self

    def add_custom_filter(
        self,
        name: str,
        filter_func: Callable,
    ) -> "PipelineBuilder":
        """Add a custom filter stage."""
        self._stages.append((name, filter_func))
        return self

    def build(
        self,
        clip,
        output_format: str = "YUV420P10",
        output_matrix: str = "709",
    ):
        """Build and apply the pipeline to a clip."""
        import vapoursynth as vs
        core = vs.core

        # Apply all stages
        for stage_name, stage_func in self._stages:
            clip = stage_func(clip)

        # Convert to output format
        format_map = {
            "YUV420P8": vs.YUV420P8,
            "YUV420P10": vs.YUV420P10,
            "YUV444P10": vs.YUV444P10,
            "RGB24": vs.RGB24,
        }

        target_format = format_map.get(output_format, vs.YUV420P10)
        clip = core.resize.Bicubic(
            clip,
            format=target_format,
            matrix_s=output_matrix,
        )

        return clip

    def get_stage_names(self) -> List[str]:
        """Get list of stage names in the pipeline."""
        return [name for name, _ in self._stages]

    def clear(self) -> "PipelineBuilder":
        """Clear all stages from the pipeline."""
        self._stages.clear()
        return self

    @staticmethod
    def from_config(config: Dict[str, Any], models_dir: str = "models") -> "PipelineBuilder":
        """Create a pipeline builder from configuration dict."""
        builder = PipelineBuilder(models_dir)

        # Add interpolation if enabled
        interp = config.get("interpolation", {})
        if interp.get("enabled", False):
            builder.add_interpolation(
                model=interp.get("model", "4.22"),
                multi=interp.get("multi", 2),
                scale=interp.get("scale", 1.0),
                scene_change=interp.get("scene_change", False),
            )

        # Add upscaling if enabled
        upscale = config.get("upscaling", {})
        if upscale.get("enabled", False):
            engine_path = upscale.get("engine", "")
            if engine_path:
                builder.add_upscaling(
                    engine_path=engine_path,
                    num_streams=upscale.get("num_streams", 3),
                    tile_size=upscale.get("tile_size", 0),
                    overlap=upscale.get("overlap", 0),
                )

        # Add scene detection if enabled
        scene = config.get("scene_detection", {})
        if scene.get("enabled", False):
            builder.add_scene_detection(
                model=scene.get("model", 12),
                threshold=scene.get("threshold", 0.5),
                fp16=scene.get("fp16", True),
                models_dir=models_dir,
            )

        return builder
