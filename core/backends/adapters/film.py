"""FILM 适配器（双帧 + 独立 timestep feed）。

真实资产契约（``models/film/film_net_fp32.onnx`` 实测）::

    img0     : [1, 3, H, W]
    img1     : [1, 3, H, W]
    timestep : [1]              (注意是一维，非 [B,1])
    output   : "output"

与 PyTorch ``film/__init__.py:238 make_timestep_tensor``（生成 ``[B,1]``）不同，
导出图把 timestep 压成一维；本适配器按**实际资产**填 ``[1]``。
对齐到 64 倍数，postprocess 完成 unpad。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import (
    AdapterContext,
    ModelAdapter,
    align_up,
    crop_to_src,
    first_image,
    pad_frame,
    resolve_names,
)


class FILMAdapter(ModelAdapter):
    """FILM 双帧适配器。"""

    model_type = "film"
    min_frames = 2
    supported_versions = ["fp32"]
    ALIGN = 64

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        names = resolve_names(ctx.input_names, ["img0", "img1", "timestep"])
        f0 = pad_frame(frames[0], ctx.pad_hw)[None].astype(np.float32)
        f1 = pad_frame(frames[-1], ctx.pad_hw)[None].astype(np.float32)
        t = np.asarray([float(timestep)], dtype=np.float32)  # 实测 shape [1]
        return {names[0]: f0, names[1]: f1, names[2]: t}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        image = first_image(outputs[0])
        return crop_to_src(image[:3], ctx.src_hw)
