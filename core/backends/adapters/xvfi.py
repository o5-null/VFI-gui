"""XVFI 适配器（5D ``[B,C,T,H,W]`` + t_value）。

参照 PyTorch ``xvfi/__init__.py:347 XVFIModel.interpolate``：
输入 ``x`` 形状 ``[B,C,T,H,W]``（T=2），``t_value`` 形状 ``[B,1]``；输出 ``[B,3,H,W]``。
padding 需对齐到 ``2**S_tst * scale * 4``（默认 128）。

.. note::
   ``models/`` 中**无 XVFI 的 ONNX 资产**，输入名与 t_value 形状为推断值
   （契约待导出确认）。输入名回落 ``["input", "t_value"]``。
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


class XVFIAdapter(ModelAdapter):
    """XVFI 双帧（时序堆叠）适配器。"""

    model_type = "xvfi"
    min_frames = 2
    supported_versions = ["x4k1000fps", "vimeo"]
    ALIGN = 128

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        names = resolve_names(ctx.input_names, ["input", "t_value"])
        f0 = pad_frame(frames[0], ctx.pad_hw)
        f1 = pad_frame(frames[-1], ctx.pad_hw)
        # [C,T,H,W] -> [1,C,T,H,W]
        stacked = np.stack([f0, f1], axis=1)[None].astype(np.float32)
        t_value = np.asarray([[float(timestep)]], dtype=np.float32)  # [1, 1]
        return {names[0]: stacked, names[1]: t_value}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        image = first_image(outputs[0])
        return crop_to_src(image[:3], ctx.src_hw)
