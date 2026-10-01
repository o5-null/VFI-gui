"""AMT 适配器（双帧 + 时间嵌入 embt）。

参照 PyTorch ``amt/__init__.py:163 AMT_S.forward(img0, img1, embt, scale_factor)``：
输入为两帧 ``[B,3,H,W]`` 与 embt ``[B,1]``，输出字典取 ``imgt_pred``。

.. note::
   ``models/`` 目录中**没有 AMT 的 ONNX 资产**，输入名与 embt 具体形状为推断值
   （契约待导出确认）。输入名优先采用后端从 session 读到的真实名字，缺失时回落
   ``["img0", "img1", "embt"]``。``scale`` 为推理期参数，导出图通常已固化，
   本适配器记录于 ``ctx.extra`` 不产生额外 feed。
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


class AMTAdapter(ModelAdapter):
    """AMT 双帧适配器。"""

    model_type = "amt"
    min_frames = 2
    supported_versions = ["s", "l", "g", "gopro"]
    ALIGN = 16

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        ctx.extra.setdefault("scale", ctx.extra.get("scale", 1.0))
        names = resolve_names(ctx.input_names, ["img0", "img1", "embt"])
        f0 = pad_frame(frames[0], ctx.pad_hw)[None].astype(np.float32)
        f1 = pad_frame(frames[-1], ctx.pad_hw)[None].astype(np.float32)
        # PyTorch: make_timestep_tensor(B, t, ...) -> [B, 1]
        embt = np.asarray([[float(timestep)]], dtype=np.float32)  # [1, 1]
        return {names[0]: f0, names[1]: f1, names[2]: embt}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        image = first_image(outputs[0])
        return crop_to_src(image[:3], ctx.src_hw)
