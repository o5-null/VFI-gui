"""MoMo 适配器（5D ``[B,3,2,H,W]``，无 timestep，固定 t=0.5，实验性）。

参照 PyTorch ``momo/__init__.py:85 MoMoModel.interpolate``：
forward 接收 ``x = stack([frame0, frame1], dim=2)``，即 ``[B,3,2,H,W]``，
执行 DDPM 多步去噪（默认 8 步）后返回单帧，不接收 timestep。

**实验性/已知限制**：
- 仅支持 t=0.5，忽略传入 ``timestep``，``ctx.extra["only_t05"]=True``。
- DDPM 多步 + 随机性：同一输入可能产生不同结果。
- 若导出为 ONNX，需把采样步数固化为常量（``num_inference_steps``）。

.. note::
   ``models/`` 中**无 MoMo 的 ONNX 资产**，输入名为推断值（契约待导出确认）。
   输入名回落 ``["input"]``。
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


class MoMoAdapter(ModelAdapter):
    """MoMo 双帧适配器（实验性，仅 t=0.5）。"""

    model_type = "momo"
    min_frames = 2
    supported_versions = ["base", "lite"]
    ALIGN = 32

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        ctx.extra["only_t05"] = True
        ctx.extra["experimental"] = True
        ctx.extra["requested_timestep"] = float(timestep)
        names = resolve_names(ctx.input_names, ["input"])
        f0 = pad_frame(frames[0], ctx.pad_hw)
        f1 = pad_frame(frames[-1], ctx.pad_hw)
        stacked = np.stack([f0, f1], axis=1)[None].astype(np.float32)  # [1,3,2,H,W]
        return {names[0]: stacked}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        image = first_image(outputs[0])
        return crop_to_src(image[:3], ctx.src_hw)
