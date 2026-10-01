"""ATM-VFI 适配器（双帧，无 timestep，固定 t=0.5）。

参照 PyTorch ``atm/__init__.py:88 ATMVFIModel.interpolate``：forward 只接收两帧，
不接收 timestep，永远输出居中帧；padding 对齐到 64。

**已知限制**：仅支持 t=0.5。适配器忽略传入 ``timestep`` 并在 ``ctx.extra`` 打上
``only_t05=True``，调用方需据此递归二分。

.. note::
   ``models/`` 中**无 ATM 的 ONNX 资产**，输入名为推断值（契约待导出确认）。
   输入名回落 ``["img0", "img1"]``。
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


class ATMAdapter(ModelAdapter):
    """ATM 双帧适配器（仅 t=0.5）。"""

    model_type = "atm"
    min_frames = 2
    supported_versions = ["base", "lite", "base_pct"]
    ALIGN = 64

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        # 显式记录：模型无 timestep 输入，永远 t=0.5。
        ctx.extra["only_t05"] = True
        ctx.extra["requested_timestep"] = float(timestep)
        names = resolve_names(ctx.input_names, ["img0", "img1"])
        f0 = pad_frame(frames[0], ctx.pad_hw)[None].astype(np.float32)
        f1 = pad_frame(frames[-1], ctx.pad_hw)[None].astype(np.float32)
        return {names[0]: f0, names[1]: f1}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        image = first_image(outputs[0])
        return crop_to_src(image[:3], ctx.src_hw)
