"""M2M 适配器（双帧 + fltTime，支持任意 timestep）。

参照 PyTorch ``m2m/__init__.py:77 M2MVFIModel.interpolate``：
``self._model(frame0, frame1, fltTimes=[fltTime])``，其中
``fltTime = make_timestep_tensor(B, t, ..., ndim=4)`` 即 ``[B,1,1,1]``；
输出取 ``outputs[0]``。模型内部自行 padding 到 16 的倍数。

.. note::
   ``models/`` 中**无 M2M 的 ONNX 资产**，输入名为推断值（契约待导出确认）。
   输入名回落 ``["frame0", "frame1", "fltTime"]``。
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


class M2MAdapter(ModelAdapter):
    """M2M 双帧适配器（任意 timestep）。"""

    model_type = "m2m"
    min_frames = 2
    supported_versions = ["default"]
    ALIGN = 16

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        names = resolve_names(ctx.input_names, ["frame0", "frame1", "fltTime"])
        f0 = pad_frame(frames[0], ctx.pad_hw)[None].astype(np.float32)
        f1 = pad_frame(frames[-1], ctx.pad_hw)[None].astype(np.float32)
        # make_timestep_tensor(..., ndim=4) -> [B, 1, 1, 1]
        flt_time = np.asarray([[[[float(timestep)]]]], dtype=np.float32)
        return {names[0]: f0, names[1]: f1, names[2]: flt_time}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        image = first_image(outputs[0])
        return crop_to_src(image[:3], ctx.src_hw)
