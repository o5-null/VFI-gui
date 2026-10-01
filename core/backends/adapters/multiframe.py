"""多帧（4 帧输入）适配器：STMFNet 与 FLAVR。

包含两个适配器：

``MultiframeAdapter`` (stmfnet)
    参照 PyTorch ``stmfnet/__init__.py:66``：需要 4 帧（``MIN_INPUT_FRAMES=4``），
    单输出、无 timestep、模型内部 padding。此处以 4 个独立输入喂入
    （默认名 ``["I0","I1","I2","I3"]``，每个 ``[1,3,PH,PW]``）。
    契约待导出确认。

    仅给 2 帧时按 PyTorch 行为做 **2→4 复制**（``stmfnet/__init__.py:87``
    ``self._model(frame0, frame0, frame1, frame1)``）：``[a,b] → [a,a,b,b]``，
    即 ``I0=I1=a``、``I2=I3=b``。不足 4 帧不再报错。

``FLAVRAdapter`` (flavr)
    真实资产契约（``models/flavr/FLAVR_{2x,4x,8x}.onnx`` 实测）::

        frames : [1, 12, H, W]   # 4 帧 × 3 通道按通道拼接
        output : [1, 3*n, H, W]  # n = 2x→1, 4x→3, 8x→7

    ``expand_frames`` 把 2 帧 ``[a,b]`` 展开为 ``[a,a,b,b]``。输出单张量按通道
    堆叠多帧，postprocess reshape 为 ``[n,3,H,W]`` 后按 timestep 选帧。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import (
    AdapterContext,
    AdapterError,
    ModelAdapter,
    align_up,
    crop_to_src,
    pad_frame,
    resolve_names,
)


class MultiframeAdapter(ModelAdapter):
    """STMFNet 4 帧适配器（4 独立输入）。"""

    model_type = "stmfnet"
    min_frames = 4
    supported_versions = ["v1"]
    ALIGN = 16

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def expand_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """把输入帧规整为 4 帧 ``[I0,I1,I2,I3]``。

        - ``>=4`` 帧：取前 4 帧。
        - ``2`` 帧 ``[a,b]``：复制为 ``[a,a,b,b]``（与 PyTorch
          ``stmfnet/__init__.py:87`` 的 ``(frame0, frame0, frame1, frame1)`` 一致）。
        - 其他数量：无法构造合法 4 帧输入 → 抛 :class:`AdapterError`。
        """
        if len(frames) >= self.min_frames:
            return list(frames[: self.min_frames])
        if len(frames) == 2:
            a, b = frames
            return [a, a, b, b]
        raise AdapterError(
            f"stmfnet 需要 2 或 {self.min_frames} 帧，收到 {len(frames)}"
        )

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        # 4 帧时 timestep 恒为 0.5（居中帧）。
        ctx.extra["only_t05"] = True
        # 输入名按位置绑定 I0..I3；expand_frames 已保证 frames == [I0,I1,I2,I3]。
        names = resolve_names(ctx.input_names, ["I0", "I1", "I2", "I3"])
        feeds: Dict[str, np.ndarray] = {}
        for i in range(self.min_frames):
            feeds[names[i]] = pad_frame(frames[i], ctx.pad_hw)[None].astype(np.float32)
        return feeds

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        out = np.asarray(outputs[0], dtype=np.float32)
        if out.ndim == 4:
            out = out[0]
        return crop_to_src(out[:3], ctx.src_hw)


class FLAVRAdapter(ModelAdapter):
    """FLAVR 4 帧适配器（单输入通道拼接，多帧输出）。"""

    model_type = "flavr"
    min_frames = 4
    supported_versions = ["2x", "4x", "8x"]
    ALIGN = 16
    # version token -> 输出帧数
    N_OUTPUTS = {"2x": 1, "4x": 3, "8x": 7}

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        return align_up(h, self.ALIGN), align_up(w, self.ALIGN)

    def expand_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """[a, b] -> [a, a, b, b]（保持首尾帧不变，利于时序连续）。"""
        if len(frames) >= self.min_frames:
            return list(frames[: self.min_frames])
        if len(frames) == 2:
            a, b = frames
            return [a, a, b, b]
        raise AdapterError(f"flavr 需要 2 或 {self.min_frames} 帧，收到 {len(frames)}")

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        # postprocess 无 timestep 参数，借此透传以决定选帧下标。
        ctx.extra["timestep"] = float(timestep)
        name = resolve_names(ctx.input_names, ["frames"])[0]
        parts = [pad_frame(frames[i], ctx.pad_hw) for i in range(self.min_frames)]
        packed = np.concatenate(parts, axis=0)[None].astype(np.float32)  # [1,12,H,W]
        return {name: packed}

    def _resolve_index(self, n_outputs: int, ctx: AdapterContext) -> int:
        if n_outputs <= 1:
            return 0
        t = float(ctx.extra.get("timestep", 0.5))
        # 参照 flavr/__init__.py:114-118
        idx = int(round(t * (n_outputs + 1))) - 1
        return max(0, min(n_outputs - 1, idx))

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        out = np.asarray(outputs[0], dtype=np.float32)
        if out.ndim == 4:
            out = out[0]  # [C,H,W]
        channels = out.shape[0]
        if channels % 3 != 0:
            raise AdapterError(f"flavr 输出通道 {channels} 非 3 的倍数")
        n_outputs = channels // 3
        idx = self._resolve_index(n_outputs, ctx)
        frame = out[idx * 3 : idx * 3 + 3]  # [3,H,W]
        return crop_to_src(frame, ctx.src_hw)
