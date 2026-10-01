"""RIFE 适配器（vs-mlrt ONNX，单输入 ``[1,C,H,W]``，C=7/11）。

真实资产契约（``models/rife_v2/rife_v4.22.onnx`` 实测）::

    input  : "input"  [N, 7, H, W]  (动态 H/W)
    output : "output" [N, 3, H, W]

``build_feeds`` 直接复用 :func:`core.backends.rife_input.pack_rife_input`。
``scale`` / ``fastmode`` / ``ensemble`` 已固化在导出的 ONNX 图中，仅记录于
``ctx.extra`` 供后端日志使用，不产生额外 feed。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ...models.asset_resolver import MODEL_ASSET_TABLE
from ..rife_input import pack_rife_input
from .base import (
    AdapterContext,
    ModelAdapter,
    crop_to_src,
    first_image,
    pad_frame,
    resolve_names,
)


def _derive_supported_versions() -> List[str]:
    """从 ``asset_resolver.MODEL_ASSET_TABLE["rife"]`` 派生基础版本令牌。

    排除 ``*_lite`` / ``*_heavy`` / ``*_ensemble`` 等变体令牌（含 ``_``），
    仅保留基础版本，并无按数值升序排序，避免与资产表手工同步产生漂移。
    """
    tokens = [v for v in MODEL_ASSET_TABLE.get("rife", {}) if "_" not in v]

    def _sort_key(token: str):
        # RIFE 版本是「点分整数」标签（4.9 < 4.10 < 4.22），不能按 float 排序
        # （float("4.10") == 4.1 会排在 4.9 之前）。
        try:
            return (0, tuple(int(part) for part in token.split(".")))
        except ValueError:
            return (1, token)

    return sorted(tokens, key=_sort_key)


class RIFEAdapter(ModelAdapter):
    """RIFE 双帧适配器。"""

    model_type = "rife"
    min_frames = 2
    # 版本集合从资产表派生，保持 List[str] 返回类型。
    supported_versions = _derive_supported_versions()

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        # RIFE ONNX 图内部自带 padding，输入分辨率无需对齐。
        return int(h), int(w)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        # scale/fastmode/ensemble 仅记录：ONNX 图已固化这些运行选项。
        ctx.extra.setdefault("scale", ctx.extra.get("scale", 1.0))
        ctx.extra.setdefault("fastmode", ctx.extra.get("fastmode", True))
        ctx.extra.setdefault("ensemble", ctx.extra.get("ensemble", False))

        in_channels = ctx.in_channels or ctx.extra.get("in_channels") or 7
        f0 = pad_frame(frames[0], ctx.pad_hw)
        f1 = pad_frame(frames[-1], ctx.pad_hw)
        packed = pack_rife_input(f0, f1, float(timestep), int(in_channels))
        name = resolve_names(ctx.input_names, ["input"])[0]
        return {name: packed}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        image = first_image(outputs[0])
        return crop_to_src(image[:3], ctx.src_hw)
