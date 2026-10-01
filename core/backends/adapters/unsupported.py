"""不支持的模型适配器（cain / sepconv）。

这两个模型不在多后端适配支持集内（spec §2）。``get_adapter`` 对它们返回本适配器：
``align_shape`` 恒等，但 ``build_feeds`` 立即抛出 :class:`AdapterError`，使调用方
在尝试推理前就得到明确错误，而非静默产出错误结果。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import AdapterContext, AdapterError, ModelAdapter


class UnsupportedAdapter(ModelAdapter):
    """显式拒绝的模型适配器。

    可实例化为具体 ``model_type``，例如 ``UnsupportedAdapter("cain")``。
    """

    model_type = "unsupported"
    min_frames = 2
    supported_versions: List[str] = []

    def __init__(self, model_type: str | None = None) -> None:
        if model_type:
            self.model_type = model_type

    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        # 恒等：为保持接口可用，即便拒绝也不改写尺寸。
        return int(h), int(w)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        raise AdapterError(
            f"模型类型 '{self.model_type}' 不在多后端适配支持集内"
            "（仅支持 rife/film/amt/xvfi/atm/momo/m2m/stmfnet/flavr）"
        )

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        raise AdapterError(f"模型类型 '{self.model_type}' 不支持后处理")
