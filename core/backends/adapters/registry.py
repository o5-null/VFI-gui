"""适配器注册表与分派。

契约见 ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md`` §5：

- ``register_adapter(adapter)``  — 按 ``adapter.model_type`` 登记
- ``get_adapter(model_type)``    — 取适配器；``cain``/``sepconv`` 返回
  :class:`UnsupportedAdapter`，**其余未注册类型抛 :class:`AdapterError`**
  （不再静默回落，避免对未知模型产出错误插值）
- ``supported_models()``         — ``{model_type: supported_versions}``，不含 cain/sepconv
- ``register_builtin_adapters()``— 一次性注册全部内建适配器

本模块**纯 Python / 可选 numpy**，禁止模块级 ``import torch``。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .amt import AMTAdapter
from .atm import ATMAdapter
from .base import (
    AdapterContext,
    AdapterError,
    ModelAdapter,
    crop_to_src,
    pad_frame,
    resolve_names,
)
from .film import FILMAdapter
from .m2m import M2MAdapter
from .momo import MoMoAdapter
from .multiframe import FLAVRAdapter, MultiframeAdapter
from .rife import RIFEAdapter
from .unsupported import UnsupportedAdapter
from .xvfi import XVFIAdapter


# model_type -> adapter 实例
ADAPTER_REGISTRY: Dict[str, ModelAdapter] = {}

# 明确拒绝的模型类型（spec §2）
_UNSUPPORTED_MODEL_TYPES = ("cain", "sepconv")

# 内建注册是否已完成
_BUILTINS_REGISTERED = False


# ---------------------------------------------------------------------------
# 通用回落适配器（未登记类型使用）
# ---------------------------------------------------------------------------

class GenericNCHWAdapter(ModelAdapter):
    """通用 NCHW 适配器：单输入 ``[1, C, H, W]``（显式选用，非自动回落）。

    约定把双帧沿通道维拼接（``C = 2 * in_channels``，默认 6）。仅供调用方在
    明确知道目标模型约定时**显式实例化**；:func:`get_adapter` 对未注册类型
    不再返回它（改为抛 :class:`AdapterError`），以免静默产出错误插值。
    """

    model_type = "generic"
    min_frames = 2
    supported_versions = ["*"]

    def align_shape(self, h: int, w: int) -> tuple[int, int]:
        return int(h), int(w)

    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        name = resolve_names(ctx.input_names, ["input"])[0]
        parts = [pad_frame(f, ctx.pad_hw) for f in frames[:2]]
        packed = np.concatenate(parts, axis=0)[None].astype(np.float32)
        return {name: packed}

    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        out = np.asarray(outputs[0], dtype=np.float32)
        if out.ndim == 4:
            out = out[0]
        return crop_to_src(out[:3], ctx.src_hw)


# ---------------------------------------------------------------------------
# 注册与查询
# ---------------------------------------------------------------------------

def register_adapter(adapter: ModelAdapter) -> None:
    """按 ``adapter.model_type`` 登记适配器实例。"""
    if not adapter.model_type:
        raise AdapterError("适配器缺少 model_type，无法登记")
    ADAPTER_REGISTRY[adapter.model_type] = adapter


def register_builtin_adapters() -> None:
    """注册全部内建适配器（幂等）。"""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for adapter in (
        RIFEAdapter(),
        FILMAdapter(),
        AMTAdapter(),
        XVFIAdapter(),
        ATMAdapter(),
        MoMoAdapter(),
        M2MAdapter(),
        MultiframeAdapter(),   # stmfnet
        FLAVRAdapter(),        # flavr
    ):
        register_adapter(adapter)
    _BUILTINS_REGISTERED = True


def get_adapter(model_type: str) -> ModelAdapter:
    """取指定模型类型的适配器。

    - 已登记类型 → 对应适配器
    - ``cain`` / ``sepconv`` → :class:`UnsupportedAdapter`（build_feeds 抛错）
    - **其余未注册类型 → 抛** :class:`AdapterError`

    对未知类型不再返回 :class:`GenericNCHWAdapter`：静默回落会对未知模型做
    「双帧 concat 取前 3 通道」的错误插值，违反 spec §9「不得静默」。
    """
    register_builtin_adapters()
    key = (model_type or "").strip().lower()
    if key in _UNSUPPORTED_MODEL_TYPES:
        return UnsupportedAdapter(key)
    adapter: Optional[ModelAdapter] = ADAPTER_REGISTRY.get(key)
    if adapter is not None:
        return adapter
    registered = sorted(
        m for m in ADAPTER_REGISTRY if m not in _UNSUPPORTED_MODEL_TYPES
    )
    raise AdapterError(
        f"模型类型 '{model_type}' 未注册任何适配器（已注册：{registered}）"
    )


def supported_models() -> Dict[str, List[str]]:
    """返回 ``{model_type: supported_versions}``，不含 cain/sepconv。"""
    register_builtin_adapters()
    return {
        model_type: list(adapter.supported_versions)
        for model_type, adapter in ADAPTER_REGISTRY.items()
        if model_type not in _UNSUPPORTED_MODEL_TYPES
    }
