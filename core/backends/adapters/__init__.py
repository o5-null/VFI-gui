"""多后端模型适配层。

按模型类型把统一的 ``[3,H,W]`` 帧输入转换为各模型 ONNX/TensorRT 所需的
feed 张量，并把原始输出规整回 ``[3,src_h,src_w]``。

对外主要接口（契约见 ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md`` §5）::

    from core.backends.adapters import (
        register_builtin_adapters, get_adapter, supported_models,
        ModelAdapter, AdapterContext, AdapterError,
    )

本包**纯 numpy / 纯 Python**：任何文件都禁止模块级 ``import torch``。
"""

from __future__ import annotations

from .amt import AMTAdapter
from .atm import ATMAdapter
from .base import AdapterContext, AdapterError, ModelAdapter
from .film import FILMAdapter
from .m2m import M2MAdapter
from .momo import MoMoAdapter
from .multiframe import FLAVRAdapter, MultiframeAdapter
from .registry import (
    ADAPTER_REGISTRY,
    GenericNCHWAdapter,
    get_adapter,
    register_adapter,
    register_builtin_adapters,
    supported_models,
)
from .rife import RIFEAdapter
from .unsupported import UnsupportedAdapter
from .xvfi import XVFIAdapter

__all__ = [
    # 基类与上下文
    "AdapterContext",
    "AdapterError",
    "ModelAdapter",
    # 注册表 API
    "ADAPTER_REGISTRY",
    "register_adapter",
    "register_builtin_adapters",
    "get_adapter",
    "supported_models",
    # 具体适配器
    "RIFEAdapter",
    "FILMAdapter",
    "AMTAdapter",
    "XVFIAdapter",
    "ATMAdapter",
    "MoMoAdapter",
    "M2MAdapter",
    "MultiframeAdapter",
    "FLAVRAdapter",
    "GenericNCHWAdapter",
    "UnsupportedAdapter",
]
