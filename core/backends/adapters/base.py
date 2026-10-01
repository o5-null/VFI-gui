"""适配层基础契约：``AdapterContext`` / ``AdapterError`` / ``ModelAdapter``。

契约见 ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md`` §5（frozen interface）。

约定
----
- 帧数据以 numpy 数组 ``[3, H, W]`` float32 RGB（``[0, 1]``）在适配器间流转。
- ``build_feeds`` 返回 ``{input_name: ndarray}``，所有数组的 H/W 均为 ``ctx.pad_hw``。
- ``postprocess`` 返回 ``[3, src_h, src_w]`` float32（内部完成 unpad）。

本模块**纯 numpy**，禁止模块级 ``import torch``（``core/backends/AGENTS.md`` 红线）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AdapterContext:
    """单次推理的上下文，由后端填充、适配器消费。"""

    input_names: List[str]
    output_names: List[str]
    src_hw: Tuple[int, int]          # 原始 (H, W)
    pad_hw: Tuple[int, int]          # 对齐后 (H, W)
    in_channels: Optional[int] = None
    device: str = "cuda:0"
    extra: Dict[str, Any] = field(default_factory=dict)


class AdapterError(Exception):
    """适配器无法为该模型构造合法输入/输出时抛出。"""


# ---------------------------------------------------------------------------
# 共享 numpy 工具
# ---------------------------------------------------------------------------

def align_up(value: int, divisor: int) -> int:
    """向上对齐到 ``divisor`` 的整数倍。"""
    if divisor <= 1:
        return int(value)
    return ((int(value) + divisor - 1) // divisor) * divisor


def pad_frame(frame: np.ndarray, pad_hw: Tuple[int, int]) -> np.ndarray:
    """将 ``[C, h, w]`` 帧右下零填充到 ``pad_hw``。

    若已足够大则裁剪到 ``pad_hw``（防止上游误传超大帧）。
    """
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 3:
        raise AdapterError(f"pad_frame 期望 [C,h,w]，收到 {arr.shape}")
    target_h, target_w = int(pad_hw[0]), int(pad_hw[1])
    c, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    if h < target_h or w < target_w:
        out = np.zeros((c, target_h, target_w), dtype=np.float32)
        out[:, : min(h, target_h), : min(w, target_w)] = arr[
            :, : target_h, : target_w
        ]
        return out
    return arr[:, :target_h, :target_w]


def first_image(array: np.ndarray) -> np.ndarray:
    """从 ``[N,C,H,W]`` 或 ``[C,H,W]`` 中取出单张 ``[C,H,W]``。"""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 4:
        return arr[0]
    if arr.ndim == 3:
        return arr
    raise AdapterError(f"first_image 期望 3/4 维数组，收到 {arr.shape}")


def crop_to_src(array: np.ndarray, src_hw: Tuple[int, int]) -> np.ndarray:
    """把 ``[C,H,W]`` 裁剪回原始 ``(H, W)`` 并返回 float32 连续数组。"""
    arr = np.asarray(array, dtype=np.float32)
    sh, sw = int(src_hw[0]), int(src_hw[1])
    return np.ascontiguousarray(arr[:, :sh, :sw])


def resolve_names(
    provided: List[str], defaults: List[str]
) -> List[str]:
    """按位置返回输入名：优先 ``provided``，数量不足处用 ``defaults`` 补齐。

    这样既尊重后端从 session/graph 读到的真实名字，也能在名字缺失时回落。
    """
    names = list(provided or [])
    out: List[str] = []
    for i, default in enumerate(defaults):
        out.append(names[i] if i < len(names) and names[i] else default)
    return out


# ---------------------------------------------------------------------------
# 适配器基类
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """按 ``model_type`` 分派的模型适配器。"""

    model_type: str = ""
    min_frames: int = 2                       # 2=双帧; 4=STMFNet/FLAVR
    supported_versions: List[str] = ["*"]

    @abstractmethod
    def align_shape(self, h: int, w: int) -> Tuple[int, int]:
        """返回模型要求对齐后的 ``(H, W)``。"""

    def expand_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """默认恒等；多帧模型在此把 2 帧展开为 N 帧。"""
        return list(frames)

    @abstractmethod
    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        """返回 ``{input_name: ndarray}``，shape 均为对齐后 ``[.., padH, padW]``。"""

    def output_names(self, ctx: AdapterContext) -> List[str]:
        """默认透传后端从 session/graph 读到的输出名。"""
        return list(ctx.output_names)

    @abstractmethod
    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        """把原始输出转成 ``[3, src_h, src_w]`` float32（含 unpad）。"""
