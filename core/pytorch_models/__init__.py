"""PyTorch 推理后端管理模块。

提供统一的 PyTorch 推理管理，包括：
- 设备管理 (CPU/CUDA/MPS/XPU)
- 模型加载和缓存
- 帧处理管道
- 与 GUI 的集成接口

架构：
- torch_backend/        # 管理层
  - base.py             # 基础抽象类
  - model_manager.py    # 模型管理器
  - frame_processor.py  # 帧处理器
  - vfi_torch/          # 模型实现
    - rife/
    - film/
    - amt/
    - ifrnet/
"""

from .base import (
    VFIModelBase,
    VFIModelInfo,
    InterpolationResult,
    InterpolationConfig,
    InterpolationStateList,
    DType,
    DTYPE_MAP,
    PerfStats,
    InferencePerfEntry,
    get_torch_device,
    clear_cuda_cache,
    assert_batch_size,
)
from .model_manager import ModelManager, get_model_manager
from .frame_processor import FrameProcessor

# 从 vfi_torch 导入模型和工具
from .vfi_torch import (
    # 基础类
    VFIConfig,
    VFIResult,
    ModelType,
    BackendType,
    PyTorchVFIModel,
    # 工具函数
    preprocess_frames,
    postprocess_frames,
    preprocess_frames_tensor,
    postprocess_frames_tensor,
    load_model_weights,
    get_device,
    clear_cache,
    download_model,
    InputPadder,
    # 模型
    RIFEModel,
    FILMModel,
    IFRNetModel,
    AMTModel,
    # 注册表
    MODEL_REGISTRY,
    get_model,
)

# 设备函数别名 (get_torch_device 是 get_device 的便捷别名)
# get_torch_device() 已在 base.py 中定义为调用 get_device()

__all__ = [
    # 基础类 (torch_backend)
    "VFIModelBase",
    "VFIModelInfo",
    "InterpolationResult",
    "InterpolationConfig",
    "InterpolationStateList",
    "DType",
    "DTYPE_MAP",
    "PerfStats",
    "InferencePerfEntry",
    # 基础类 (vfi_torch)
    "VFIConfig",
    "VFIResult",
    "ModelType",
    "BackendType",
    "PyTorchVFIModel",
    # 设备管理
    "get_torch_device",
    "get_device",
    "clear_cuda_cache",
    "clear_cache",
    # 模型管理
    "ModelManager",
    "get_model_manager",
    "FrameProcessor",
    # 工具函数
    "preprocess_frames",
    "postprocess_frames",
    "preprocess_frames_tensor",
    "postprocess_frames_tensor",
    "load_model_weights",
    "download_model",
    "InputPadder",
    "assert_batch_size",
    # 模型
    "RIFEModel",
    "FILMModel",
    "IFRNetModel",
    "AMTModel",
    # 注册表
    "MODEL_REGISTRY",
    "get_model",
]
