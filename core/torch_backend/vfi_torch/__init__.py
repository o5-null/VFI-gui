"""
VFI (Video Frame Interpolation) PyTorch 模型模块。
提供多种插值模型的统一接口。
"""

from typing import Optional

from .base import (
    BaseVFIModel,
    VFIConfig,
    VFIResult,
    ModelType,
    BackendType,
)
from .utils import (
    preprocess_frames,
    postprocess_frames,
    load_model_weights,
    get_device,
    clear_cache,
    download_model,
    InputPadder,
)

# 模型实现
from .rife import RIFEModel
from .film import FILMModel
from .ifrnet import IFRNetModel
from .amt import AMTModel

# 模型注册表
MODEL_REGISTRY = {
    ModelType.RIFE: RIFEModel,
    ModelType.FILM: FILMModel,
    ModelType.IFRNET: IFRNetModel,
    ModelType.AMT: AMTModel,
}


def get_model(model_type: ModelType, config: Optional[VFIConfig] = None) -> BaseVFIModel:
    """
    根据类型获取模型实例。
    
    Args:
        model_type: 插值模型类型
        config: 可选的模型配置
        
    Returns:
        模型实例
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"未知模型类型: {model_type}。可用类型: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config)


__all__ = [
    # 基础类
    "BaseVFIModel",
    "VFIConfig",
    "VFIResult",
    "ModelType",
    "BackendType",
    # 工具函数
    "preprocess_frames",
    "postprocess_frames",
    "load_model_weights",
    "get_device",
    "clear_cache",
    "download_model",
    "InputPadder",
    # 模型
    "RIFEModel",
    "FILMModel",
    "IFRNetModel",
    "AMTModel",
    # 工厂函数
    "get_model",
    "MODEL_REGISTRY",
]
