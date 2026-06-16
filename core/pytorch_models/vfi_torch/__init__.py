"""
VFI (Video Frame Interpolation) PyTorch 模型模块。
提供多种插值模型的统一接口。
"""

from typing import Optional

from .base import (
    PyTorchVFIModel,
    VFIConfig,
    VFIResult,
    ModelType,
    BackendType,
)
from .utils import (
    preprocess_frames,
    postprocess_frames,
    preprocess_frames_tensor,
    postprocess_frames_tensor,
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
from .gmfss import GMFSSModel
from .xvfi import XVFIModel
from .eisai import EISAIModel

# 模型注册表
MODEL_REGISTRY = {
    ModelType.RIFE: RIFEModel,
    ModelType.FILM: FILMModel,
    ModelType.IFRNET: IFRNetModel,
    ModelType.AMT: AMTModel,
    ModelType.GMFSS: GMFSSModel,
    ModelType.XVFI: XVFIModel,
    ModelType.EISAI: EISAIModel,
}


def get_model(model_type: ModelType, config: Optional[VFIConfig] = None) -> PyTorchVFIModel:
    """
    根据类型获取模型实例。
    委托给 base.get_model 以确保单一注册来源。
    
    Args:
        model_type: 插值模型类型
        config: 可选的模型配置
        
    Returns:
        模型实例
    """
    from .base import get_model as _base_get_model
    if config is None:
        config = VFIConfig(model_type=model_type)
    return _base_get_model(config)


__all__ = [
    # 基础类
    "PyTorchVFIModel",
    "VFIConfig",
    "VFIResult",
    "ModelType",
    "BackendType",
    # 工具函数
    "preprocess_frames",
    "postprocess_frames",
    "preprocess_frames_tensor",
    "postprocess_frames_tensor",
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
    "GMFSSModel",
    # 工厂函数
    "get_model",
    "MODEL_REGISTRY",
]
