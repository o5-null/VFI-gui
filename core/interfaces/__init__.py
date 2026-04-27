"""Core interfaces for VFI-gui.

This module defines abstract interfaces for core components,
enabling better decoupling and testability.
"""

from .imodel_manager import IModelManager, ModelInfo, ModelStatus
from .icodec_manager import ICodecManager
from .iconfig import IConfig

__all__ = [
    "IModelManager",
    "ModelInfo",
    "ModelStatus",
    "ICodecManager", 
    "IConfig",
]
