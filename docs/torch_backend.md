# Torch Backend 设计文档

## 概述

`torch_backend` 是 VFI-gui 的 PyTorch 推理后端，提供统一的模型管理、帧处理和 GUI 集成接口。

## 架构

```
core/torch_backend/
├── __init__.py             # 模块入口，导出所有接口
├── base.py                 # 基础抽象类和工具函数
├── model_manager.py        # 模型加载、缓存、下载管理
├── frame_processor.py      # 帧处理循环和内存管理
└── vfi_torch/              # 模型实现
    ├── __init__.py         # 模型注册表
    ├── base.py             # BaseVFIModel, VFIConfig, VFIResult
    ├── utils.py            # 工具函数
    ├── rife/               # RIFE 模型 (4.0-4.26)
    ├── film/               # FILM 模型
    ├── amt/                # AMT 模型 (S/L/G)
    └── ifrnet/             # IFRNet 模型
```

## 核心组件

### 1. VFIConfig

模型配置类，包含所有推理参数：

```python
from core.torch_backend import VFIConfig, ModelType

config = VFIConfig(
    model_type=ModelType.RIFE,
    model_version="4.22",
    multiplier=2,           # 插值倍数
    scale=1.0,              # 分辨率缩放
    fp16=True,              # 半精度推理
    fast_mode=False,        # RIFE 快速模式
    ensemble=False,         # RIFE 集成模式
)
```

### 2. BaseVFIModel

所有模型的抽象基类：

```python
class BaseVFIModel(ABC):
    def load_model(self, checkpoint_path: str) -> None:
        """加载模型权重"""
        pass
    
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs
    ) -> VFIResult:
        """插值单帧"""
        pass
    
    def interpolate_batch(
        self,
        frames: torch.Tensor,
        multiplier: int = 2,
        callback: Callable[[int, int], None] = None,
    ) -> torch.Tensor:
        """批量插值"""
        pass
```

### 3. 模型注册表

```python
from core.torch_backend import MODEL_REGISTRY, get_model, ModelType

# 获取模型
model = get_model(ModelType.RIFE, config)

# 或直接使用类
from core.torch_backend import RIFEModel
model = RIFEModel(config)
model.load_model("models/rife/rife49.pth")
```

### 4. 工具函数

```python
from core.torch_backend import (
    get_device,           # 获取设备 (cuda/cpu/mps)
    clear_cache,          # 清理 CUDA 缓存
    preprocess_frames,    # NHWC -> NCHW
    postprocess_frames,   # NCHW -> NHWC
    download_model,       # 下载模型
    InputPadder,          # 输入填充
)
```

## 支持的模型

| 模型 | 类型 | 版本 | 描述 |
|------|------|------|------|
| RIFE | ModelType.RIFE | 4.0, 4.6, 4.7, 4.17, 4.22, 4.26 | 实时中间流估计 |
| FILM | ModelType.FILM | fp32 | 大运动帧插值 |
| IFRNet | ModelType.IFRNET | S_Vimeo90K, L_Vimeo90K | 中间特征细化网络 |
| AMT | ModelType.AMT | s, l, g | 全对多场变换 |

## GUI 集成

### TorchProcessor

与 `VideoProcessor` 接口兼容的 PyTorch 处理器：

```python
from core.torch_processor import TorchProcessor

processor = TorchProcessor(
    video_path="input.mp4",
    config={
        "interpolation": {
            "model_type": "rife",
            "model_version": "4.22",
            "multi": 2,
            "scale": 1.0,
        },
        "output": {
            "codec": "libx265",
            "quality": 22,
            "preset": "medium",
        },
    },
    torch_config={
        "models_dir": "models",
        "output_dir": "output",
        "fp16": True,
        "clear_cache_every": 10,
    },
)

# 连接信号
processor.progress_updated.connect(on_progress)
processor.stage_changed.connect(on_stage)
processor.log_message.connect(on_log)
processor.finished.connect(on_finished)
processor.error_occurred.connect(on_error)

# 控制
processor.start()      # 开始
processor.pause()      # 暂停
processor.resume()     # 恢复
processor.cancel()     # 取消
```

### 信号

| 信号 | 参数 | 描述 |
|------|------|------|
| progress_updated | (current, total, fps) | 进度更新 |
| stage_changed | (stage_name) | 阶段变化 |
| log_message | (message) | 日志消息 |
| finished | (output_path) | 处理完成 |
| error_occurred | (error_message) | 发生错误 |

## 模型文件结构

```
models/
├── rife/
│   ├── rife47.pth
│   ├── rife49.pth
│   ├── rife417.pth
│   └── rife426.pth
├── film/
│   └── film_net_fp32.pt
├── amt/
│   ├── amt-s.pth
│   ├── amt-l.pth
│   └── amt-g.pth
└── ifrnet/
    ├── IFRNet_S_Vimeo90K.pth
    └── IFRNet_L_Vimeo90K.pth
```

## 添加新模型

1. 在 `vfi_torch/` 下创建模型目录：

```python
# vfi_torch/newmodel/__init__.py
from ..base import BaseVFIModel, VFIConfig, VFIResult, ModelType

class NewModelModel(BaseVFIModel):
    MODEL_NAME = "newmodel"
    SUPPORTED_VERSIONS = ["v1", "v2"]
    DEFAULT_VERSION = "v1"
    
    def load_model(self, checkpoint_path: str) -> None:
        # 加载模型
        pass
    
    def interpolate(self, frame0, frame1, timestep=0.5, **kwargs) -> VFIResult:
        # 插值逻辑
        pass
```

2. 在 `vfi_torch/__init__.py` 注册：

```python
from .newmodel import NewModelModel

MODEL_REGISTRY[ModelType.NEWMODEL] = NewModelModel
```

3. 在 `vfi_torch/base.py` 添加枚举：

```python
class ModelType(Enum):
    # ...
    NEWMODEL = "newmodel"
```

## 性能优化

### 半精度推理

```python
config = VFIConfig(
    model_type=ModelType.RIFE,
    fp16=True,  # 启用 FP16
)
```

### 缓存清理

```python
# 自动清理
torch_config = {
    "clear_cache_every": 10,  # 每 10 帧清理一次
}

# 手动清理
from core.torch_backend import clear_cache
clear_cache()
```

### 批量处理

```python
# 使用 interpolate_batch 自动批处理
output = model.interpolate_batch(frames, multiplier=2)
```

## 与 VapourSynth 处理器对比

| 特性 | TorchProcessor | VideoProcessor |
|------|----------------|----------------|
| 依赖 | PyTorch | VapourSynth + TensorRT |
| 模型 | RIFE/FILM/AMT/IFRNet | RIFE (vsrife) |
| 精度 | FP32/FP16 | FP16 (TensorRT) |
| 速度 | 较慢 | 较快 |
| 兼容性 | 广泛 | 需要 TensorRT |

## 参考

- [RIFE](https://github.com/hzwer/Practical-RIFE)
- [FILM](https://github.com/google-research/frame-interpolation)
- [AMT](https://github.com/MCG-NKU/AMT)
- [IFRNet](https://github.com/ltkong218/IFRNet)
- [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
