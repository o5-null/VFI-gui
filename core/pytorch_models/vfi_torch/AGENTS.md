# AGENTS.md — VFI 模型层

基于 `PyTorchVFIModel` 基类的 9 个视频帧插值模型实现。

> 本文件是项目分层 AGENTS.md 的第二层。上一层：[`VFI-gui/AGENTS.md`](../../../AGENTS.md)

---

## 支持的模型一览

| 模型 | 实现路径 | 限制 |
|------|----------|------|
| RIFE | `rife/` | — |
| FILM | `film/` | — |
| IFRNet | `ifrnet/` | — |
| AMT | `amt/` | — |
| XVFI | `xvfi/` | — |
| GMFSS | `gmfss/` | — |
| M2M | `m2m/` | 纯 PyTorch softsplat 实现（无 CuPy），~800ms CPU 224p |
| ATM | `atm/` | 仅 t=0.5 |
| MoMo | `momo/` | 仅 t=0.5, DDPM 8-step |

> **ATM/MoMo t=0.5 限制原因**: forward 不接收 timestep 参数，永远输出居中帧。多帧插值需递归二分，质量逐层下降。

---

## 模型注册

新增 VFI 模型需同时注册 **4 处**：

1. `base.py` — `ModelType` 枚举 + `get_model()` 的 `model_classes` 字典
2. `__init__.py` — `MODEL_REGISTRY` 字典 + `__all__` 导出
3. `../../../model_manager.py` — `MODEL_DEFINITIONS` 字典 + 下载 URL
4. `../../../scripts/run_inference.py` — `DEFAULT_CKPTS`（如有权重文件）

### 自定义算子

`ops/` 目录包含共享的纯 PyTorch 算子实现，用于需要特殊 CUDA 操作的模型：

| 算子 | 文件 | 用途 |
|------|------|------|
| `softsplat` | `ops/softsplat.py` | Softmax 前向扭曲（scatter_add + 双线性插值），替代 CuPy |
| `costvol` | `ops/costvol.py` | 相关性代价体（unfold + einsum），PWC-Net 风格 9×9 搜索 |

M2M 是首个使用这些算子的模型。新增模型若依赖这些算子，直接 `from ..ops import softsplat_func, costvol_func`。

---

## 接口约定

所有 VFI 模型继承 `PyTorchVFIModel`（定义于 `base.py`），基类提供：

**基类已实现**（子类无需重写）：
- `__init__(config, device, dtype)` — 从 `VFIConfig` 或 device/dtype 初始化
- `interpolate_batch(frames, multiplier, callback)` — 多帧批量插值
- `interpolate_with_result(frame0, frame1, timestep)` — 返回 `VFIResult`（含 metadata）
- `__call__(frames, **kwargs)` — callable 接口

**子类必须实现的两个抽象方法**：

```python
@abstractmethod
def load_model(self, checkpoint_path: str, **kwargs) -> None:
    """加载模型权重。只负责从本地路径加载，不触发下载。"""
    pass

@abstractmethod
def interpolate(
    self,
    frame0: torch.Tensor,
    frame1: torch.Tensor,
    timestep: float = 0.5,
    **kwargs
) -> torch.Tensor:
    """插值一对帧。"""
    pass
```

---

## 模型下载管理（⚡ 重要约束）

**`core/model_manager.py` 是模型下载 URL 和下载编排的唯一信源。**
单个模型文件（`rife/__init__.py`、`amt/__init__.py` 等）**不得**实现自己的下载逻辑。

- ❌ 禁止：在模型文件内定义 `_get_model_url()`、调用 `download_model()`
- ❌ 禁止：在模型文件内硬编码 Checkpoint URL
- ❌ 禁止：在模型文件内导入或调用 `utils.download_model()`
- ✅ 正确做法：`load_model()` 只做两件事：
  1. 调用 `ModelManager.ensure_checkpoint(model_type, ckpt_name)` 确保权重存在
  2. 从本地路径加载权重到模型结构
- ✅ `model_manager.py` 持有所有下载 URL（`MODEL_DEFINITIONS` 覆盖 15 个模型）
- ✅ `utils.download_model()` 作为底层函数，仅由 `ModelManager` 内部调用

**当前违反此规则的文件**（需逐步修复）：

| 文件 | 问题 |
|------|------|
| `amt/__init__.py` | `_get_model_url()` + `download_model()` 调用 |
| `ifrnet/__init__.py` | `_get_model_url()` + `download_model()` + `from utils import BASE_MODEL_URLS` |
| `film/__init__.py` | `_get_model_url()` + `download_model()` 调用 |
| `gmfss/__init__.py` | 硬编码 `CheckpointInfo` 权重 URL |
| `xvfi/__init__.py` | 导入了 `download_model` 但未使用（需清理 import） |

---

## 工具函数规范（`utils.py`）

允许所有模型共享的模块级工具：

| 函数 | 用途 |
|------|------|
| `load_model_weights(path)` | 加载 checkpoint（支持 state_dict 和 TorchScript 自动检测） |
| `InputPadder(dims, divisor=16)` | 填充/去填充，使输入维度可被 divisor 整除 |
| `get_device(device="auto")` | 自动选择 CUDA / XPU / MPS / CPU |
| `clear_cache()` | GPU 缓存清理（CUDA + XPU） |
| `preprocess_frames(frames)` | NHWC uint8 numpy → NCHW float32 tensor |
| `postprocess_frames(tensor)` | NCHW float32 tensor → NHWC uint8 numpy |
| `download_model(name, filename, save_dir)` | **仅由 ModelManager 调用** |

---

## 去重约束

新写或修改模型文件时，禁止以下重复模式：

### 1. `warp()` 函数

- ❌ 不要在模型文件内自定义 warp
- ✅ 应使用 `utils.py` 的共享实现
- 🔴 当前 4 个独立实现（amt, ifrnet, rife, momo）待统一

### 2. `ResBlock` 类

- ❌ 不要在多个模型文件内重定义结构相似的 ResBlock
- ✅ 使用参数化版本：`ResBlock(in_c, out_c, activation=nn.ReLU)`
- 🔴 AMT 用 PReLU，IFRNet 用 ReLU，其他一样

### 3. `interpolate()` batch/squeeze 模板

- ❌ 不要手写 batch 维度检查 + unsqueeze/squeeze
- ✅ 基类计划提供模板方法模式，子类只实现 `_interpolate_impl()`
- 🔴 当前 6/8 个模型文件重复此模板

### 4. `timestep` 张量创建

- ❌ 不要重复写 `torch.full((batch, 1), timestep, device=..., dtype=...)`
- ✅ 使用 `utils.make_timestep_tensor(batch, timestep, device, dtype)`（待抽取）

---

## 代码风格

### 导入顺序
标准库 → 第三方 → 本地模块（使用相对导入）：
```python
from __future__ import annotations
import sys, os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import load_model_weights, InputPadder
```

### 命名规范
- 函数/变量：`snake_case`
- 类：`PascalCase`
- 常量：`UPPER_SNAKE_CASE`
- 私有：前缀 `_`（如 `_create_model()`、`_model`）
- 内部网络子模块：`PascalCase`（如 `IFBlock`、`ResBlock`）

### 错误处理
```python
if not self._is_loaded:
    raise RuntimeError("Model not loaded. Call load_model() first.")
```

---

## 架构约束

- ATM/MoMo 仅支持 2x 插值（t=0.5），forward 不接收 timestep 参数
- `core/torch_backend/` 已删除，所有历史代码已迁移至此目录
- 模型权重存放于 `models/<model_type>/`，由 `ModelManager` 统一管理
- **不要使用子代理**（deep agent）实现模型代码，全部在主 context 手写
