# Agent 工作日志 — M2M VFI 模型实现

## 2026-06-17 — M2M (Many-to-Many Splatting) 模型实施

### 任务概述
在 VFI-gui 中实现 M2M 视频帧插值模型，包含：
- 纯 PyTorch softsplat（softmax 前向扭曲）和 costvol（相关性代价体）算子
- M2M 完整架构（PWC-Net 流网络 + 运动精化网络 + 多分支 splatting）
- 4 处注册（base.py, \_\_init\_\_.py, model\_manager.py, run\_inference.py）

### 文件清单

#### 新增文件
| 文件 | 行数 | 功能 |
|------|------|------|
| `core/pytorch_models/vfi_torch/ops/softsplat.py` | 155 | 纯 PyTorch softmax splatting（scatter_add + 双线性插值） |
| `core/pytorch_models/vfi_torch/ops/costvol.py` | 117 | 纯 PyTorch 相关性代价体（unfold + einsum，9×9 搜索窗） |
| `core/pytorch_models/vfi_torch/ops/__init__.py` | 19 | Ops 模块导出 |
| `core/pytorch_models/vfi_torch/m2m/arch.py` | 809 | M2M_PWC 完整架构 |
| `core/pytorch_models/vfi_torch/m2m/__init__.py` | 143 | M2MVFIModel 包装器（load_model / interpolate） |

### 架构说明

```
M2M_PWC
├── netFlow (PWC-Net 风格金字塔流估计)
│   ├── netExtract (特征金字塔: 8→16→32→64→96→128→192)
│   └── bidir():
│       ├── netCorr (代价体 + 流估计解码器 ×6 层)
│       └── 双向流 (tenFwd, tenBwd)
├── MRN (运动精化网络)
│   ├── motion_encdec (编解码器: down0..3 + up0..3)
│   │   └── 三平面注意力 (conv_C/H/W → 均值)
│   └── 分支复制: 4 分支
└── Many-to-Many Splatting
    ├── 4 分支流 → forwarp_mframe_mask()
    │   └── one_fdir(): 每个分支 softsplat 前向扭曲
    └── 合并: 加和平均 + mask 修复遮挡像素
```

- **参数量**: 7,610,715
- **推理耗时**: ~803ms @ CPU 224×224
- **模型权重**: `M2M.pth`（来自 ComfyUI-Frame-Interpolation Releases）

### 关键技术决策

#### 1. 纯 PyTorch softsplat（替代 CuPy）

原版 softsplat 需要 CuPy CUDA kernel。我们的纯 PyTorch 实现使用 `scatter_add_` + 双线性插值权重：

```python
# 对每个源像素，计算 4 个最近邻目标像素的权重
# 用 scatter_add_ 累加值到目标位置
# 最终除以累加的权重归一化
for corner in 4_corners:
    output[b].scatter_add_(dim=1,
        index=index[b].expand(C, -1),  # B, H*W → C, H*W
        src=src[b] * weight[b])        # 加权源值
```

- 每源像素贡献到 4 个近邻目标像素（双线性）
- 累加权重和值，最后值/权重归一化
- 性能：CPU 可运行，GPU 上与原生 kernel 差异不大

#### 2. 纯 PyTorch costvol

```python
tenTwo_unfold = F.unfold(tenTwo, kernel_size=9, padding=4)  # [B, C*81, H*W]
tenTwo_unfold = tenTwo_unfold.view(B, C, 81, H*W)
corr = torch.einsum("b c l, b c w l -> b w l", tenOne_flat, tenTwo_unfold)
```

- `F.unfold` 提取局部窗口
- `einsum` 计算每个空间位置与 81 个邻域的 dot product
- 输出 `[B, 81, H, W]`

#### 3. 4 分支架构

M2M 的核心创新：每个像素估计 4 个候选光流方向，各自 splat 到中间帧位置，然后加权平均。这解决了单流估计在遮挡/运动边界处的模糊问题。

### 调试过程

1. **RuntimeError: einsum mismatch** — `tenTwo_unfold` 5D vs 4D einsum，修正为 `(B, C, win, L)` 4D
2. **Conv2d channel mismatch** — `EncDec.down1` 误用为 `down0`（两路输入都是 8 通道，应使用相同 down0）
3. **cat dim mismatch** — 5D `[branch, B, ...]` 布局导致 softsplat 输入错位，改为 4D `[B*branch, ...]` 扁平布局

### 注册检查清单

- [x] `base.py` — `ModelType.M2M` 枚举 + `model_classes[ModelType.M2M]`
- [x] `__init__.py` — `MODEL_REGISTRY` + `__all__` 导出
- [x] `model_manager.py` — `MODEL_DEFINITIONS` + 下载 URL
- [x] `run_inference.py` — `DEFAULT_CKPTS` 条目

### 架构约束

- 与 ATM/MoMo 不同，M2M 支持任意 timestep（`fltTimes` 参数）
- 4 分支 splatting 自动处理遮挡：`mask * fallback_img` 补全未覆盖像素
- 输入归一化：逐图像 mean/std 标准化，输出反归一化
- 纯 PyTorch ops 无额外依赖（CuPy, Taichi 均不需要）

---

*Last updated: 2026-06-17*
