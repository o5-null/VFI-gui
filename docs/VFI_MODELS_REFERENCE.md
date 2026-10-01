# 视频插帧模型参考文档

## 概述

本文档整理了 ComfyUI-Frame-Interpolation 项目中支持的所有视频插帧模型，供 VFI-gui 开发参考。

**源项目**: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation  
**代码来源**: 约99%代码来自 VSGAN-tensorrt-docker

---

## 支持的插帧模型 (16种)

| 模型 | 目录 | 特点 | 最小帧数要求 |
|------|------|------|-------------|
| RIFE | `vfi_models/rife/` | 实时插帧，支持4.0-4.9版本 | 2帧 |
| FILM | `vfi_models/film/` | 大运动插帧，Google出品 | 2帧 |
| IFRNet | `vfi_models/ifrnet/` | 中间特征细化网络 | 2帧 |
| AMT | `vfi_models/amt/` | 全对多场变换 | 2帧 |
| GMFSS Fortuna | `vfi_models/gmfss_fortuna/` | 动画专用 | 2帧 |
| IFUnet | `vfi_models/ifunet/` | RIFE+IFUNet融合 | 2帧 |
| M2M | `vfi_models/m2m/` | 多对多splatting | 2帧 |
| SepConv | `vfi_models/sepconv/` | 自适应可分离卷积 | 2帧 |
| ST-MFNet | `vfi_models/stmfnet/` | 时空多流网络 | 4帧 |
| FLAVR | `vfi_models/flavr/` | 流无关视频表示 | 4帧 |
| ATM-VFI | `vfi_models/atm/` | 注意力到运动Transformer | 2帧，仅2x |
| MoMo | `vfi_models/momo/` | 解耦运动建模 | 2帧，仅2x |
| CAIN | `vfi_models/cain/` | 通道注意力 | 2帧 |
| EISAI | `vfi_models/eisai/` | - | 2帧 |
| XVFI | `vfi_models/xvfi/` | 极端视频插帧 | 2帧 |

---

## 模型下载URL

### 基础下载地址
```python
BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]
```

### RIFE 模型
```python
CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
}
```

### FILM 模型
- `film_net_fp32.pt`

### IFRNet 模型
```python
CKPT_NAMES = [
    "IFRNet_S_Vimeo90K.pth",
    "IFRNet_L_Vimeo90K.pth", 
    "IFRNet_S_GoPro.pth",
    "IFRNet_L_GoPro.pth"
]
```

### AMT 模型
```python
CKPT_CONFIGS = {
    "amt-s.pth": {"network": AMT_S, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 3}},
    "amt-l.pth": {"network": AMT_L, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 5}},
    "amt-g.pth": {"network": AMT_G, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 5}},
    "gopro_amt-s.pth": {"network": AMT_S, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 3}}
}
```
下载地址: `https://huggingface.co/lalala125/AMT/resolve/main/{ckpt_name}`

---

## 核心工具函数

### 帧预处理
```python
def preprocess_frames(frames):
    """将 NHWC 格式转换为 NCHW"""
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames):
    """将 NCHW 格式转换回 NHWC"""
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()
```

### 通用帧循环
```python
def generic_frame_loop(
    model_name,
    frames,
    clear_cache_after_n_frames,
    multiplier,  # 插帧倍数
    return_middle_frame_function,  # 模型推理函数
    *return_middle_frame_function_args,
    interpolation_states=None,
    use_timestep=True,
    dtype=torch.float32,
    batch_size=1
):
    ...
```

### 模型下载
```python
def load_file_from_github_release(model_type, ckpt_name):
    """从GitHub release下载模型文件"""
    ...
```

---

## 模型接口模式

### 模式1: 时间步推理 (RIFE, IFRNet, AMT等)
```python
def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor):
    return model(frame_0, frame_1, timestep, scale_factor)
```

### 模式2: 递归推理 (FILM)
```python
def inference(model, img_batch_1, img_batch_2, inter_frames):
    # 二分递归生成中间帧
    ...
```

### 模式3: 批量推理 (RIFE优化)
```python
# 支持batch_size > 1，多帧对并行处理
middle_frames = interpolation_model(
    frame0_batch,  # [B, C, H, W]
    frame1_batch,  # [B, C, H, W]
    timestep_tensor,  # [B, 1, 1, 1]
    scale_list,
    fast_mode,
    ensemble,
)
```

---

## RIFE 详细参数

```python
class RIFE_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["rife47.pth", "rife49.pth", "rife417.pth", "rife426.pth"],),
                "frames": ("IMAGE",),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float32"}),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES",)
            }
        }
```

### RIFE 版本差异
- **4.26**: 不支持ensemble
- **4.0-4.9**: 支持fast_mode, ensemble, scale_factor
- scale_list计算:
  - 4.26: `[16/scale, 8/scale, 4/scale, 2/scale, 1/scale]`
  - 其他: `[8/scale, 4/scale, 2/scale, 1/scale]`

---

## FILM 详细参数

```python
class FILM_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["film_net_fp32.pt"],),
                "frames": ("IMAGE",),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
            }
        }
```

### FILM 特点
- 使用 `torch.jit.load` 加载模型
- 仅支持 float32
- 二分递归生成中间帧
- 适合大运动场景

---

## IFRNet 详细参数

```python
class IFRNet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["IFRNet_S_Vimeo90K.pth", "IFRNet_L_Vimeo90K.pth", 
                               "IFRNet_S_GoPro.pth", "IFRNet_L_GoPro.pth"],),
                "frames": ("IMAGE",),
                "clear_cache_after_n_frames": ("INT", {"default": 10}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
            }
        }
```

### IFRNet 架构选择
- **S (Small)**: 轻量级，速度快
- **L (Large)**: 精度更高
- **Vimeo90K**: 通用视频
- **GoPro**: 运动模糊场景

---

## AMT 详细参数

```python
class AMT_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["amt-s.pth", "amt-l.pth", "amt-g.pth", "gopro_amt-s.pth"],),
                "frames": ("IMAGE",),
                "clear_cache_after_n_frames": ("INT", {"default": 1}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000})
            }
        }
```

### AMT 特点
- 需要 InputPadder 处理16的倍数
- 使用 `embt` 参数传递时间步
- 返回字典 `{"imgt_pred": tensor}`

---

## 内存管理

### 缓存清理
```python
from comfy.model_management import soft_empty_cache

# 每处理N帧清理一次
if frames_processed >= clear_cache_after_n_frames:
    soft_empty_cache()
    gc.collect()
```

### 模型缓存 (RIFE示例)
```python
_model_cache: Dict[Tuple, torch.nn.Module] = {}

def get_model(ckpt_name, dtype, torch_compile):
    cache_key = (ckpt_name, dtype, torch_compile)
    if cache_key not in _model_cache:
        model = load_model(ckpt_name)
        _model_cache[cache_key] = model
    return _model_cache[cache_key]
```

---

## 推荐使用场景

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 通用视频 | RIFE 4.9 | 平衡速度和质量 |
| 动画 | GMFSS Fortuna | 专为动画优化 |
| 大运动 | FILM | 处理大幅度运动 |
| 实时处理 | RIFE + fast_mode | 最快速度 |
| 低显存 | RIFE + float16 | 减半显存占用 |
| 高质量 | AMT-G 或 IFRNet-L | 更高精度 |

---

## 文件结构参考

```
vfi_models/
├── __init__.py
├── rife/
│   ├── __init__.py      # RIFE_VFI 类
│   └── rife_arch.py     # IFNet 架构
├── film/
│   ├── __init__.py      # FILM_VFI 类
│   └── film_arch.py     # FILM 架构
├── ifrnet/
│   ├── __init__.py      # IFRNet_VFI 类
│   ├── IFRNet_S_arch.py # Small 架构
│   └── IFRNet_L_arch.py # Large 架构
├── amt/
│   ├── __init__.py      # AMT_VFI 类
│   └── amt_arch.py      # AMT 架构
└── ops/
    └── ...              # 通用操作
```

---

## 参考文献

### RIFE
```bibtex
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={ECCV},
  year={2022}
}
```

### FILM
```bibtex
@inproceedings{reda2022film,
  title={FILM: Frame Interpolation for Large Motion},
  author={Reda, Fitsum and Kontkanen, Janne and Tabellion, Eric and Sun, Deqing and Pantofaru, Caroline and Curless, Brian},
  booktitle={ECCV},
  year={2022}
}
```

### IFRNet
```bibtex
@InProceedings{Kong_2022_CVPR,
  author={Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie},
  title={IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation},
  booktitle={CVPR},
  year={2022}
}
```

### AMT
```bibtex
@inproceedings{licvpr23amt,
  title={AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation},
  author={Li, Zhen and Zhu, Zuo-Liang and Han, Ling-Hao and Hou, Qibin and Guo, Chun-Le and Cheng, Ming-Ming},
  booktitle={CVPR},
  year={2023}
}
```
