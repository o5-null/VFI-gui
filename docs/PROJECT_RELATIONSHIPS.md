# 项目架构关系文档

## 概述

本文档说明 VFI-gui 项目与相关参考项目之间的架构关系。

---

## 项目关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    VFI-gui (我们的项目)                      │
│  PyQt6 GUI + VapourSynth脚本生成 + vspipe+ffmpeg流水线      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              VSGAN-tensorrt-docker (基础设施)               │
│  • scene_detect.py - 场景检测 (17种ONNX模型)                │
│  • rife_trt.py - RIFE TensorRT直接推理(不推荐，有artifacts) │
│  • dedup.py - 帧去重                                        │
│  • download.py - 模型下载工具                               │
│  • utils.py - 工具函数 (FastLineDarkenMOD等)                │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│    vs-mlrt      │ │    vs-rife      │ │ ComfyUI-Frame-Interp│
│  (推理框架)     │ │  (RIFE专用包)   │ │  (多模型扩展)       │
│                 │ │                 │ │                     │
│ • RIFE ✅       │ │ • RIFE ✅       │ │ • RIFE ✅           │
│ • Waifu2x       │ │ • 场景检测      │ │ • FILM ✅           │
│ • RealESRGAN    │ │ • TensorRT优化  │ │ • IFRNet ✅         │
│ • CUGAN         │ │ • 多版本支持    │ │ • AMT ✅            │
│ • DPIR          │ │                 │ │ • GMFSS ✅          │
│ • SAFA          │ │                 │ │ • STMFNet ✅        │
│ • SCUNet        │ │                 │ │ • FLAVR ✅          │
│ • SwinIR        │ │                 │ │ • CAIN ✅           │
│ • ArtCNN        │ │                 │ │ • ATM ✅            │
│                 │ │                 │ │ • MoMo ✅           │
│                 │ │                 │ │ • M2M ✅            │
│                 │ │                 │ │ • SepConv ✅        │
│                 │ │                 │ │ • IFUnet ✅         │
│                 │ │                 │ │ • XVFI ✅           │
│                 │ │                 │ │ • EISAI ✅          │
└─────────────────┘ └─────────────────┘ └─────────────────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    推理后端                                  │
│  TensorRT (vstrt) │ ONNX Runtime (vsort) │ NCNN (vsncnn)   │
│  OpenVINO (vsov)  │ MIGraphX (vsmigx)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 各项目详细说明

### 1. VSGAN-tensorrt-docker

**定位**: 基础设施提供者

**GitHub**: https://github.com/styler00dollar/VSGAN-tensorrt-docker

**核心功能**:
- 场景检测 (17种ONNX模型)
- 帧去重
- 模型下载工具
- TensorRT推理示例

**注意**: 
- `rife_trt.py` 不推荐使用，有artifacts问题
- RIFE插帧通过 `vs-rife` 包实现，不在src/中
- `download.py` 包含FILM下载函数，但无推理代码

**已集成到VFI-gui**:
- `core/vsgan/src/scene_detect.py`
- `core/vsgan/src/dedup.py`
- `core/vsgan/src/download.py`
- `core/vsgan/src/utils.py`

---

### 2. vs-mlrt

**定位**: VapourSynth ML推理运行时框架

**GitHub**: https://github.com/AmusementClub/vs-mlrt

**支持的后端**:

| 后端 | 目录 | 适用平台 | 特点 |
|------|------|---------|------|
| TensorRT | `vstrt/` | NVIDIA GPU | 最快，需要构建engine |
| ONNX Runtime | `vsort/` | CPU/NVIDIA GPU | 通用，支持CUDA |
| OpenVINO | `vsov/` | Intel CPU/GPU | Intel优化 |
| NCNN | `vsncnn/` | 通用GPU (Vulkan) | 跨平台，无CUDA依赖 |
| MIGraphX | `vsmigx/` | AMD GPU | AMD优化 |

**vsmlrt.py 支持的模型**:

```python
__all__ = [
    "Waifu2x",      # 超分辨率/去噪
    "DPIR",         # 去噪/去块效应
    "RealESRGAN",   # 超分辨率
    "CUGAN",        # 超分辨率
    "RIFE",         # 视频插帧 ✅
    "SAFA",         # 时空超分辨率
    "SCUNet",       # 去噪
    "SwinIR",       # 超分辨率/去噪
    "ArtCNN",       # 超分辨率
]
```

**RIFE支持详情**:
- 版本: 4.0 ~ 4.26 (包括 lite, heavy 变体)
- 模型路径: `models/rife/`, `models/rife_v2/`
- 支持场景检测、多倍插帧、scale调整

**使用示例**:
```python
from vsmlrt import RIFE, RIFEModel, Backend

clip = RIFE(
    clip,
    multi=2,                    # 插帧倍数
    scale=1.0,                  # 处理分辨率
    model=RIFEModel.v4_22,      # 模型版本
    backend=Backend.TRT(fp16=True),  # TensorRT后端
)
```

---

### 3. vs-rife

**定位**: RIFE专用VapourSynth包

**GitHub**: https://github.com/HolyWu/vs-rife

**特点**:
- 官方维护
- TensorRT优化
- 支持场景检测
- 多版本RIFE模型 (4.0-4.26)

**安装**:
```bash
pip install vsrife
```

**使用**:
```python
from vsrife import rife

clip = rife(
    clip, 
    model="4.22", 
    multi=2, 
    scale=1.0, 
    sc=False,      # 场景检测
    trt=True,      # TensorRT加速
)
```

---

### 4. ComfyUI-Frame-Interpolation

**定位**: 多模型插帧扩展

**GitHub**: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

**README说明**: "约99%代码来自VSGAN-tensorrt-docker"

**真实含义**:
- 工具函数来自VSGAN: `preprocess_frames`, `postprocess_frames`, `generic_frame_loop`
- 模型下载机制来自VSGAN: `load_file_from_github_release`
- **16种插帧模型架构是ComfyUI自己实现的**

**支持的插帧模型**:

| 模型 | 目录 | 特点 | 最小帧数 |
|------|------|------|---------|
| RIFE | `vfi_models/rife/` | 实时插帧，4.0-4.9 | 2 |
| FILM | `vfi_models/film/` | 大运动插帧，Google | 2 |
| IFRNet | `vfi_models/ifrnet/` | 中间特征细化 | 2 |
| AMT | `vfi_models/amt/` | 全对多场变换 | 2 |
| GMFSS Fortuna | `vfi_models/gmfss_fortuna/` | 动画专用 | 2 |
| IFUnet | `vfi_models/ifunet/` | RIFE+IFUNet融合 | 2 |
| M2M | `vfi_models/m2m/` | 多对多splatting | 2 |
| SepConv | `vfi_models/sepconv/` | 自适应可分离卷积 | 2 |
| ST-MFNet | `vfi_models/stmfnet/` | 时空多流网络 | 4 |
| FLAVR | `vfi_models/flavr/` | 流无关视频表示 | 4 |
| ATM-VFI | `vfi_models/atm/` | 注意力到运动 | 2 (仅2x) |
| MoMo | `vfi_models/momo/` | 解耦运动建模 | 2 (仅2x) |
| CAIN | `vfi_models/cain/` | 通道注意力 | 2 |
| EISAI | `vfi_models/eisai/` | - | 2 |
| XVFI | `vfi_models/xvfi/` | 极端视频插帧 | 2 |

---

## 模型来源对比

### 插帧模型支持对比

| 模型 | VSGAN | vs-mlrt | vs-rife | ComfyUI |
|------|-------|---------|---------|---------|
| RIFE | ❌ (仅vs-rife) | ✅ | ✅ | ✅ |
| FILM | ❌ (仅下载) | ❌ | ❌ | ✅ |
| IFRNet | ❌ | ❌ | ❌ | ✅ |
| AMT | ❌ | ❌ | ❌ | ✅ |
| GMFSS | ❌ | ❌ | ❌ | ✅ |
| ST-MFNet | ❌ | ❌ | ❌ | ✅ |
| FLAVR | ❌ | ❌ | ❌ | ✅ |
| 其他8种 | ❌ | ❌ | ❌ | ✅ |

### 超分辨率模型支持对比

| 模型 | VSGAN | vs-mlrt | 说明 |
|------|-------|---------|------|
| RealESRGAN | ✅ | ✅ | 通用超分辨率 |
| CUGAN | ✅ | ✅ | 动画超分辨率 |
| Waifu2x | ✅ | ✅ | 动画去噪/放大 |
| DPIR | ✅ | ✅ | 去噪/去块 |
| SwinIR | ❌ | ✅ | 超分辨率/去噪 |
| SCUNet | ❌ | ✅ | 去噪 |
| ArtCNN | ❌ | ✅ | 超分辨率 |

---

## VFI-gui 集成建议

### 短期方案 (当前)
- 使用 `vs-rife` 包进行RIFE插帧
- 使用 VSGAN 的 `scene_detect` 进行场景检测
- 使用 VSGAN 的 `dedup` 进行去重

### 中期方案
- 集成 `vs-mlrt` 统一后端
- 支持多种推理后端 (TensorRT, ONNX Runtime)
- 添加超分辨率支持 (RealESRGAN, CUGAN)

### 长期方案
- 移植 ComfyUI 的多模型支持
- 支持 FILM, IFRNet, AMT 等16种插帧模型
- 统一模型管理和下载

---

## 模型下载地址

### VSGAN模型
```
https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/
```

### ComfyUI模型
```
https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/
```

### vs-mlrt模型
```
https://github.com/AmusementClub/vs-mlrt/releases/download/v7/models.v7.7z
```

### FILM模型
```
https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/
```

### AMT模型
```
https://huggingface.co/lalala125/AMT/resolve/main/
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
