# RIFE ONNX 全模型双后端冒烟测试 — 2026-10-01

**时间:** 2026-10-01 18:35:00
**状态:** ⚠️ 已知问题（10/12 通过，TRT-RTX 上 v4.0 / v4.6 identity 失败）

## 问题描述 / 任务概述

对当前两个已实现推理后端 —— **ONNX Runtime**（`core/backends/onnx_backend.py`）与 **TensorRT-RTX**（`core/backends/tensorrt_rtx_backend.py`）—— 进行**所有 RIFE 模型**的冒烟测试。

两个后端声明的 RIFE 版本均为 `["4.0","4.6","4.7","4.17","4.22","4.26"]`（`onnx_backend.py:81`、`tensorrt_rtx_backend.py:74`）。此前 `models/rife_v2/` 仅有 `rife_v4.26.onnx`，需先补齐其余 5 个 ONNX 权重。

测试类型：**冒烟测试**（能加载、能推理、输出 shape/dtype/数值正常 + 时延）。

## 根因分析 / 执行过程

### 1. 权重补齐

ONNX 模型唯一来源为 vs-mlrt 的 `external-models` release（GitHub `AmusementClub/vs-mlrt`），每个 `rife_vX.Y.7z` 内含 `rife/`(11ch) 与 `rife_v2/`(7ch)。项目采用 **7ch 变体**，解析路径 `models/rife_v2/rife_<version>.onnx`。

- 4.7 / 4.17 / 4.22 有独立包；
- **4.0 与 4.6 无独立包**，仅存在于大包 `rife_v2_v4.7z`。

下载 4 个 7z 至 `D:\code\VFI\.cache\onnx_dl\`，用 NanaZip `7z e -o<out>` 提取到 `models/rife_v2/`。最终目录：

| 文件 | 大小 (B) |
|------|----------|
| `rife_v4.0.onnx` | 20,705,015 |
| `rife_v4.6.onnx` | 21,297,017 |
| `rife_v4.7.onnx` | 21,374,472 |
| `rife_v4.17.onnx` | 21,532,608 |
| `rife_v4.22.onnx` | 37,365,121 |
| `rife_v4.26.onnx` | 22,748,049 |

> `core/model_manager.py:105 MODEL_DEFINITIONS` 只登记 `.pth` 下载源，**不含任何 ONNX 下载源**；ONNX 由 `_scan_onnx` 扫描 `models_dir/**/*.onnx`。

### 2. 执行环境

- `runtime/cuda`：torch CUDA=True，**NVIDIA RTX 3080**；onnxruntime 1.30.0；tensorrt_rtx 1.6.1.120
- `runtime/xpu`：torch XPU=True，Intel Arc 140V (16GB)
- 两个后端均需 CUDA，故统一使用 `D:\code\VFI\runtime\cuda\Scripts\python.exe`
- 脚本 `_resolve_runtime_python()` 探测的是 `VFI-gui\runtime\{xpu,cuda}`（不存在），故不会 re-exec，须显式用 cuda runtime python 调用

### 3. 执行命令

```powershell
# ONNX Runtime，6 个版本
D:\code\VFI\runtime\cuda\Scripts\python.exe scripts\test_onnx_backend.py --model models/rife_v2/rife_v4.0.onnx --runs 5
# TensorRT-RTX，6 个版本
D:\code\VFI\runtime\cuda\Scripts\python.exe scripts\test_tensorrt_rtx_backend.py --model models/rife_v2/rife_v4.0.onnx --runs 5
```

日志：`D:\code\VFI\.cache\smoke\{onnx,trt}_<version>.log`，帧尺寸 256×256，timestep 0.5。

## 解决方案 / 测试结果

### ONNX Runtime 后端（6/6 全通过）

| 版本 | identity MAE | 范围 [min, max] | 时延 (5 run) | 结果 |
|------|--------------|-----------------|--------------|------|
| 4.0  | 0.01643 | [0.031, 0.968] | 8.09 ms | ✅ |
| 4.6  | 0.01184 | [0.022, 0.982] | 10.50 ms | ✅ |
| 4.7  | 0.00682 | [0.015, 0.981] | 9.35 ms | ✅ |
| 4.17 | 0.00239 | [0.001, 0.999] | 13.14 ms | ✅ |
| 4.22 | 0.00672 | [0.002, 0.996] | 11.26 ms | ✅ |
| 4.26 | 0.00247 | [0.004, 0.999] | 12.32 ms | ✅ |

全部 session provider = `CUDAExecutionProvider`；硬检查（shape/dtype/finite、identity MAE<0.02、范围 [-0.05,1.05]、timing）通过。

### TensorRT-RTX 后端（4/6 通过）

| 版本 | identity MAE | 范围 [min, max] | 时延 (5 run) | 引擎缓存 | 结果 |
|------|--------------|-----------------|--------------|----------|------|
| 4.0  | **0.11059** | [0.027, 0.973] | 5.16 ms | MISS→built | ❌ identity |
| 4.6  | **0.14724** | [0.025, 0.983] | 4.01 ms | MISS→built | ❌ identity |
| 4.7  | 0.00682 | [0.004, 0.984] | 4.11 ms | MISS→built | ✅ |
| 4.17 | 0.00238 | [0.002, 0.999] | 13.40 ms | MISS→built | ✅ |
| 4.22 | 0.00670 | [0.003, 0.996] | 4.81 ms | MISS→built | ✅ |
| 4.26 | 0.00246 | [0.004, 0.999] | 7.51 ms | HIT | ✅ |

- 引擎缓存均正确创建，冷启动复用 MAE mtime 不变 = True。
- 4.0 / 4.6 复跑（使用已缓存引擎）**数值精确复现**（0.11059 / 0.14724），排除 GPU 抢占/偶发因素。

### 关键发现

- **同一份** `rife_v4.0.onnx` / `rife_v4.6.onnx` 在 ONNX 后端 MAE=0.016/0.012（通过），在 TRT-RTX 上却达 0.111/0.147（阈值 0.02）→ **TRT-RTX 精度路径显著放大误差**。
- 注意 v4.0/v4.6 即使 ONNX 上其 identity MAE 也约为其它版本的 **3–7 倍**（0.016/0.012 vs 0.002–0.007），说明这两个模型本身对数值精度更敏感。
- TRT 后端 `_ensure_engine` 仅设置 workspace 内存池与优化 profile，**未显式指定精度标志**，默认走 FP32 + Ampere **TF32** 加速路径（`tensorrt_rtx_backend.py:269-276`）。
- 输出并非乱码：范围检查与 timestep 敏感性（t=0.25→0.5→0.75 差异 0.177/0.185）均正常，属"可用但精度不足"。

**待排查方向：** 为 v4.0 / v4.6 单独构建禁用 TF32（强制 FP32）的引擎，或将 identity 阈值按版本分档；判断大包 `rife_v2_v4.7z` 内的 4.0/4.6 与独立包的导出精度是否一致。

## 验证

- ONNX v4.0 单跑预先验证链路（identity MAE=0.01643，CUDA EP，8.25ms）。
- 两后端 × 6 版本批量运行，退出码：ONNX 全 0；TRT 4.7/4.17/4.22/4.26=0，4.0/4.6=1。
- 对失败项复跑确认可复现。

## 相关文件

- `core/backends/onnx_backend.py`（RIFE 版本注册 `:81`）
- `core/backends/tensorrt_rtx_backend.py`（版本注册 `:74`，引擎构建 `:269-276`）
- `core/model_manager.py`（`MODEL_DEFINITIONS:105`，`_scan_onnx:442-455`）
- `scripts/test_onnx_backend.py`、`scripts/test_tensorrt_rtx_backend.py`
- `models/rife_v2/rife_v4.{0,6,7,17,22,26}.onnx`
- `models/trt_rtx_cache/rife_v4.{0,6,7,17,22,26}_7_256x256.engine`
- 日志：`D:\code\VFI\.cache\smoke\*.log`

## 经验总结

1. **ONNX 权重无自动下载**：需手动从 vs-mlrt `external-models` release 取 7z；4.0/4.6 只能从大包 `rife_v2_v4.7z` 提取。
2. **务必显式用 runtime python**：脚本的 runtime 自动探测因路径约定错位而失效，默认回退 sys.executable。
3. **TRT-RTX 默认 TF32**：对数值敏感的旧版模型可能超出 identity 阈值；ONNX 与 TRT 的精度路径不同，同一 ONNX 结果可能不一致。
4. **冒烟阈值应按版本校准**：统一 0.02 identity 阈值对 RIFE 4.0/4.6 过严。
