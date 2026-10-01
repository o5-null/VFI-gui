# ONNX Runtime 推理引擎实现

- 日期：2026-10-01 18:15
- 状态：✅ 完成（第一阶段；tensorrt-rtx 为第二阶段）

## 任务概述

按路线变更：**弃用 CUDA/torch 推理引擎**，改为两个引擎 **onnxruntime** 与 **tensorrt-rtx**。本阶段实现 ONNX Runtime 推理引擎（RIFE）并完成测试；tensorrt-rtx 留待第二阶段。

## 关键决策

1. **原生 onnxruntime 后端** — 直接使用 `onnxruntime.InferenceSession` + `CUDAExecutionProvider`，不集成 vs-mlrt VapourSynth 插件、不依赖 VapourSynth。
2. **模型来源** — 采用 vs-mlrt 预构建 ONNX（项目内原本零 ONNX 资产）。
3. **弃用落地方式** — 标记弃用 + 保留回退：torch/cuda 路径保留并标注 `DEPRECATED`，新任务默认走 ONNX，风险最低。
4. **执行提供程序（EP）** — 默认优先 CUDA EP，可选 DirectML EP，CPU 兜底；TensorRT EP 默认不自动启用（阶段二再接入）。
5. **DirectML 跳过** — `onnxruntime` / `onnxruntime-gpu` / `onnxruntime-directml` 安装同一个顶层 `onnxruntime` 模块，无法与 CUDA EP 同 venv 共存；用户确认本阶段跳过 DirectML。

## 实现

| 文件 | 变更 |
|------|------|
| `core/backends/onnx_backend.py` | 新建 `class OnnxBackend(BaseBackend)`（约 320 行） |
| `core/backends/__init__.py` | `_register_builtin_backends()` 新增 ONNX 注册块（:65-70） |
| `core/backends/torch_backend.py` | `TorchBackend.DEPRECATED = True` + docstring 标注弃用 |
| `scripts/test_onnx_backend.py` | 新建冒烟测试脚本 |
| `AGENTS.md` | 后端状态表更新 + ONNX 模型/依赖说明 |
| `runtime-requirements-cuda.txt` | `onnxruntime` → `onnxruntime-gpu>=1.27.0` |
| `docs/todo.md` | ONNX 待办标记完成 |

### OnnxBackend 关键点

- 属性：`BACKEND_TYPE = BackendType.ONNX`，`BACKEND_NAME = "ONNX Runtime"`，`SUPPORTED_MODELS = {"rife": ["4.0","4.6","4.7","4.17","4.22","4.26"]}`。
- 方法：`initialize / load_model / _resolve_model_path / _parse_device_id / _pack_input / infer / infer_batch / cancel / unload_model / cleanup`。
- 模块级禁止 import；`torch` / `onnxruntime` 均延迟导入（符合 `core/backends/AGENTS.md`）。
- 输入契约：`NCHW [1,C,H,W]` float32，RGB 取值 **[0,1] 原样喂入，无归一化**。
  - `C==7`（v2 模型）：`[f0, f1, tmap]`
  - `C==11`（v1 模型）：追加 `[horizontal(2j/(W-1)-1), vertical(2i/(H-1)-1), mul_h=2/(W-1), mul_w=2/(H-1)]`
  - 通道数、输入/输出名从 `session.get_inputs()/get_outputs()` 动态读取。
- EP 选择（`_build_providers`，:207）：
  - `_PROVIDER_PRIORITY = [Tensorrt, CUDA, Dml, CPU]`（:50）
  - `_DEFAULT_ACCELERATORS = ["CUDAExecutionProvider", "DmlExecutionProvider"]`（:58）
  - 支持 `config.extra["execution_providers"]` 覆盖；不可用的 EP 跳过并 warning，CPU 兜底。

### 使用方式

配置 `inference.backend = "onnx"` 即由 `BackendFactory` 选中新后端（`core/task_parser.py:148`）。

## 模型与运行时

- 模型：`models/rife_v2/rife_v4.26.onnx`（22,748,049 B，7 通道，multi=1，无需 tilesize 约束）。
  - 来源：`https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.26.7z`（同包另含 11 通道的 `rife/rife_v4.26.onnx`）。
- 运行时 `runtime/cuda`：卸载 CPU 版后安装 `onnxruntime-gpu==1.30.0` + cu13 运行库（`nvidia-*` 13.x、`nvidia-cudnn-cu13==9.27.0.42`）。
- 可用 providers：`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`。

## 测试结果

测试脚本 `scripts/test_onnx_backend.py`（256×256，runs=10），检查：shape/dtype/finite、identity、取值范围、timestep 敏感性、provider 报告、timing。

| 指标 | CPU EP | CUDA EP |
|------|--------|---------|
| session providers | `['CPUExecutionProvider']` | `['CUDAExecutionProvider','CPUExecutionProvider']` |
| identity MAE | 0.00247 | 0.00247（一致） |
| 输出范围 | [0.0008, 0.9989] | [0.0044, 0.9971] |
| 平均耗时 | ~44.6–55 ms | **11.09 ms**（约 4–5×） |

- 全部硬检查通过；timestep 敏感性正常（相邻 t 平均差 ≈0.29）。
- 非致命警告：`No registered plugin EP device found for 'CUDAExecutionProvider' with device_id=0` — ORT 1.30 plugin-EP 探测后回退内置 CUDA EP 工厂，GPU 实际生效（由 11 ms 耗时佐证）。

## 已知约束

- **DirectML 与 CUDA EP 不能同 venv 共存**，需独立环境；本阶段跳过。代码已支持 `DmlExecutionProvider`，如需启用单独建 venv。
- TensorRT EP 目前可用但默认不启用，阶段二接入 tensorrt-rtx（`NvTensorRTRTXExecutionProvider`，目前仅 Windows）。

## 输出文件

- 引擎：`core/backends/onnx_backend.py`
- 测试：`scripts/test_onnx_backend.py`
- 模型：`models/rife_v2/rife_v4.26.onnx`
- 本日志：`docs/dev-docs/20261001-181500-onnx-inference-backend.md`

## 经验总结

1. **ONNX EP 互斥** — `onnxruntime` / `-gpu` / `-directml` 共用同一模块，一个 venv 只能有一个；CUDA 与 DirectML 无法共存。
2. **CUDA EP 无 provider 级 fp16 开关** — fp16 只能靠模型级转换；仅 TRT EP 有 `trt_fp16_enable`。
3. **值为 [0,1] 直喂** — vs-mlrt RIFE ONNX 不做归一化，直接按 [0,1] 输入，v2 为 7 通道 `[A,B,t]`。
4. **plugin-EP 警告可忽略** — ORT 1.30 对内置 EP 也会先探测 plugin 通道，回退属正常。
