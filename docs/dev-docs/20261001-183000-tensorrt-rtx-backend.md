# TensorRT-RTX 推理引擎实现

- 日期：2026-10-01 18:30
- 状态：✅ 完成（第二阶段）

## 任务概述

承接第一阶段，实现第二个推理引擎 **tensorrt-rtx**（NVIDIA TensorRT for RTX，面向 RTX GPU 的推理库），使用其原生 Python API `tensorrt_rtx`，与 ONNX Runtime 引擎并列为两大加速引擎。

## 关键决策

1. **原生 `tensorrt_rtx` API**（非 ORT 插件 EP）。理由：`tensorrt-rtx` 与已装的 `onnxruntime-gpu` 完全独立（不同模块/DLL），可同 venv 共存；而 ORT 插件 EP `onnxruntime-ep-nv-tensorrt-rtx` 需卸载现有 `onnxruntime-gpu`（共用同一 `onnxruntime` 顶层模块），会牺牲现有 CUDA EP 后端。
2. **惰性构建静态引擎**。加载时尺寸未知，故 `load_model()` 仅解析 ONNX 取 I/O 信息；首次 `infer()` 按实际 `[1,C,H,W]` 构建 min=opt=max 的静态 engine 并落盘缓存。
3. **引擎磁盘缓存**：`models/trt_rtx_cache/<model>_<C>_<H>x<W>.engine`。构建仅约 0.8s，缓存使冷启动直接复用。
4. **共享输入打包**：抽出 `core/backends/rife_input.py`，`OnnxBackend` 与 `TensorRTRTXBackend` 共用，消除重复。
5. **新增枚举**：`BackendType.TENSORRT_RTX = "tensorrt_rtx"`（保留旧 `TENSORRT="tensorrt"` 未实现占位）。

## 实现

| 文件 | 变更 |
|------|------|
| `core/backends/tensorrt_rtx_backend.py` | 新建 `class TensorRTRTXBackend(BaseBackend)`（约 440 行） |
| `core/backends/rife_input.py` | 新建共享打包函数 `pack_rife_input(...)` |
| `core/backends/onnx_backend.py` | `_pack_input` 改为委托共享函数（去重，行为不变） |
| `core/backends/__init__.py` | 注册 `TensorRTRTXBackend` |
| `core/types.py` | `BackendType` 新增 `TENSORRT_RTX` |
| `core/engine_manager.py` | `_BACKEND_EXECUTION_MODE` 新增映射 |
| `scripts/test_tensorrt_rtx_backend.py` | 新建冒烟测试 |
| `runtime-requirements-cuda.txt` | 新增 `tensorrt-rtx>=1.6.0` |
| `AGENTS.md` / `docs/todo.md` | 状态更新 |

### 关键 API 流程（已实测）

```python
import tensorrt_rtx as trt
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger); parser.parse(open(path,"rb").read())
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
profile = builder.create_optimization_profile()
profile.set_shape("input", shape, shape, shape)       # 静态
config.add_optimization_profile(profile)
blob = bytes(builder.build_serialized_network(network, config))
engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
ctx = engine.create_execution_context(); ctx.set_input_shape("input", shape)
ctx.set_tensor_address("input", in_t.data_ptr())
ctx.set_tensor_address("output", out_t.data_ptr())
ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
```

### 性能陷阱（已修复）

原实现每次 `infer()` 都调用 `engine.create_execution_context()`——该调用首次约 **4.6 秒**，导致单次推理约 2.7 秒。改为**按 engine/shape 缓存 execution context 与输出 buffer**（每次仅重设输入地址）后，耗时降至正常水平。

## 测试结果

`scripts/test_tensorrt_rtx_backend.py`（256×256，runs=5）：

| 检查 | 结果 |
|------|------|
| shape/dtype/finite | PASS |
| identity MAE | 0.00246（阈值 0.02） |
| 输出范围 | [0.0036, 0.9993] |
| timestep 敏感性 | 相邻 t 平均差 ≈0.29 |
| 引擎缓存 | MISS→构建落盘；冷启动后端 HIT，文件 mtime 不变 |
| end-to-end timing | ~11.6 ms（含打包 + H2D/D2H 拷贝） |

对照（同一测试框架，256×256）：ONNX Runtime CUDA EP end-to-end ~12.6 ms。

- **纯 `execute_async_v3` 约 2.2–2.8 ms**（探针，buffer 复用无拷贝）；end-to-end 数值受 H2D/D2H 与输入打包主导，与 ORT CUDA EP 接近，但内核更快、引擎构建更快（0.8s）。
- engine 构建 0.8 s，序列化约 27.9 MB。

### 回归

`scripts/test_onnx_backend.py` 重跑 **ALL HARD CHECKS PASSED**（identity MAE=0.00247，CUDA EP timing ~12.6 ms），确认 `_pack_input` 重构无回归。

## 已知约束

- 使用当前 CUDA 流，TensorRT 打印 `Using default stream in enqueueV3()` 性能提示（功能正确）。
- execution context 按 shape 缓存，非线程安全——同一 backend 实例应由单线程串行调用（符合现有调度模型）。
- TensorRT-RTX 引擎与库版本强绑定，升级 `tensorrt-rtx` 后缓存需重建；实现已对反序列化失败做兜底重建。
- DirectML / ORT 插件 EP 路线未采用（见「关键决策」）。

## 输出文件

- 引擎：`core/backends/tensorrt_rtx_backend.py`、`core/backends/rife_input.py`
- 测试：`scripts/test_tensorrt_rtx_backend.py`
- 引擎缓存：`models/trt_rtx_cache/rife_v4.26_7_256x256.engine`
- 本日志：`docs/dev-docs/20261001-183000-tensorrt-rtx-backend.md`

## 经验总结

1. **execution context 极贵** — TensorRT-RTX 的 `create_execution_context()` 首调数秒，务必缓存复用。
2. **选型看模块冲突** — 原生 `tensorrt_rtx` 与 `onnxruntime-gpu` 互不干扰；ORT 插件 EP 会与现有 ORT 模块冲突。
3. **`create_network(0)` 可用** — 该 TRT-RTX 构建的 `NetworkDefinitionCreationFlag` 仅含 `STRONGLY_TYPED`，实测传 0 即可正常解析/构建。
4. **静态 profile 足够** — 每分辨率一个 engine，构建仅 0.8s，比猜测 min/max 动态范围更简单可靠。
