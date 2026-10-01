# AGENTS.md — Backend 后端层

推理后端抽象层：统一的推理接口 + 策略化后端选择。

> 本文件是项目分层 AGENTS.md 的第二层。上一层：[`VFI-gui/AGENTS.md`](../../AGENTS.md)

---

## 结构

```
core/backends/
├── base_backend.py       # BaseBackend(ABC) — 后端接口定义
├── backend_factory.py    # BackendFactory — 按类型创建后端实例
├── inprocess_backend.py  # InProcessBackend — 进程内 PyTorch 推理  ✅
├── subprocess_backend.py # SubProcessBackend — 子进程 PyTorch 推理 ✅
├── torch_backend.py      # TorchBackend — 旧式后端（待废弃）
├── directml_backend.py   # DirectML 后端（计划）
├── ncnn_backend.py       # NCNN 后端（计划）
├── cuda_stream_pool.py   # CUDA 流池并行
└── inference_thread_pool.py  # 推理线程池
```

## 架构

### BaseBackend（策略模式接口）

```
BaseBackend(ABC)
├── BackendType.VAPOURSYNTH → VapourSynthBackend
├── BackendType.TORCH       → InProcessBackend / SubProcessBackend
├── BackendType.TENSORRT    → ❌ 未实现
└── BackendType.ONNX        → ❌ 未实现
```

### 纯推理接口

```python
class BaseBackend(ABC):
    def infer(self, request: InferenceRequest) -> InferenceResult: ...
    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]: ...

    # 生命周期
    def initialize(self, model: Any, device: str) -> None: ...
    def shutdown(self) -> None: ...
```

**核心约束**（定义在 `base_backend.py` 的 docstring 中）：
- Backend **不接触文件路径**，只接收 numpy/tensor 数据
- Backend **不自主决定 IO 时机**，由 TaskScheduler 调度
- Backend **不直接写文件**，推理结果返回给 TaskScheduler

### BackendFactory（工厂模式）

```python
backend = BackendFactory.create(backend_type=BackendType.PYTORCH, config=...)
```

- 按 `BackendType` 枚举选择后端实现
- InProcess：同进程推理，适合单个 GPU
- SubProcess：子进程隔离，适合多 GPU 并行

### InferenceStrategySelector（优先级策略）

```
选择优先级:
1. BATCH        — 模型支持 batch 推理（最高吞吐）
2. CUDA_STREAMS — NVIDIA CUDA GPU 可用（流式并行）
3. MULTI_MODEL  — CPU/XPU 回退（实例并行）
4. SERIAL       — 单线程默认
```

## 约束

- ❌ Backend 内不允许 `import torch` 在模块级（lazy import）
- ❌ Backend 不允许依赖 `cv2`/`PIL`
- ❌ Backend 不允许直接写文件或操作路径
- ✅ 新增后端：继承 `BaseBackend` + 在 `BackendFactory.create()` 注册
- ✅ 所有后端元数据通过类属性声明（`BACKEND_TYPE`, `SUPPORTED_MODELS` 等）
