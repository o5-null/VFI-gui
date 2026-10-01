# ONNX / TensorRT-RTX 后端多模型推理支持

- 日期：2026-10-01
- 契约文档：`VFI-gui/docs/ONNX_TRT_MULTI_MODEL_SPEC.md`（本次固化的权威接口定义）
- 范围：为 `OnnxBackend` 与 `TensorRTRTXBackend` 增加多模型类型分派支持；不含 ONNX 导出。

## 目标与决策

- UI 层「多种模型类型」需求 → 后端支持集为 **9 型**：`rife / film / amt / xvfi / atm / momo / m2m / stmfnet / flavr`。
- `cain / sepconv` 无 PyTorch 参考实现，按支持集约定**明确拒绝**（`UnsupportedAdapter`）。
- 架构：新增按 `model_type` 分派的 **ModelAdapter 注册表**，把「路径解析 / 输入打包 / shape 对齐 / 输出后处理 / 帧数语义」从两个后端抽出；后端只保留「执行」与「引擎缓存」。
- 执行层由单输入单输出 `[1,C,H,W]→[1,3,H,W]` **泛化为 N 输入 / N 输出**。
- 多帧模型（STMFNet/FLAVR）Phase 1 用「2→4 复制」，`InferenceRequest` 接口不变。

## 改动清单

### 新增
- `core/models/asset_resolver.py`：`MODEL_ASSET_TABLE` 作为版本↔checkpoint↔onnx 的**单一真相源**；`DEFAULT_VERSIONS`、`checkpoint_to_version`、`version_to_checkpoint`、`resolve_checkpoint_path`、`resolve_onnx_path(model_type, version, models_dir, explicit=None)`、`supported_model_types`。
- `core/models/__init__.py`
- `core/backends/adapters/`：`base.py`（`AdapterContext` / `AdapterError` / `ModelAdapter` + numpy 工具）、`registry.py`、`rife.py`、`film.py`、`amt.py`、`xvfi.py`、`atm.py`、`momo.py`、`m2m.py`、`multiframe.py`（STMFNet + FLAVR）、`unsupported.py`、`__init__.py`。
- `tests/backends/test_adapters.py`（25 例）、`tests/backends/test_onnx_dispatch.py`（12 例）。

### 修改
- `core/backends/onnx_backend.py`：`SUPPORTED_MODELS` 由 registry 派生；`load_model` 经 `asset_resolver.resolve_onnx_path`；`infer` 走 adapter 流水线并收集**全部**输出；`_detect_fixed_hw` 静态尺寸预检；`only_t05` 守卫。
- `core/backends/tensorrt_rtx_backend.py`：同上；`_read_engine_io` 收集全部 I/O；输出 buffer 设备改用 `ctx.device`（修原硬编码 `"cuda"`）；engine 缓存 key 覆盖 C/H/W；`_is_cuda_device` 校验；引擎 I/O 名不一致告警。
- `core/backends/__init__.py`：注册 `register_builtin_adapters()`。
- `core/task_orchestrator.py:428-434`、`core/task_scheduler.py:407-413`：检查 `load_model` 返回值，失败 fail-fast。
- 参数语义统一（`model_version` ≡ 版本令牌，checkpoint 名为派生物）：`core/model_manager.py`、`core/model_selection.py`、`ui/viewmodels/pipeline_viewmodel.py`、`core/benchmark/benchmark_runner.py:561`。

## 验证证据

- `pytest tests/backends/ -q` → **37 passed**（XPU runtime）。
- 真实资产端到端冒烟（CPUExecutionProvider）：
  - `models/rife_v2/rife_v4.22.onnx`：`input[1,7,H,W]` → `[1,3,H,W]`，输出 `[3,64,64]` 值域≈[0,1]。
  - 传 checkpoint 名 `rife49.pth`：归一化为令牌后一致通过。
  - `models/film/film_net_fp32.onnx`：`img0,img1,timestep` → `[3,256,256]`；64×64 请求返回可操作错误（`fixed_hw=(256,256)`）。
- backends/adapters 无模块级 `import torch`（lazy only）；`py_compile` 全通过。
- TRT 真机引擎未跑（本机无 `tensorrt_rtx` 环境），逻辑经只读审阅确认。

## oracle 复核（已完成并修复）

复核结论初判「需修后发布」，已修：
- **M1** 多帧 `expand_frames` 未做 2→4 复制（原直接报错）→ 已实现 `[a,b]→[a,a,b,b]`，I0..I3 顺序与 PyTorch 一致。
- **M2** FILM/FLAVR 静态 256 资产无预检 → 两后端加 `_detect_fixed_hw` + `infer` 守卫。
- **M3** `get_adapter` 未知类型静默回落 `GenericNCHWAdapter`（静默错帧风险）→ 改为 `raise AdapterError`。
- **M4** `task_scheduler.py` 未检查 `load_model` 返回值 → 已 fail-fast。
- **m1** ATM/MoMo `only_t05` 无消费方，`t≠0.5` 静默错帧 → 两后端 `infer` 守卫。
- **M5** RIFE `supported_versions` 硬编码漂移 → 改为从 `MODEL_ASSET_TABLE` 派生（26 项顺序一致；注意排序须用点分整数元组，不能用 `float`，否则 `4.10 < 4.9`）。

## 已知限制 / 待决策

- **FILM / FLAVR ONNX 资产固定 256×256**：要任意分辨率需重导动态轴。
- **资产覆盖缺口（M5 级，待决策）**：`models/cain/cain.onnx`（动态 6 通道，可适配）、`models/ifrnet/*.onnx`、`models/rife_ensemble/*`、rife lite/heavy 变体存在于磁盘，但 `model_selection` 仍由 .pth checkpoint 驱动，ONNX-only 变体不在可选列表。
- **线程安全**：`_ensure_engine`/`_ensure_context` 惰性缓存无锁，多线程并行 `infer` 共享实例不安全（非回归）。
- `core/model_selection.py:_select_default_checkpoint` 仍硬编码 `rife49.pth`/`film_net_fp32.pt`（Nit）。
