# ONNX / TensorRT-RTX 后端多模型类型支持 — 集成契约

> 本文档是并行实施的**接口契约**（frozen interface）。所有改动必须严格对齐此处签名与职责划分，避免多 writer 冲突。
> 状态：🔄 实施中

## 1. 目标与范围

让 `OnnxBackend`（`core/backends/onnx_backend.py`）与 `TensorRTRTXBackend`
（`core/backends/tensorrt_rtx_backend.py`）通过**按 `model_type` 分派的适配层**支持多种模型类型，
不再只有 RIFE。

**在本范围内**：后端侧加载/推理/后处理通用化。
**不在本范围内**：ONNX 导出（假设非 RIFE 的 `.onnx` 文件后续存在）。

## 2. 支持集（权威定义）

`MODEL_DEFINITIONS`（`core/model_manager.py:105`）有 11 项：rife/film/amt/m2m/flavr/stmfnet/cain/atm/momo/sepconv/xvfi。
但 `MODEL_REGISTRY`（`core/pytorch_models/vfi_torch/__init__.py:39`）只注册 9 个。

| 状态 | model_type |
|---|---|
| ✅ 可适配（9） | rife, film, amt, xvfi, atm, momo, m2m, stmfnet, flavr |
| ❌ 明确拒绝（2，无 PyTorch 实现） | cain, sepconv |

后端 `SUPPORTED_MODELS` 必须由 adapter registry 派生（见 §5），**cain/sepconv 不得出现**。

## 3. 文件所有权（并行 writer 边界）

| 所有者 | 文件 | 动作 |
|---|---|---|
| **A1** | `core/models/asset_resolver.py` | 新增 |
| **A1** | `core/backends/adapters/__init__.py` | 新增 |
| **A1** | `core/backends/adapters/base.py` | 新增 |
| **A1** | `core/backends/adapters/registry.py` | 新增 |
| **A1** | `core/backends/adapters/rife.py`, `film.py`, `amt.py`, `xvfi.py`, `atm.py`, `momo.py`, `m2m.py`, `multiframe.py`, `unsupported.py` | 新增 |
| **B** | `core/model_manager.py` | 修改（表迁移） |
| **B** | `core/model_selection.py` | 修改 |
| **B** | `ui/viewmodels/pipeline_viewmodel.py` | 修改 |
| **C** | `core/backends/onnx_backend.py` | 修改 |
| **C** | `core/backends/tensorrt_rtx_backend.py` | 修改 |
| **C** | `core/backends/__init__.py` | 修改（注册 adapters） |
| **C** | `core/task_orchestrator.py` | 修改（检查 load_model 返回值） |

> 同一文件不得被两个所有者修改。A1 与 B 无文件冲突；C 依赖 A1 的接口。

## 4. `core/models/asset_resolver.py`（A1）

单一真相源：版本令牌 ↔ checkpoint ↔ ONNX 相对路径。

```python
# model_type -> version_token -> {"checkpoint": Optional[str], "onnx": Optional[str]}
# checkpoint: 相对 core 模型目录的权重文件名（models/<model_type>/<checkpoint>）
# onnx:       相对 models_dir 的 ONNX 路径（如 "rife_v2/rife_v4.22.onnx"）
MODEL_ASSET_TABLE: Dict[str, Dict[str, Dict[str, Optional[str]]]]

DEFAULT_VERSIONS: Dict[str, str]   # 合并自 model_manager.DEFAULT_VERSIONS

def checkpoint_to_version(model_type: str, checkpoint: str) -> Optional[str]: ...
def version_to_checkpoint(model_type: str, version: str) -> Optional[str]: ...
def resolve_checkpoint_path(model_type: str, version: str, models_dir: str) -> Optional[Path]: ...
def resolve_onnx_path(
    model_type: str,
    version: str,
    models_dir: str,
    explicit: Optional[str] = None,   # onnx_path / checkpoint_path 覆盖
) -> Optional[Path]:
    """version 可为版本令牌或 checkpoint 名（先经 checkpoint_to_version 归一化）。
    显式路径优先。返回候选路径（即使不存在，便于日志）；无法解析返回 None。"""
def supported_model_types() -> List[str]: ...      # 有适配实现的类型（不含 cain/sepconv）
```

**必须**：A1 需先实际检查 `models/` 目录，用真实存在的 ONNX 文件名填充 `onnx` 字段
（已知：`models/rife_v2/rife_v*.onnx`、`models/film/film_net_fp32.onnx`）；不存在资产的类型 `onnx` 填 `None`。

## 5. Adapters 包（A1）

### `adapters/base.py`

```python
@dataclass
class AdapterContext:
    input_names: List[str]
    output_names: List[str]
    src_hw: Tuple[int, int]          # 原始 (H, W)
    pad_hw: Tuple[int, int]          # 对齐后 (H, W)
    in_channels: Optional[int] = None
    device: str = "cuda:0"
    extra: Dict[str, Any] = field(default_factory=dict)

class AdapterError(Exception): ...

class ModelAdapter(ABC):
    model_type: str = ""
    min_frames: int = 2                       # 2=双帧; 4=STMFNet/FLAVR
    supported_versions: List[str] = ["*"]

    @abstractmethod
    def align_shape(self, h: int, w: int) -> Tuple[int, int]: ...

    def expand_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """默认恒等；多帧模型在此把 2 帧展开为 N 帧。"""
        return list(frames)

    @abstractmethod
    def build_feeds(
        self, frames: List[np.ndarray], timestep: float, ctx: AdapterContext
    ) -> Dict[str, np.ndarray]:
        """返回 {input_name: ndarray}，shape 均为对齐后 [.., padH, padW]。"""

    def output_names(self, ctx: AdapterContext) -> List[str]:
        return list(ctx.output_names)

    @abstractmethod
    def postprocess(self, outputs: List[np.ndarray], ctx: AdapterContext) -> np.ndarray:
        """把原始输出转成 [3, src_h, src_w] float32（含 unpad）。"""
```

### `adapters/registry.py`

```python
ADAPTER_REGISTRY: Dict[str, ModelAdapter]

def register_adapter(adapter: ModelAdapter) -> None: ...
def get_adapter(model_type: str) -> ModelAdapter:
    """未知类型直接 raise AdapterError（不得静默回落，见 §9）；cain/sepconv 返回 UnsupportedAdapter。"""
def supported_models() -> Dict[str, List[str]]:   # 供后端 SUPPORTED_MODELS 派生
def register_builtin_adapters() -> None: ...
```

### 各适配器要点（基于 `core/pytorch_models/vfi_torch/<type>/__init__.py` 语义）

| adapter | align | min_frames | 关键逻辑 |
|---|---|---|---|
| `rife` | 恒等 | 2 | `build_feeds` 复用 `pack_rife_input`（`rife_input.py:30`），C=7/11；透传 `scale/fastmode/ensemble` |
| `film` | 64 | 2 | 独立 timestep feed（参照 `film/__init__.py:238 make_timestep_tensor`）；postprocess unpad（`:245`）。**实测 `film_net_fp32.onnx` 输入为静态 256×256**（由后端 `_fixed_hw` 预检，非 256 明确失败） |
| `amt` | 16 | 2 | embt feed；`scale` 参数；输出取 imgt_pred（`:296-304`） |
| `xvfi` | 128 | 2 | 5D `[B,C,T,H,W]`；`t_value` |
| `atm` | 64 | 2 | 忽略 timestep，固定 t=0.5；记 `only_t05` |
| `momo` | 32 | 2 | 忽略 timestep；5D；标记实验性 |
| `m2m` | 16 | 2 | 任意 timestep |
| `multiframe` | 16 | 4 | `expand_frames([a,b]) -> [a,a,b,b]`（STMFNet 四输入 I0..I3）；FLAVR 单输入 `[1,12,H,W]` 堆叠、按 timestep 选输出帧（`flavr/__init__.py:114-118`）。**实测 `FLAVR_{2x,4x,8x}.onnx` 亦为静态 256×256** |
| `unsupported` | 恒等 | 2 | cain/sepconv：`build_feeds` 抛 `AdapterError` |

> 若真实 ONNX 契约与上表推断不符（尤其 film/amt 的输入数量与名称），以**实际资产**为准并在适配器 docstring 记录。

### `adapters/__init__.py`

导出 `register_builtin_adapters`、`get_adapter`、`supported_models`、`ModelAdapter`、`AdapterContext`、`AdapterError`。

## 6. 两个后端改造（C）

**共享**：执行前统一走 adapter 流水线。

```python
adapter = get_adapter(model_config["model_type"])
frames  = adapter.expand_frames([frame0, frame1])
pad_hw  = adapter.align_shape(H, W)
ctx     = AdapterContext(input_names=..., output_names=..., src_hw=(H,W),
                         pad_hw=pad_hw, in_channels=..., device=config.get_device(),
                         extra={...})
feeds   = adapter.build_feeds(frames, timestep, ctx)
# ONNX: session.run(adapter.output_names(ctx), feeds)
# TRT : 逐 name set_tensor_address + N 输出 buffer
out     = adapter.postprocess(raw_outputs, ctx)   # [3, H, W]
```

### `onnx_backend.py`
- `SUPPORTED_MODELS = supported_models()`（从 `adapters.registry` 派生，模块级或类属性）。
- `load_model`：先 `is_model_supported`（`base_backend.py:150`）校验，不支持即 `logger.error` + `return False`；用 `asset_resolver.resolve_onnx_path` 解析路径，删除 `_resolve_model_path`。
- 泛化 N 输入/N 输出：从 `session.get_inputs()/get_outputs()` 读全部名字，写入 `AdapterContext`；删除固定 `_input_name="input"` 与 `_in_channels or 7`。
- `infer`/`infer_batch` 走 §6 流水线。

### `tensorrt_rtx_backend.py`
- `SUPPORTED_MODELS = supported_models()`；`load_model` 增加校验；删除 `_resolve_model_path`（用 `asset_resolver`）。
- `_read_engine_io`（`:296`）改为收集**全部**输入/输出名（List）。
- `_ensure_context`（`:311`）支持 N 输入/N 输出 buffer；输出 buffer device 用 `ctx.device`（修复 `:336` 硬编码 `"cuda"`）。
- `_ensure_engine` 缓存 key 用对齐后 shape（`adapter.align_shape`）。
- `infer`/`infer_batch` 走 §6 流水线。

### `core/backends/__init__.py`
- `_register_builtin_backends()`（`:56`）内或之后调用 `register_builtin_adapters()`。

### `core/task_orchestrator.py`
- `:429-430` `load_model(...)` 返回值必须检查：失败则记录错误并让任务失败（不静默继续）。同样检查 `core/task_scheduler.py:408`。

## 7. 参数语义统一（B）

现状 bug：UI 把 checkpoint 名写进 `model_version`（`ui/viewmodels/pipeline_viewmodel.py:143-144,388,461`），
而 ONNX/TRT 后端按版本拼 `rife_v<ver>.onnx`。

**规范**：`model_version` ≡ **版本令牌**（`"4.22"`/`"fp32"`/`"s"`）；checkpoint 名是派生物。

- `core/model_manager.py`：`CHECKPOINT_VERSION_MAP`（:264）/`DEFAULT_VERSIONS`（:280）改为从 `asset_resolver.MODEL_ASSET_TABLE` 派生或直接 re-export。
- `core/model_selection.py`：`_version_to_checkpoint`（:240）/`_checkpoint_to_version`（:271）改为查 `asset_resolver`；`get_selection` 返回版本令牌。
- `ui/viewmodels/pipeline_viewmodel.py`：`:143-144` 不再用 `checkpoint_name` 覆盖版本；`to_pipeline_config()` 的 `model_version` 输出版本令牌。建议把 `_checkpoint` 语义改为版本令牌（可保留字段名以缩小 diff，但注释说明）。

## 8. 验证策略

1. **adapter 纯 numpy 单测**（不依赖 onnxruntime/torch）：`tests/backends/test_adapters.py`
   - `align_shape`：FILM (65,65)→(128,128)；AMT (17,17)→(32,32)；RIFE 恒等。
   - `expand_frames`：[a,b]→[a,a,b,b] 元素同一性。
   - `build_feeds`：RIFE C=7/C=11 与 `pack_rife_input` 基准一致；feed name/shape/dtype。
   - `postprocess`：pad→unpad 往返 shape == 原 (h,w)；FLAVR idx 边界（t=0/0.5/1）。
2. **后端分派测**：`tests/backends/test_onnx_dispatch.py` 用 monkeypatch 假 session 验证 load_model→infer 走对 adapter，无需 .onnx。
3. **真实资产冒烟**（已执行，XPU runtime / CPUExecutionProvider）：`models/rife_v2/rife_v4.22.onnx` 输入 `input[1,7,H,W]`→输出 `[1,3,H,W]`，实数 [3,64,64] 值域≈[0,1]；传 checkpoint 名 `rife49.pth` 归一化为令牌后一致通过；`models/film/film_net_fp32.onnx` 输入 `img0,img1,timestep`→`[3,256,256]`（静态 256）。TRT 真机引擎未跑（无 tensorrt_rtx 环境）。
4. **参数语义回归**：`to_pipeline_config()` 的 `model_version` 恒为版本令牌；`resolve_onnx_path` 对 checkpoint 名与版本串输入命中同一文件。

## 9. 已知限制（须在代码中显式处理，不得静默）

- **cain/sepconv**：无 PyTorch 参考实现 → `UnsupportedAdapter` 明确拒绝（显式报错，非遗漏），不进 SUPPORTED_MODELS。注：`models/cain/cain.onnx` 实际存在（动态 `input[1,6,h,w]`→`frame[?,3,h,w]`），若后续要支持需补 adapter 与支持集修订。
- **静态空间维资产**：`models/film/film_net_fp32.onnx` 与 `models/flavr/FLAVR_{2x,4x,8x}.onnx` 输入固定 256×256。两后端 `load_model` 经 `_detect_fixed_hw` 记录，`infer` 对非该尺寸返回明确失败（不缩放、不填充）；要任意分辨率须重导动态轴。
- **未知 model_type**：`registry.get_adapter` 必须 `raise AdapterError`，禁止回落 GenericNCHWAdapter 造成静默错帧（GenericNCHWAdapter 仅作显式工具，不作自动回落）。
- **ATM/MoMo**：仅 t=0.5 → 适配器忽略 timestep 并记录 `ctx.extra["only_t05"]`；**消费方**为两后端 `infer`（`build_feeds` 后、执行前）：`t≠0.5` 时返回明确失败，不得静默返回 0.5 帧。
- **STMFNet/FLAVR**：多帧 → Phase 1 用"2→4 复制"（与现有 PyTorch `interpolate` 行为一致）；Phase 2 才考虑扩 `InferenceRequest.context_frames`。
- **MoMo**：DDPM 多步 + 随机性 → 标为实验性。
- **动态 shape**：TRT 必须静态 profile，用对齐后 shape；换分辨率会重建 engine。
