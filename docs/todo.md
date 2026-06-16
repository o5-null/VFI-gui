# VFI-gui 待办事项

> 最后更新: 2026-06-16 | 旧版 `D:\code\VFI\todo.md` 已合并至此

---

## 🔴 高优先级

### 已完成

- [x] 核心架构重构 (2026-05 批次)
  - `core/types.py` — 812 行核心类型定义
  - `core/events.py` — Blinker 事件系统
  - `core/task_orchestrator.py` — 任务编排器
  - `core/task_parser.py` — 任务解析
  - `core/task_scheduler.py` — 并行流式调度
  - `core/subtask_generator.py` — 子任务生成
  - `core/engine_manager.py` + `core/engine_preloader.py` — 引擎管理
  - `core/checkpoint_manager.py` — 断点续传
  - `core/result_validator.py` — 结果验证
  - `core/io/streaming_reader.py` — PyAV 流式读取
  - `core/io/frame_cache.py` — 缓存管理
  - `core/io/ordered_buffer.py` — 乱序重排
  - `core/io/frame_lifecycle.py` — 生命周期
  - `core/config/` — 8 领域配置系统

- [x] UI 层重构 (ViewModel + Controller 模式)
  - `ui/viewmodels/` — 5 个 ViewModel
  - `ui/controllers/` — 3 个 Controller
  - `ui/pages/` — ConfigPage + ProcessPage
  - `ui/app.py` — DI 容器

- [x] 模型实现 (8/8 已完成)
  - RIFE (4.0~4.26) — `core/pytorch_models/vfi_torch/rife/`
  - FILM — `core/pytorch_models/vfi_torch/film/`
  - IFRNet — `core/pytorch_models/vfi_torch/ifrnet/`
  - AMT (s/l/g) — `core/pytorch_models/vfi_torch/amt/`
  - XVFI — `core/pytorch_models/vfi_torch/xvfi/`
  - GMFSS Fortuna — `core/pytorch_models/vfi_torch/gmfss/`
  - ATM-VFI — `core/pytorch_models/vfi_torch/atm/` (仅 t=0.5)
  - MoMo VFI — `core/pytorch_models/vfi_torch/momo/` (仅 t=0.5, DDPM 8-step)

- [x] FILM TorchScript 完整修复 (2026-04-28)
  - 文件: `core/pytorch_models/vfi_torch/film/__init__.py`
  - 添加模型 FP16 转换
  - 添加输入帧 dtype 转换
  - 添加 64 像素对齐填充
  - 修复返回类型兼容 Tensor 和 VFIResult

- [x] 统一插帧模型接口 — `core/pytorch_models/base.py` + `vfi_torch/base.py`
- [x] 模型下载和管理 — `core/model_manager.py` + `core/pytorch_models/model_manager.py`
- [x] 后端自动选择 — `core/backends/backend_factory.py` InferenceStrategySelector
- [x] 多 GPU 支持 — CUDA / ROCm / XPU (device_manager.py)
- [x] 批处理队列 — QueueManager + QueueViewModel
- [x] VFR 支持 — StreamingFramePairReader PTS
- [x] 帧去重 — PreprocessPipeline + DuplicateDetector
- [x] 日志系统 — `core/logger.py` loguru
- [x] 场景检测重写 — `core/preprocess/scene_detect.py`

---

### 待实现



#### 2. Tile 处理实现 (中等优先级)
**问题**: FILM/RIFE 等模型无 tile 处理，4K 视频全帧处理导致 OOM

**影响文件**:
- `core/pytorch_models/vfi_torch/film/__init__.py`
- `core/pytorch_models/vfi_torch/rife/__init__.py`
- `core/pytorch_models/vfi_torch/base.py`

**实现要点**:
```python
def interpolate_tiled(self, frame0, frame1, tile_size=512, overlap=64):
    """Tile 处理减少内存峰值"""
    h, w = frame0.shape[-2:]

    result = torch.zeros_like(frame0)
    weights = torch.zeros(1, 1, h, w, device=frame0.device)

    # Hann 窗口权重用于边缘混合
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # 处理每个 tile
            # 混合权重避免接缝
            pass

    return result / weights
```

---

#### 3. 帧分块处理 (中等优先级)
**问题**: `torch_backend.py` 单次将所有帧移至 GPU，大视频 OOM

**影响文件**:
- `core/backends/torch_backend.py`

**实现要点**:
```python
def _interpolate_frames_chunked(self, frames, multiplier, chunk_size=50):
    """分块处理，避免一次性加载所有帧"""
    n = len(frames)

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_frames = frames[chunk_start:chunk_end].to(device)

        output = self._process_chunk(chunk_frames, multiplier)
        yield output

        del chunk_frames
        clear_cache()
```

---

## 🟡 中优先级

#### 4. TensorRT 推理后端
**影响文件**: `core/backends/tensorrt_backend.py`（新建）

**参考**: VSGAN-tensorrt-docker 的 rife_trt.py

---

#### 5. ONNX Runtime 推理后端
**影响文件**: `core/backends/onnx_backend.py`（新建）

**参考**: vs-mlrt / vsort

---

#### 6. 超分辨率集成
- [ ] RealESRGAN
- [ ] CUGAN (动漫优化)
- [ ] Waifu2x
- [ ] 分块处理 (tiling)
- [ ] 超分模型管理

---

#### 7. 缓存清理优化 (低优先级)
**问题**: `clear_cache_every` 默认为 10 帧，间隔过大

**当前配置**:
- `core/pytorch_models/vfi_torch/base.py`
- 默认值: `clear_cache_every: int = 10`

**建议修改**:
```python
if height > 2000:    # 4K+
    clear_cache_every = 1
elif height > 1000:  # 1080p
    clear_cache_every = 3
else:
    clear_cache_every = 10
```

#### 8. 提前缓存清理 (低优先级)
**影响文件**:
- `core/backends/torch_backend.py`

**修改**:
```python
clear_cache()
frames_chw = frames.permute(0, 3, 1, 2).to(device)
```

---

## 🟢 低优先级

#### 9. 预设配置
- [ ] 预置配置模板（流畅/质量/高性能）
- [ ] 导出/导入配置

#### 10. GPU 监控
**问题**: `TaskViewModel` GPU 信号是占位符，无数据源

**影响文件**:
- `ui/viewmodels/task_viewmodel.py`（GPU 属性）
- `ui/widgets/progress/gpu_monitor.py`（HwMonitorPlaceholder）

---

#### 11. 文档完善
- [ ] 用户使用手册
- [ ] 开发者文档
- [ ] API 文档

#### 12. 测试与质量
- [ ] 单元测试（目前零测试覆盖）
- [ ] 集成测试
- [ ] 性能基准测试
- [ ] CI/CD 配置

#### 13. 额外功能
- [ ] 颜色校正
- [ ] 时序修复 (vs_temporalfix)
- [ ] AMD ROCm 支持验证

---

## 📊 内存优化效果预估

| 方案 | 内存占用下降 | 实现难度 | 状态 |
|------|-------------|----------|------|
| FP16 转换 | 50% | ⭐ | ✅ 已完成 |
| Tile 处理 (512×512) | 75% | ⭐⭐⭐ | ⬜ |
| 分块帧处理 | 90% | ⭐⭐⭐⭐ | ⬜ |
| 缓存清理优化 | 10-20% | ⭐ | ⬜ |
| 组合方案 | 95% | - | ⬜ |

---

## 🔧 其他已知问题

### Intel XPU 特定限制
- 单个张量分配上限 4GB (IPEX Issue #629)
- 需要确保中间张量不超过此限制

### TorchScript 模型警告
```
UserWarning: 'torch.load' received a zip file that looks like a TorchScript archive
```
- 位置: `core/pytorch_models/vfi_torch/utils.py` 第 205 行
- 建议: 使用 `torch.jit.load` 直接加载 TorchScript 模型

### 静态分析已知问题
- `Import "torch" could not be resolved` — pyright 环境无 torch
- `Cannot access attribute "cpu" for class "ndarray"` — Union 窄化局限

---

## 📝 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-16 | 合并 `D:\code\VFI\todo.md`，更新文件路径为重构后位置，添加遗漏项 |
| 2026-06-16 | ATM-VFI + MoMo VFI 模型移植 (84e3b1a)，更新完成状态 |
| 2026-04-28 | 修复 FILM TorchScript FP16 问题 |
| 2026-04-21 | 初始代办创建 |
