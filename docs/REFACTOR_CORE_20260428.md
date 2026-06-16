# Core 目录重组记录

**日期**: 2026-04-28  
**任务**: 整理 VFI-gui/core 目录结构，解决命名不规范和功能重叠问题

---

## 变更摘要

### 删除文件

| 文件 | 原因 |
|------|------|
| `core/pipeline.py` | 依赖已移除的 VapourSynth，工作流已由 workers/ 处理 |
| `core/processor.py` | 已废弃，被 `task_orchestrator.py` 替代 |
| `core/torch_processor.py` | 已废弃，被 `task_orchestrator.py` 替代 |
| `core/config.py` | 兼容层，`config/` 包已导出所有内容 |
| `core/backends/torch_backend_refactored.py` | 重复代码（未完成的重构版本） |

### 重命名

| 原名称 | 新名称 | 原因 |
|--------|--------|------|
| `core/models.py` | `core/model_manager.py` | 名称过于泛化，包含核心 ModelManager 类 |
| `core/torch_backend/` | `core/pytorch_models/` | 与 `backends/torch_backend.py` 文件命名冲突 |

### 移动

| 原位置 | 新位置 | 原因 |
|--------|--------|------|
| `core/graphify-out/` | 项目根目录 `graphify-out/` | 工具生成产物不应在 core/ |

---

## 导入更新

以下文件的导入语句已更新：

| 文件 | 变更 |
|------|------|
| `core/__init__.py` | 移除 Processor, PipelineBuilder 导出；更新 `models` → `model_manager` |
| `core/model_selection.py` | `from core.models` → `from core.model_manager` |
| `ui/widgets/model_panel.py` | `from core.models` → `from core.model_manager` |
| `ui/widgets/model_manager_panel.py` | `from core.models` → `from core.model_manager` |
| `ui/viewmodels/model_viewmodel.py` | `from core.models` → `from core.model_manager` |
| `core/backends/torch_backend.py` | `from ..torch_backend` → `from ..pytorch_models` |

---

## 最终目录结构

```
core/
├── backends/               # 后端抽象层
│   ├── __init__.py         # BaseBackend, BackendFactory, BackendType
│   ├── torch_backend.py    # TorchBackend 实现
│   └── inference_thread_pool.py
├── benchmark/              # 设备检测和性能测试
├── config/                 # 配置模块（领域配置类）
│   ├── config_facade.py    # ConfigFacade 统一访问入口
│   ├── pipeline_config.py
│   ├── ui_config.py
│   ├── runtime_config.py
│   └── ...
├── interfaces/             # 抽象接口 (IModelManager, ICodecManager)
├── io/                     # 帧数据读写、序列化
├── pytorch_models/         # PyTorch 模型实现（原 torch_backend/）
│   ├── vfi_torch/          # RIFE, FILM, AMT, IFRNet, XVFI, EISAI
│   ├── base.py
│   ├── model_manager.py
│   └── frame_processor.py
├── utils/                  # 工具函数
├── workers/                # 后台工作线程（下载、处理）
├── codec_manager.py        # 编解码管理
├── config_provider.py      # 配置单例访问 (get_config, set_config)
├── constants.py            # 应用常量
├── dependency_manager.py   # 外部依赖检测 (FFmpeg)
├── device_manager.py       # 设备检测 (CUDA, ROCm, XPU, CPU)
├── device_type.py          # DeviceType 枚举
├── events.py               # Blinker 事件系统
├── i18n.py                 # 国际化
├── logger.py               # Loguru 日志配置
├── model_inspector.py      # 模型文件深度检测
├── model_manager.py        # 模型管理核心（原 models.py）
├── model_selection.py      # 模型选择状态管理
├── network.py              # 网络工具（代理、下载）
├── paths.py                # 路径管理
├── queue_manager.py        # 批处理队列管理
├── runtime_manager.py      # 运行时环境激活（venv）
├── task_orchestrator.py    # 任务编排（主处理入口）
└── __init__.py             # 模块导出
```

---

## 设计决策

### 1. 命名冲突解决

**问题**: 文件名与目录同名导致导入混淆
- `config.py` ↔ `config/` 目录
- `backends/torch_backend.py` ↔ `torch_backend/` 目录

**方案**: 
- 删除兼容层文件（config.py）
- 重命名目录使用更准确的描述性名称（torch_backend → pytorch_models）

### 2. 废弃文件清理

**问题**: 废弃文件保留导致维护负担和混淆
- processor.py, torch_processor.py 标记废弃但仍存在
- torch_backend_refactored.py 是未完成的重构副本

**方案**: 直接删除而非保留 deprecated/ 目录
- 项目未发布，无向后兼容需求
- 减少代码库混乱

### 3. VapourSynth 依赖移除

**背景**: 用户已删除 vsgan/ 目录，项目不再使用 VapourSynth
- pipeline.py 依赖 `import vapoursynth as vs`
- 工作流改由 workers/ 处理

**方案**: 直接删除 pipeline.py

---

## 后续建议

1. **device_manager.py + runtime_manager.py**: 
   - 当前职责边界不够清晰（DeviceManager 检测硬件，RuntimeManager 激活 venv）
   - 建议添加文档注释明确职责分离

2. **config_provider.py**: 
   - 仅为单例访问模式，可以考虑移入 config/ 包或合并到 config_facade.py

3. **model_inspector.py + model_manager.py**: 
   - 功能相关但职责不同（检测 vs 管理）
   - 当前命名清晰，保持分离

---

## 参考

- AIM Memory: `CoreReorg_20260428` (`.aim/memory.jsonl`)
- 相关文档: `docs/torch_backend.md` (已同步更新)