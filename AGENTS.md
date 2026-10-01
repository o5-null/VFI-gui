# AGENTS.md — VFI-gui

基于 PyQt6 的高性能多后端视频帧插值 GUI 应用。

## 支持的插帧模型

位于 `core/pytorch_models/vfi_torch/`，通过 `MODEL_REGISTRY` 统一注册。

| 模型 | 路径 | 说明 | 限制 |
|------|------|------|------|
| RIFE | `rife/` | Real-Time Intermediate Flow Estimation | — |
| FILM | `film/` | Frame Interpolation for Large Motion | — |
| IFRNet | `ifrnet/` | Intermediate Flow Regression Network | — |
| AMT | `amt/` | Arbitrary-scale Motion estimation | — |
| XVFI | `xvfi/` | eXtreme Video Frame Interpolation | — |
| GMFSS | `gmfss/` | Animation-dedicated (GMFlow + FusionNet) | — |
| M2M | `m2m/` | Many-to-Many Splatting VFI | ~800ms CPU 224p |
| ATM | `atm/` | Attention-to-Motion VFI | **仅 t=0.5** |
| MoMo | `momo/` | Diffusion-based VFI, DDPM 8-step | **仅 t=0.5** |

> **ATM/MoMo 限制原因**: forward 不接受 timestep 参数，永远输出 t=0.5 居中帧。多帧插值需递归二分，质量逐层下降。

## 推理后端

| 后端 | 路径 | 状态 |
|------|------|------|
| ONNX Runtime (RIFE) | `core/backends/onnx_backend.py` | ✅ 已实现（RIFE ONNX，CUDA EP 默认 / DirectML 可选 / CPU 回退） |
| PyTorch (InProcess) | `core/backends/inprocess_backend.py` | ⚠️ 已弃用（保留为回退） |
| PyTorch (SubProcess) | `core/backends/subprocess_backend.py` | ⚠️ 已弃用（保留为回退） |
| TensorRT-RTX (RIFE) | `core/backends/tensorrt_rtx_backend.py` | ✅ 已实现（原生 `tensorrt_rtx`，ONNX 输入，引擎磁盘缓存 `models/trt_rtx_cache/`） |
| TensorRT (旧) | — | ❌ 未实现 |

> PyTorch/torch 推理路径已弃用（见 `TorchBackend.DEPRECATED`），优先使用 ONNX Runtime 或 TensorRT-RTX；模型路径统一为 `models/rife_v2/rife_<v>.onnx`（回退 `models/rife/`），依赖 `tensorrt-rtx>=1.6.0`（`runtime-requirements-cuda.txt`）。
> ONNX 模型：`models/rife_v2/rife_v4.26.onnx`（来源 vs-mlrt `external-models`，输入 `[1,7,H,W]` RGB [0,1]）。
> 运行时依赖 `onnxruntime-gpu>=1.27`（见 `runtime-requirements-cuda.txt`）；DirectML EP 与 CUDA EP 不能同 venv 共存。

## 硬件加速

| 硬件 | Runtime 环境 |
|------|-------------|
| NVIDIA CUDA | `runtime/cuda/` |
| Intel XPU | `runtime/xpu/` |
| CPU | 默认回退 |

选择流程：`RuntimeManager.auto_select_runtime()` → CUDA → XPU → CPU

## Runtime 环境（uv 虚拟环境）

GPU runtime 由 **uv** 管理，创建自 `pyproject.toml`。系统 Python 和默认 venv **缺少 torch 等依赖**，所有操作必须使用 runtime Python。

### 安装包

```bash
# 方法 1：指定 --python
uv pip install --python D:\code\VFI\runtime\xpu\Scripts\python.exe <package>

# 方法 2：cd 到 runtime 目录（利用 .python-version 自动识别）
cd D:\code\VFI\runtime\xpu
uv pip install <package>
```

### 运行测试/脚本

```bash
# XPU
D:\code\VFI\runtime\xpu\Scripts\python.exe -m pytest tests/
D:\code\VFI\runtime\xpu\Scripts\python.exe scripts/run_inference.py

# CUDA
D:\code\VFI\runtime\cuda\Scripts\python.exe -m pytest tests/
D:\code\VFI\runtime\cuda\Scripts\python.exe scripts/run_inference.py
```

## Scripts（`scripts/`）

| Script | 功能 | 运行方式 |
|--------|------|----------|
| `run_tests.py` | 带覆盖率测试 | `python scripts/run_tests.py` |
| `run_lint.py` | Ruff + Pyright 静态检查 | `python scripts/run_lint.py` |
| `run_coverage.py` | 覆盖率报告 | `python scripts/run_coverage.py` |
| `run_inference.py` | 模型推理冒烟测试 | `python scripts/run_inference.py` |

### run_inference.py

```bash
# 所有注册模型
python scripts/run_inference.py

# 指定模型
python scripts/run_inference.py --models rife,film

# 指定设备
python scripts/run_inference.py --device cpu
```

权重文件位于 `models/`：

| 模型 | 权重路径 | 状态 |
|------|----------|------|
| RIFE | `models/rife/rife47.pth` | ✅ |
| FILM | `models/film/film_net_fp32.pt` | ✅ 1766ms @ 1080p XPU |
| IFRNet | `models/ifrnet/IFRNet_L_Vimeo90K.pth` | ❌ 架构不兼容 |
| AMT | `models/amt/amt-g.pth` | ❌ 输入通道不匹配 |
| GMFSS | `models/gmfss_fortuna/*.pkl` | ❌ 权重与架构冲突 |
| XVFI | `models/xvfi/XVFInet_Vimeo_exp1_latest.pt` | ❌ 张量尺寸不匹配 |
| ATM | `models/atm/atm-vfi-base.pt` | ✅ 3741ms @ 1080p XPU |
| MoMo | `models/momo/momo-base.pth` | ✅ 13371ms @ 1080p XPU |

---

## 文档规范

所有项目文档存放于 `docs/`，按类型分目录。命名与结构必须遵循以下约定。

### 目录结构

```
docs/
├── todo.md                          # 待办事项（唯一，持续维护）
├── DEVELOPMENT.md                   # 主题参考文档（长期有效）
├── EVENT_SYSTEM.md                  # 主题参考文档
├── I18N.md                          # 主题参考文档
├── ARCHITECTURE_REFACTORING.md      # 主题参考文档
├── PROJECT_RELATIONSHIPS.md         # 主题参考文档（项目关系图）
├── VFI_MODELS_REFERENCE.md          # 主题参考文档（VFI 模型对照表）
└── dev-docs/                        # 开发工作日志（按任务归档）
    ├── agent.md                     # 工作日志索引（唯一，按时间顺序追加）
    ├── 20260422-074400-fix-fp16-dtype-mismatch.md
    ├── 20260617-120000-m2m-model-implementation.md
    └── 20260621-120000-code-review-fixes.md
```

### 文档类型与命名约定

| 类型 | 位置 | 命名格式 | 示例 |
|------|------|----------|------|
| **主题参考文档** | `docs/` | `UPPER_SNAKE_CASE.md` | `EVENT_SYSTEM.md`, `I18N.md` |
| **工作日志** | `docs/dev-docs/` | `YYYYMMDD-HHMMSS-<kebab-case-title>.md` | `20260621-120000-code-review-fixes.md` |
| **待办事项** | `docs/` | `todo.md`（唯一） | — |
| **日志索引** | `docs/dev-docs/` | `agent.md`（唯一） | — |

**命名规则：**
- 时间戳采用本地时间（Asia/Hong_Kong），24 小时制
- kebab-case 标题用英文小写，单词连字符分隔，描述任务主题
- 主题文档无日期前缀（长期有效）；工作日志必有日期前缀（一次性记录）

### 主题参考文档（`docs/`）

长期有效的参考文档，描述系统某一方面（事件系统、i18n、架构等）。

**结构要求：**
- 首行 `# <Title>` 标题
- 内容按章节组织，使用 `##` / `###` 分级
- 代码示例用 ` ``` ` 围栏块，标注语言
- 表格用于结构化对比（信号列表、配置项等）

**何时创建：** 引入新的子系统、架构模式或需要长期参考的设计决策时。

**何时更新：** 相关系统变更时同步更新，保持与代码一致。

### 工作日志（`docs/dev-docs/`）

单次任务的执行记录（bug 修复、新功能实现、重构、审查等）。

**文件命名：** `YYYYMMDD-HHMMSS-<kebab-case-title>.md`

**结构要求（章节顺序固定）：**
```markdown
# <标题> — <日期>

**时间:** YYYY-MM-DD HH:MM:SS
**状态:** ✅ 已完成 / 🔄 进行中 / ⚠️ 已知问题

## 问题描述 / 任务概述
<做什么、为什么>

## 根因分析 / 执行过程
<分析步骤、参考对比、问题定位>

## 解决方案 / 修复方案
<修改前后的代码对比，方案选择理由>

## 验证
<验证步骤、测试结果>

## 相关文件
<涉及的文件列表>

## 经验总结
<可复用的经验、教训>
```

**约束：**
- 每个文件只记录一个任务，不混合多个无关任务
- 代码示例必须包含修改前/修改后对比
- 文件路径用反引号包裹（如 `core/types.py`）
- 行号引用格式：`文件路径:行号`（如 `subprocess_backend.py:438`）

### 工作日志索引（`docs/dev-docs/agent.md`）

按时间顺序追加的工作日志摘要。

**格式：**
```markdown
## YYYY-MM-DD HH:MM — <任务标题>

### 任务概述
<一句话描述>

### 执行过程
1. <步骤>

### 审查/修复结果
<结论>

### 输出文件
- `<详细文档路径>`

### 经验总结
1. <可复用经验>

---
```

**约束：**
- 新条目追加在 `---` 分隔线之前（即文件末尾的 `*Last updated:*` 行之前）
- 每个条目必有 `### 输出文件` 指向详细文档
- 更新 `*Last updated: YYYY-MM-DD HH:MM*` 时间戳
- 保持简洁，详细内容放在对应的工作日志文件中

### 待办事项（`docs/todo.md`）

持续维护的待办清单。

**结构要求：**
- 按 `🔴 高优先级` / `🟡 中优先级` / `🟢 低优先级` 分组
- 每组内分 `### 已完成` 和 `### 待实现`
- 待办项格式：`- [ ] <描述>` / `- [x] <描述>`
- 长任务含 `**影响文件**` 和 `**实现要点**` 子项
- 底部 `## 📝 变更日志` 表格记录重要变更

**约束：**
- 整个项目只有一个 `todo.md`，不分散到各模块
- 完成的待办移到对应组的 `### 已完成` 下，不删除
- 引用外部文档时用 `> 详见 <路径>` 引用块

### 现有文档迁移

以下历史文档不符合当前命名规范，但暂不重命名以保持引用稳定。新增文档必须遵循本规范。

| 文件 | 问题 | 处理方式 |
|------|------|----------|
| `docs/REFACTOR_CORE_20260428.md` | 日期后缀但不在 dev-docs/，应为工作日志 | 内容性质为重构记录，保留原位；类似新文档应放 dev-docs/ |
| `docs/torch_backend.md` | snake_case 命名 | 保留原位；类似新文档应用 UPPER_SNAKE_CASE |
| `docs/ARCHITECTURE_REFACTORING.md` | 无日期但内容为重构记录 | 保留原位；按主题文档对待 |

---

## 分层 AGENTS

本项目的 AGENTS.md 按模块分层：

| 层级 | 文件 | 覆盖范围 |
|------|------|----------|
| 项目层 | `AGENTS.md`（本文件） | 项目架构、后端、硬件、脚本、文档规范、CI |
| **模型层** | `core/pytorch_models/vfi_torch/AGENTS.md` | 模型接口规范、下载管理、去重约束 |
| 配置层 | `core/config/AGENTS.md` | ConfigFacade + 8 域配置模式 |
| IO 层 | `core/io/AGENTS.md` | 帧读写、乱序重排、缓存策略 |
| 后端层 | `core/backends/AGENTS.md` | 推理后端抽象 + 工厂选择 |
| UI 层 | `ui/AGENTS.md` | ViewModel/Controller 双层架构 |
| 预处理层 | `core/preprocess/AGENTS.md` | 场景检测 + 重复帧管线 |
