# Agent 工作日志

## 2026-04-22 07:44 - FP16 dtype 不匹配错误修复

### 任务概述
修复 VFI 模型在 FP16 推理时出现的 `RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same` 错误。

### 执行过程

1. **问题分析**
   - 错误信息表明模型权重是 float16 (Half)，但输入张量是 float32
   - 追踪代码路径：`processor.py` -> `frame_processor.py` -> `rife/__init__.py`

2. **参考对比**
   - 对比 `ComfyUI-Frame-Interpolation/vfi_utils.py` 的实现
   - 发现 ComfyUI 在数据流转中始终保持 dtype 一致性

3. **问题定位**
   - 文件：`core/torch_backend/vfi_torch/rife/__init__.py`
   - 位置：第 231 行
   - 根因：`timestep` 张量创建时未指定 dtype，使用了默认的 float32

4. **修复实施**
   - 添加 `dtype=img0.dtype` 参数
   - 确保 timestep 与输入帧的 dtype 一致

### 修复结果

✅ 成功修复，timestep 张量现在会自动匹配输入帧的 dtype

### 输出文件

- 修复文档：`docs/dev-docs/20260422-074400-fix-fp16-dtype-mismatch.md`

### 经验总结

1. **PyTorch dtype 一致性**：在混合精度推理时，所有参与运算的张量必须保持相同的 dtype
2. **参考成熟实现**：ComfyUI-Frame-Interpolation 的实现提供了正确的 dtype 处理模式
3. **快速定位技巧**：从错误堆栈追踪到具体模型实现，对比参考项目找差异

---

## 2026-06-21 12:00 - 代码审查与修复文档

### 任务概述
使用 code-review-skill 对 VFI-gui 项目进行系统性代码审查，覆盖 Python 代码质量、架构设计、Qt/线程、错误处理与安全四个维度。

### 执行过程

1. **加载审查技能** — 读取 code-review-skill 的 SKILL.md 及 8 个参考指南（python/qt/universal/architecture/performance/security/common-bugs/checklist）
2. **并行扫描** — 文件体积 / Python 反模式 / 架构 / Qt 线程 / 安全（因 subagent 不可用，改用直接工具）
3. **问题汇总** — 🔴 blocking 1 项 / 🟡 important 6 项 / 🟢 nit 4 项
4. **文档撰写** — 遵循 dev-docs 约定创建修复文档

### 审查结果

🔄 Request Changes — 项目整体架构清晰、无重大安全漏洞，主要问题集中在代码复用、运行时校验、日志规范和文件体积方面。

### 输出文件

- 修复文档：`docs/dev-docs/20260621-120000-code-review-fixes.md`

### 经验总结

1. **assert 不适合运行时校验** — `python -O` 模式下会被移除，应替换为显式 `raise`
2. **核心库代码必须用 logger** — print 无法被日志系统捕获、无法分级、无法轮转
3. **DRY 违规常发生在 UI 层** — UI 重新实现 core 已有逻辑时需警惕
4. **UI 线程禁止阻塞** — `wait()`/`sleep()`/同步 IO 都会冻结事件循环
5. **文档与代码状态需同步** — 标记"待废弃"的代码若仍在使用会误导维护者

---

## 2026-06-21 12:30 — 工作区根目录文档清洗

### 任务概述
清理 `D:\code\VFI` 根目录散落的文档，统一文档存储规范。

### 执行过程

1. **调研现状** — 扫描根目录文档分布，识别 9 类问题（重复目录、过时文档、散落文件等）
2. **确认方案** — 通过 question 工具确认 5 个关键决策
3. **执行清理**:
   - 合并 `重构/` 17 个文件为 `重构记录.md`（8491 行）
   - 过时文档移到 `archive/`（项目架构设计.md、模型路径管理重构.md）
   - `docs/项目架构关系.md` 移到 `VFI-gui/docs/PROJECT_RELATIONSHIPS.md`
   - 删除 `.sisyphus/`（与 `.omo/` 重复）
   - 删除 `安装引导.txt`
   - 删除空 `docs/` 目录
4. **更新规范** — 根目录 `AGENTS.md` 添加工作区结构和文档规范章节

### 清理结果

✅ 根目录从 39 个条目精简为结构化布局，`重构/` 从 18 文件→1 文件

### 输出文件

- 更新 `D:\code\VFI\AGENTS.md`（工作区级规范 + 清理记录）
- 更新 `D:\code\VFI\.graphifyignore`（添加 `archive/` 排除）

### 经验总结

1. **定期清理归档** — 重构完成后应及时合并记录，避免文件碎片化
2. **重复目录早识别** — `.sisyphus/` 与 `.omo/` 完全重复，应统一为单一工作目录
3. **文档分层存放** — 工作区级规范在根目录，项目文档在 `VFI-gui/docs/`，历史归档在 `archive/`

---

## 2026-10-01 18:15 — ONNX Runtime 推理引擎实现

### 任务概述
按路线变更弃用 CUDA/torch 推理引擎，新增 ONNX Runtime 推理引擎（RIFE）并测试；tensorrt-rtx 留待第二阶段。

### 执行过程

1. **侦察** — 摸清后端抽象/工厂、调用链、帧值域（float32 RGB [0,1]）、vs-mlrt RIFE ONNX 契约（[1,C,H,W]，C=7 或 11）
2. **实现** — 新建 `core/backends/onnx_backend.py`（`OnnxBackend`）+ 工厂注册；`TorchBackend.DEPRECATED`
3. **模型** — 下载 vs-mlrt `external-models` 的 `rife_v4.26.7z`，落盘 `models/rife_v2/rife_v4.26.onnx`（7ch）
4. **EP** — 默认 CUDA EP、可选 DirectML、CPU 兜底；`runtime/cuda` 改装 `onnxruntime-gpu==1.30.0`（cu13）
5. **测试** — `scripts/test_onnx_backend.py` 冒烟，CPU vs CUDA 对比

### 结果

✅ 全部硬检查通过。CUDA EP 平均 11.09 ms，较 CPU（~44.6–55 ms）提速约 4–5×，identity MAE 与 CPU 一致（0.00247）。

### 输出文件

- 引擎：`core/backends/onnx_backend.py`
- 测试：`scripts/test_onnx_backend.py`
- 工作日志：`docs/dev-docs/20261001-181500-onnx-inference-backend.md`

### 经验总结

1. **ONNX EP 互斥** — `onnxruntime` / `-gpu` / `-directml` 共用同一模块，CUDA 与 DirectML 不能同 venv 共存
2. **值为 [0,1] 直喂** — vs-mlrt RIFE ONNX 不归一化，v2 为 7 通道 `[A,B,t]`
3. **CUDA EP 无 provider 级 fp16 开关** — fp16 只能模型级转换，仅 TRT EP 有 `trt_fp16_enable`

---

## 2026-10-01 18:30 — TensorRT-RTX 推理引擎实现

### 任务概述
实现第二个推理引擎 tensorrt-rtx（原生 `tensorrt_rtx` API），与 ONNX Runtime 并列为两大加速引擎。

### 执行过程

1. **侦察** — 确认环境（RTX 3080 / CUDA 13 / Py3.12），探针跑通 ONNX→engine→执行全流程
2. **选型** — 原生 `tensorrt_rtx`（与 `onnxruntime-gpu` 无模块冲突），弃用 ORT 插件 EP 路线
3. **实现** — 新建 `core/backends/tensorrt_rtx_backend.py`（惰性构建 + 磁盘引擎缓存）+ 共享打包 `core/backends/rife_input.py`；新增 `BackendType.TENSORRT_RTX`
4. **测试** — `scripts/test_tensorrt_rtx_backend.py` 冒烟 + ONNX 后端回归

### 结果

✅ 全部硬检查通过。identity MAE=0.00246；引擎构建 0.8s、序列化 27.9MB、磁盘缓存命中；纯内核 2.2–2.8ms，end-to-end ~11.6ms（对照 ORT CUDA EP ~12.6ms）。修复了「每次推理重建 execution context 导致 2.7s/帧」的缺陷。

### 输出文件

- 引擎：`core/backends/tensorrt_rtx_backend.py`、`core/backends/rife_input.py`
- 测试：`scripts/test_tensorrt_rtx_backend.py`
- 工作日志：`docs/dev-docs/20261001-183000-tensorrt-rtx-backend.md`

### 经验总结

1. **execution context 极贵** — `create_execution_context()` 首调数秒，必须按 shape 缓存复用
2. **选型看模块冲突** — 原生 `tensorrt_rtx` 与 `onnxruntime-gpu` 可共存；ORT 插件 EP 会冲突
3. **静态 profile 足够** — 每分辨率一个 engine，构建仅 0.8s

---

## 2026-10-01 18:35 — RIFE ONNX 全模型双后端冒烟测试

### 任务概述

补齐 5 个 RIFE ONNX 权重，对 ONNX Runtime 与 TensorRT-RTX 两个后端跑全部 6 个版本的冒烟测试。

### 执行过程

1. 从 vs-mlrt `external-models` release 下载 4 个 7z，提取 `rife_v2/rife_v4.{0,6,7,17,22,26}.onnx`（4.0/4.6 仅存在于大包 `rife_v2_v4.7z`）
2. 用 `runtime/cuda` python 分别对两后端 × 6 版本运行 `scripts/test_{onnx,tensorrt_rtx}_backend.py --runs 5`

### 审查/修复结果

- ONNX Runtime：6/6 全通过（identity MAE 0.0024–0.0164，时延 8.1–13.1ms）
- TensorRT-RTX：4/6 通过；**v4.0 MAE=0.11059、v4.6 MAE=0.14724 超阈值 0.02 失败**（精确可复现）
- 同文件在 ONNX 通过、TRT 失败 → TRT 默认 TF32 路径放大误差；两模型本身对精度更敏感

### 输出文件

- `docs/dev-docs/20261001-183500-rife-onnx-all-models-smoke-test.md`
- `models/rife_v2/rife_v4.{0,6,7,17,22,26}.onnx`

### 经验总结

1. ONNX 权重无自动下载，需手动从 vs-mlrt release 取；4.0/4.6 只能从大包提取
2. 必须显式用 runtime python（脚本的 runtime 自动探测路径错位失效）
3. TRT-RTX 默认 TF32，对数值敏感模型可能超出 identity 阈值

---

## 2026-10-01 19:00 - ONNX 模型资产扩充（RIFE 全版本 + IFRNet/CAIN/FILM/FLAVR）

### 任务概述
在 RIFE 双后端冒烟之后：补齐更多 RIFE 版本（lite/heavy/ensemble），拉取其他模型已公开 ONNX（IFRNet/CAIN），并为无 ONNX 的可导出模型（FILM/FLAVR）转换 ONNX。范围约束：不扩展后端，仅准备资产。

### 执行过程

1. 从 vs-mlrt `external-models` release 拉取 `rife_v4.{7..26}.7z` + 大包 `rife_v2_v4.7z` + `rife_ensemble_v1.7z`，提取并落盘
2. 下载上游 ONNX：IFRNet（MoonApp-IFRNet-ONNX）、CAIN（MoonApp-CAIN-ONNX v1）
3. 后台并行导出：FILM（重建架构 + 加载 TorchScript state_dict）、FLAVR（UNet3D3D，2x/4x/8x）
4. 新建 `scripts/validate_onnx_assets.py` 统一校验

### 结果

- `models/rife_v2`：41/41 通过（identity MAE 0.0016–0.0189）
- `models/rife_ensemble`：3/6 通过（11ch，4.2/4.3/4.4 恒等残余 0.03–0.08）
- `models/ifrnet` 2/2、`models/cain` 1/1、`models/film` 1/1、`models/flavr` 3/3 结构校验 OK
- FILM ONNX t=0.5 与原 TorchScript max diff=0.0；FLAVR vs PyTorch MAE < 1e-5

### 输出文件

- `docs/dev-docs/20261001-190000-onnx-asset-expansion-film-flavr.md`
- `scripts/validate_onnx_assets.py`、`scripts/export_film_onnx.py`、`scripts/export_flavr_onnx.py`
- `models/{rife_v2,rife_ensemble,ifrnet,cain,film,flavr}/` 下的 ONNX 资产

### 经验总结

1. 校验 11ch RIFE 必须复用 `pack_rife_input`，手工 meshgrid 布局错误会把 MAE 从 0.008 假性放大到 0.27
2. FILM 官方 TorchScript 硬编码 t=0.5，直导 ONNX 不可行；"重建架构 + 加载 state_dict" 是忠实可行路径
3. FLAVR 有模块级全局状态（`useBias`/`batchnorm`），批量构建须逐模型显式设置
4. 「上游有 ONNX」≠「项目后端可加载」：本次仅备资产，未改 `BackendType`/`SUPPORTED_MODELS`
5. 待决策：`core/pytorch_models/vfi_torch/flavr/resnet_3d.py` 的 encoder BN 隐性不一致（`strict=False` 静默吞掉 100 个 BN key）

---

*Last updated: 2026-10-01 19:00*
