# AGENTS.md — Preprocess 预处理层

帧对决策管线：场景检测 + 重复帧检测。

> 本文件是项目分层 AGENTS.md 的第二层。上一层：[`VFI-gui/AGENTS.md`](../../AGENTS.md)

---

## 结构

```
core/preprocess/
├── pipeline.py       # PreprocessPipeline — 帧对决策主管线
├── scene_detect.py   # SceneDetectorBase + SceneDetectorFactory
└── dup_detect.py     # DuplicateDetector — 基于像素差异的重复帧检测
```

## 架构

### PreprocessPipeline（组合模式）

```python
pipeline = PreprocessPipeline(config, backend_type)

# 对每对帧做流式决策
decision = pipeline.decide(frame0, frame1)
# 返回值: FramePairDecision
#   - INTERPOLATE: 正常插值
#   - SCENE_CUT:   场景切换，写 frame0
#   - DUPLICATE:   重复帧，写 frame0
#   - LAST_FRAME:  序列结束
```

**决策优先级**：
1. `LAST_FRAME` — frame1 为 None（序列终止）
2. `SCENE_CUT` — 场景检测报告切换
3. `DUPLICATE` — 重复检测报告相似帧
4. `INTERPOLATE` — 正常处理

### SceneDetector（策略模式）

```python
detector = SceneDetectorFactory.create(config, backend_type)
# 支持后端：
#   - TORCH: 神经网络场景检测
#   - CPU:   像素差异检测
```

## 约束

- ✅ 单遍流式设计：`decide()` 不依赖未来帧
- ✅ 结果通过 `FramePairAction` 枚举表达
- ❌ 不允许在 `PreprocessPipeline` 内部调用后端推理（通过 `TaskScheduler` 调度）
