# AGENTS.md — IO 层

视频帧的流式读取、写入、缓存和生命周期管理。

> 本文件是项目分层 AGENTS.md 的第二层。上一层：[`VFI-gui/AGENTS.md`](../../AGENTS.md)

---

## 结构

```
core/io/
├── ordered_buffer.py    # OrderedResultBuffer — 乱序推理结果按帧序写出
├── frame_lifecycle.py   # FrameLifecycle — 帧写一次 + 消费者跟踪
├── frame_cache.py       # FrameCache — 引用计数 + LRU 驱逐
├── frame_reader.py      # FrameReader — 视频帧读取
├── streaming_reader.py  # StreamingFramePairReader — 流式帧对读取
├── frame_writer.py      # FrameWriter — 帧写出
├── async_io.py          # AsyncFileHandler — 异步文件操作
├── serializers.py       # 序列化工具
├── data_validator.py    # SchemaValidator — 数据校验
└── export_import_manager.py  # 配置导入/导出
```

## 核心模式

### OrderedResultBuffer（乱序重排）

```python
# 推理可以乱序完成，但输出必须按帧序
buffer = OrderedResultBuffer(writer)
buffer.submit(frame_idx=5, data=...)  # 缓存，因为 next_write=0
buffer.submit(frame_idx=0, data=...)  # 立即写出 0，然后 1...直到缺帧
```

- 所有写入在锁保护下进行（线程安全）
- `_next_write` 指针：只有连续帧到达时触发写出
- 思路：顺序写入，乱序缓冲

### FrameLifecycle（写一次 + 消费者跟踪）

```python
# 三条规则
# 1. 每帧只写一次
# 2. 帧在所有消费者完成后才能释放
# 3. 场景切/重复帧直接写出，不走推理
lifecycle = FrameLifecycle()
lifecycle.register(frame_idx=5, subtask_id="pair_4_5")
lifecycle.register(frame_idx=5, subtask_id="pair_5_6")
# frame_5 被两个子任务引用 → 两个都完成后才释放
```

### FrameCache（引用计数 + LRU）

```python
# acquire() → 增引用
# release() → 减引用，归零后可驱逐
# 内存超限 → 驱逐 LRU + refcount=0 的帧
cache = FrameCache(max_memory_mb=4096)
bundle = cache.put(file_path, frames, metadata, consumer_id)
```

- 锁保护所有操作（线程安全）
- LRU 驱逐仅在内存超限时触发
- 引用计数归零是**可驱逐前提**，不是立即释放

## 数据流

```
FrameReader → StreamingFramePairReader → [Inference] → OrderedResultBuffer → FrameWriter
                  ↑                                   ↑
              FrameCache ←────────────────────── FrameLifecycle
```

## 约束

- ❌ `OrderedResultBuffer` 不允许直接调用 `_writer.write()`，必须通过 `submit()`
- ❌ `FrameLifecycle` 不允许手动释放帧，必须通过 `can_release()` 检查
- ✅ 新增 IO 组件应遵循现有的锁模式（`threading.Lock`）
- ✅ 帧数据传递使用 `ProcessedFrameData`（定义在 `core/types.py`）
