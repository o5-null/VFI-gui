# AGENTS.md — UI 层

PyQt6 图形界面，使用 ViewModel/Controller 模式 + Blinker 事件解耦。

> 本文件是项目分层 AGENTS.md 的第二层。上一层：[`VFI-gui/AGENTS.md`](../../AGENTS.md)

---

## 结构

```
ui/
├── main_window.py      # MainWindow — 主窗口（qBittorrent 风格布局）
├── app.py              # VFIApp — 应用入口
├── controllers/        # 无状态控制器（Blinker 信号收发）
│   ├── processing_controller.py  # 处理流程控制
│   ├── queue_controller.py       # 队列控制
│   └── settings_controller.py    # 设置控制
├── viewmodels/         # 有状态视图模型（QObject）
│   ├── pipeline_viewmodel.py     # 管线状态
│   ├── queue_viewmodel.py        # 队列状态
│   ├── task_viewmodel.py         # 任务状态 + GPU 监控
│   ├── device_viewmodel.py       # 设备状态
│   └── codec_viewmodel.py        # 编码器状态
├── widgets/            # 自定义组件
│   ├── sidebar/         # 侧边栏（状态筛选/分类/标签）
│   ├── task_list/       # 任务表格
│   ├── details/         # 任务详情标签页
│   └── dialogs/         # 对话框
├── pages/              # 页面（当前未使用，统一视图）
└── styles/             # 主题/图标管理
```

## 架构模式

### ViewModel / Controller 双层

```
用户操作 → Controller（无状态，发信号）
                │
                ▼ Blinker 信号
                │
ViewModels（有状态 QObject，收信号更新 UI）
                │
                ▼ Qt signal/slot
                │
            Widgets 渲染
```

**Controllers** — 无状态，纯逻辑：
```python
class ProcessingController:
    def start_processing(self, task):
        events.processing_started.send(self, task=task)
```

**ViewModels** — 有状态，数据持有者：
```python
class TaskViewModel(QObject):
    stateChanged = Signal()
    def __init__(self):
        self.tasks: Dict[str, TaskProgressVO] = {}
```

### 主窗口布局（qBittorrent 风格）

```
Menu Bar ──────── 文件 | 编辑 | 视图 | 工具 | 帮助
Toolbar ───────── [打开] [添加] [开始] [暂停] [停止]
├── Sidebar      │  TaskTableView
│   状态筛选      │  ┌──────────────────────────┐
│   分类          │  │ 名称 │ 状态 │ 进度 │ FPS  │
│   标签          │  └──────────────────────────┘
├────────────────┤  TaskDetailsTabs
│   状态栏        │  [通用] [进度] [日志] [GPU]
```

**重要设计决策**：无页面切换，单一统一视图。`ConfigPage` 改为对话框。

## 数据流

```
QueueController ──→ QueueViewModel ──→ Sidebar 更新
       │
       ▼
ProcessingController ──→ TaskViewModel ──→ TaskTableView
       │                       │
       ▼                       ▼
    Backend              GPU Monitor（占位）
```

## 约束

- ❌ Controller 不能持有状态（是无状态的逻辑层）
- ❌ ViewModel 不能直接操作 Widget（通过 Qt Signal 更新视图）
- ✅ 跨层通信使用 `core/events.py` 的 Blinker 信号
- ✅ ViewModel 内部状态变化使用 Qt `Signal`
- ✅ GPU 监控当前是占位符，无数据源连接
