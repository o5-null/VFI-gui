# VFI-gui 架构重构文档

## 重构概述

本次重构旨在解决原架构中的耦合问题，提升代码的可维护性和可测试性。

## 原架构问题分析

### 1. 高度耦合
- **UI-核心层耦合**: `codec_manager.py` <-> `codec_settings.py` (103 连接)
- **模型管理耦合**: `models.py` 与多个 UI 面板高度耦合
- **主窗口上帝对象**: `MainWindow` 与几乎所有 UI 控件有直接连接 (52 条边)

### 2. 职责过重
- `MainWindow` 承担了处理、设置、队列管理等多个职责
- `ModelManager` 被 97 个节点引用，成为瓶颈

### 3. 架构边界模糊
- UI 层直接依赖核心层实现
- 缺乏清晰的接口定义

## 新架构设计

### 分层架构

```
UI Layer (ui/)
├── widgets/          # 纯 UI 控件
├── viewmodels/       # 视图模型层 (NEW)
└── controllers/      # 控制器层 (NEW)

Core Layer (core/)
├── interfaces/       # 抽象接口 (NEW)
├── models.py         # 模型管理
├── codec_manager.py  # 编解码管理
├── config.py         # 配置管理
└── ...

Backend Layer (core/backends/, core/torch_backend/)
└── ...
```

### 新增组件

#### 1. ViewModel 层 (`ui/viewmodels/`)

**目的**: 解耦 UI 控件与核心层

**组件**:
- `CodecViewModel`: 编解码配置管理
- `ModelViewModel`: 模型管理
- `ProcessingViewModel`: 视频处理操作
- `QueueViewModel`: 批处理队列管理

**示例**:
```python
# 重构前: UI 直接依赖核心层
class CodecSettingsWidget(QWidget):
    def __init__(self):
        self._codec_manager = CodecManager()  # 直接依赖

# 重构后: UI 通过 ViewModel 间接访问
class CodecSettingsWidget(QWidget):
    def __init__(self):
        self._viewmodel = CodecViewModel()  # 通过 ViewModel
```

#### 2. Controller 层 (`ui/controllers/`)

**目的**: 拆分 MainWindow 的职责

**组件**:
- `ProcessingController`: 处理视频操作
- `SettingsController`: 设置管理
- `QueueController`: 队列管理

**示例**:
```python
# 重构前: MainWindow 处理所有逻辑
class MainWindow(QMainWindow):
    def start_processing(self): ...
    def save_settings(self): ...
    def add_to_queue(self): ...

# 重构后: 职责分离
class MainWindow(QMainWindow):
    def __init__(self):
        self._processing_ctrl = ProcessingController()
        self._settings_ctrl = SettingsController()
        self._queue_ctrl = QueueController()
```

#### 3. 接口层 (`core/interfaces/`)

**目的**: 定义清晰的契约，支持依赖注入和测试

**组件**:
- `IModelManager`: 模型管理接口
- `ICodecManager`: 编解码管理接口
- `IConfig`: 配置管理接口

**示例**:
```python
class IModelManager(ABC):
    @abstractmethod
    def refresh(self) -> None: ...
    
    @abstractmethod
    def get_installed_checkpoints(self) -> List[CheckpointInfo]: ...

# 实现接口
class ModelManager(IModelManager):
    ...

# 使用接口
class ModelViewModel:
    def __init__(self, model_manager: IModelManager):
        self._model_manager = model_manager
```

## 迁移指南

### 1. 更新导入

**旧代码**:
```python
from core import Config, ModelManager, CodecManager
```

**新代码**:
```python
from ui.controllers import SettingsController
from ui.viewmodels import ModelViewModel, CodecViewModel
```

### 2. 替换直接依赖

**旧代码**:
```python
class MyWidget(QWidget):
    def __init__(self):
        self.config = Config()
        self.model_manager = ModelManager(self.config)
```

**新代码**:
```python
class MyWidget(QWidget):
    def __init__(self):
        self._settings_ctrl = SettingsController()
        self._model_viewmodel = ModelViewModel()
```

### 3. 使用信号通信

**旧代码**:
```python
def on_processing_done(self):
    self.update_ui()
```

**新代码**:
```python
def __init__(self):
    self._processing_ctrl.processing_finished.connect(self._on_finished)
```

## 孤立节点处理

原架构中有 317 个孤立节点，处理方式：

### 1. 工具函数整合
- 将分散的工具函数整合到 `core/utils/` 模块
- 添加统一的导入接口

### 2. 常量定义
- 将魔法字符串提取为常量
- 集中到 `core/constants.py`

### 3. 类型定义
- 将重复的类型定义统一
- 集中到 `core/types.py`

## 测试策略

### 1. 单元测试
- 使用 Mock 实现接口进行隔离测试
- 测试 ViewModel 逻辑而不依赖 UI

### 2. 集成测试
- 测试 Controller 与 ViewModel 的集成
- 测试 ViewModel 与核心层的集成

### 3. 示例
```python
def test_codec_viewmodel():
    # 使用 Mock CodecManager
    mock_manager = Mock(spec=ICodecManager)
    viewmodel = CodecViewModel(mock_manager)
    
    # 测试逻辑
    viewmodel.set_codec("hevc_nvenc")
    assert viewmodel.get_current_codec() == "hevc_nvenc"
```

## 性能考虑

### 1. 延迟加载
- ViewModel 延迟创建核心层对象
- 控制器按需初始化

### 2. 信号批处理
- 减少不必要的信号发射
- 使用 `blockSignals()` 进行批量更新

### 3. 缓存策略
- ViewModel 缓存常用数据
- 避免重复查询核心层

## 后续优化建议

1. **依赖注入容器**: 考虑使用 DI 框架管理对象生命周期
2. **事件总线**: 对于跨组件通信，考虑引入事件总线
3. **状态管理**: 对于复杂状态，考虑引入 Redux/MobX 模式
4. **异步处理**: 将更多操作移至后台线程

## 文件清单

### 新增文件
```
ui/viewmodels/
├── __init__.py
├── codec_viewmodel.py
├── model_viewmodel.py
├── processing_viewmodel.py
└── queue_viewmodel.py

ui/controllers/
├── __init__.py
├── processing_controller.py
├── settings_controller.py
└── queue_controller.py

core/interfaces/
├── __init__.py
├── imodel_manager.py
├── icodec_manager.py
└── iconfig.py
```

### 修改文件
- `ui/main_window.py`: 使用新的 Controllers
- `ui/widgets/codec_settings.py`: 使用 CodecViewModel
- `ui/widgets/model_panel.py`: 使用 ModelViewModel
- `ui/widgets/batch_queue.py`: 使用 QueueViewModel

## 验证清单

- [ ] ViewModel 层正确解耦 UI 和核心层
- [ ] Controller 层正确拆分 MainWindow 职责
- [ ] 接口层定义清晰，可被 Mock
- [ ] 所有单元测试通过
- [ ] 性能无明显下降
- [ ] 代码覆盖率不降低
