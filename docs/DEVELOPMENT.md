# VFI-gui 开发文档

## 项目概述

VFI-gui 是一个基于 PyQt6 的桌面应用程序，为 VSGAN-tensorrt-docker 视频处理工作流提供图形界面。

**功能特性：**
- RIFE 视频插帧
- ESRGAN/CUGAN 超分辨率放大
- 场景检测
- 批处理队列管理
- 多语言支持 (i18n)

## 环境配置

### 虚拟环境

项目使用 **uv** 进行虚拟环境管理。

```bash
# 创建虚拟环境 (Python 3.12+)
cd E:\code\VFI\VFI-gui
uv venv --python 3.12

# 激活虚拟环境
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
uv pip install -r requirements.txt
```

**注意：** venv 位于项目根目录 `E:\code\VFI\.venv`，而非 `VFI-gui\.venv`。

### 依赖项

| 包名 | 版本 | 说明 |
|------|------|------|
| PyQt6 | 6.11.0 | GUI 框架 |
| vapoursynth | 74+ | 视频处理框架 (R74+ 支持 Python 3.12+) |
| polib | 1.2.0 | PO/MO 翻译文件处理 |

### VapourSynth 安装

**重要：** VapourSynth R74+ 才支持直接 pip 安装。

```bash
# R74+ (推荐)
uv pip install vapoursynth

# R73 及更早版本需要先安装系统级 VapourSynth
# 否则会报错: "Couldn't detect vapoursynth installation path"
```

## 国际化 (i18n)

VFI-gui 使用行业标准的 gettext 方案实现多语言支持。

### 支持的语言

| 代码 | 语言 |
|------|------|
| en | English |
| zh_CN | 简体中文 |
| zh_TW | 繁體中文 |

### 目录结构

```
VFI-gui/
├── core/
│   └── i18n.py              # i18n 管理器
├── locales/
│   ├── messages.pot         # 翻译模板
│   ├── zh_CN/
│   │   └── LC_MESSAGES/
│   │       ├── messages.po  # 简体中文翻译
│   │       └── messages.mo  # 编译后的翻译
│   └── zh_TW/
│       └── LC_MESSAGES/
│           ├── messages.po  # 繁体中文翻译
│           └── messages.mo  # 编译后的翻译
└── compile_translations.py  # 编译脚本
```

### 使用方法

```python
from core import tr, init_i18n, get_i18n

# 初始化 (在 main.py 启动时调用)
init_i18n('locales')

# 翻译字符串
print(tr('Ready'))  # 输出: 就绪 (中文) / Ready (英文)

# 运行时切换语言
get_i18n().set_language('zh_CN')

# 监听语言变化
get_i18n().language_changed.connect(lambda lang: print(f'切换到: {lang}'))
```

### 编译翻译

修改 `.po` 文件后，需要编译为 `.mo` 文件：

```bash
python compile_translations.py
```

脚本使用 **polib** 库处理 PO/MO 文件（行业标准）。

### 添加新翻译

1. 在 `locales/messages.pot` 添加新条目
2. 更新各语言的 `.po` 文件
3. 运行 `python compile_translations.py`
4. 在代码中使用 `tr('新字符串')`
5. 为 UI 组件添加 `retranslate_ui()` 方法

### UI 组件国际化

所有 UI 组件都实现了 `retranslate_ui()` 方法，用于语言切换时更新界面：

```python
class MyWidget(QWidget):
    def _setup_ui(self):
        self.label = QLabel(tr('Hello'))
    
    def retranslate_ui(self):
        """语言切换时调用"""
        self.label.setText(tr('Hello'))
```

## 配置系统

配置文件位于用户目录：`~/.config/vfi-gui/settings.json`

### 配置结构

```json
{
  "pipeline": {
    "interpolation": { "enabled": false, "model": "4.22", "multi": 2 },
    "upscaling": { "enabled": true, "engine": "...", "num_streams": 3 },
    "scene_detection": { "enabled": false, "model": 12 }
  },
  "output": { "codec": "hevc_nvenc", "quality": 22 },
  "vapoursynth": { "portable_path": "...", "num_threads": 4 },
  "ui": { "language": "zh_CN" }
}
```

## 运行应用

```bash
cd E:\code\VFI\VFI-gui
python main.py
```

## 开发指南

### 代码风格

- 使用 4 空格缩进
- 最大行长度 100 字符
- 使用类型注解
- 函数命名使用 snake_case

### 项目结构

```
VFI-gui/
├── core/               # 核心逻辑
│   ├── config.py       # 配置管理
│   ├── i18n.py         # 国际化
│   ├── models.py       # 模型管理
│   ├── pipeline.py     # 处理管道
│   ├── processor.py    # 视频处理器
│   └── queue_manager.py # 批处理队列
├── ui/                 # 用户界面
│   ├── main_window.py  # 主窗口
│   ├── widgets/        # UI 组件
│   └── styles/         # 样式表
├── locales/            # 翻译文件
├── main.py             # 入口文件
└── requirements.txt    # 依赖列表
```
