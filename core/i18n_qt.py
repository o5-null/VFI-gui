"""Qt translation bridge for VFI-gui.

Provides DictTranslator — a QTranslator subclass that uses a Python dict
for translations, bridging the gettext-based i18n system with Qt's self.tr().

This avoids the need for .qm binary files while making QObject.tr() work
with Chinese translations at runtime.
"""



from PyQt6.QtCore import QTranslator, QCoreApplication, QObject


# zh_CN (Simplified Chinese) translation dictionary
ZH_CN_TRANSLATIONS: dict[str, str] = {
    # === MainWindow ===
    "VFI-gui": "VFI-gui",
    "File": "文件",
    "Open Video...": "打开视频...",
    "Open Folder...": "打开文件夹...",
    "Exit": "退出",
    "Edit": "编辑",
    "Settings...": "设置...",
    "View": "视图",
    "Toggle Toolbar": "切换工具栏",
    "Help": "帮助",
    "About VFI-gui": "关于 VFI-gui",
    "Benchmark": "性能测试",
    "Main Toolbar": "主工具栏",
    "Open": "打开",
    "Settings": "设置",
    "About": "关于",
    "Ready": "就绪",
    "Loading...": "加载中...",
    "Processing": "处理中",
    "Paused": "已暂停",
    "Completed": "已完成",
    "Failed": "失败",
    "Cancelled": "已取消",
    "Cancelling...": "正在取消...",
    "Open Video": "打开视频",
    "Open Video dialog will be implemented in future phase.": "打开视频功能将在后续版本中实现。",
    "Open Folder": "打开文件夹",
    "Open Folder dialog will be implemented in future phase.": "打开文件夹功能将在后续版本中实现。",
    "Select Video": "选择视频",
    "Select Folder": "选择文件夹",

    # === ConfigPage ===
    "Drag & drop video here or click to select": "拖放视频到此处或点击选择",
    "Interpolation settings placeholder": "插帧设置",
    "Scene detection settings placeholder": "场景检测设置",
    "Output settings placeholder": "输出设置",
    "Queue preview placeholder": "队列预览",
    "Device info placeholder": "设备信息",
    "Validation Error": "验证错误",
    "No Video Selected": "未选择视频",
    "Please select a video file first.": "请先选择一个视频文件。",
    "▶ Start Processing": "▶ 开始处理",
    "+ Add to Queue": "+ 添加到队列",

    # === ProcessPage ===
    "Hardware Monitor\n(Future: GPU stats)": "硬件监控\n（后续：GPU 统计）",
    "Queue List\n(Future: Queue items)": "队列列表\n（后续：队列项）",
    "Task Log\n(Future: Processing log)": "任务日志\n（后续：处理日志）",
    "Current Task: {0}": "当前任务：{0}",
    "No task running": "没有正在运行的任务",
    "▶ Resume": "▶ 继续",
    "⏸ Pause": "⏸ 暂停",
    "⏹ Stop": "⏹ 停止",
    "◀ Back to Config": "◀ 返回配置",

    # === VideoDropZone ===
    "Drop video here or click to browse": "拖放视频到此处或点击浏览",
    "Browse": "浏览",
    "Change": "更改",
    "Clear": "清除",
    "Size unknown": "大小未知",
    "Video files (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv)": "视频文件 (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv)",
    "Select Video": "选择视频",

    # === InterpolationGroup ===
    "Interpolation": "插帧",
    "Enable Interpolation": "启用插帧",
    "Select interpolation model type": "选择插帧模型类型",
    "Model Type:": "模型类型：",
    "Select model checkpoint/version": "选择模型版本",
    "Checkpoint:": "模型版本：",
    "Frame multiplier for interpolation": "插帧倍率",
    "Multiplier:": "倍率：",
    "Scale factor for model inference": "模型推理缩放系数",
    "Scale:": "缩放：",
    "Scene Change Detection": "场景变化检测",
    "Enable scene change detection for better interpolation": "启用场景变化检测以获得更好的插帧效果",

    # === UpscalingGroup ===
    "Upscaling": "超分辨率",
    "Enable Upscaling": "启用超分辨率",
    "Select upscaling engine": "选择超分辨率引擎",
    "Engine:": "引擎：",

    # === SceneDetectGroup ===
    "Scene Detection": "场景检测",
    "Enable Scene Detection": "启用场景检测",
    "Select scene detection method": "选择场景检测方法",
    "Method:": "方法：",
    "Detection threshold (higher = more sensitive)": "检测阈值（越高越灵敏）",
    "Threshold:": "阈值：",

    # === OutputGroup ===
    "Output": "输出",
    "Select output codec": "选择输出编码",
    "Codec:": "编码：",
    "Quality value (0=best, 51=worst for most codecs)": "质量值（0=最佳，51=最差）",
    "Quality:": "质量：",
    "Encoding preset (speed vs quality)": "编码预设（速度与质量）",
    "Preset:": "预设：",
    "Copy Audio Stream": "复制音频流",
    "Copy original audio stream without re-encoding": "复制原始音频流，不重新编码",

    # === DevicePanel ===
    "Device Info": "设备信息",
    "Detecting...": "检测中...",
    "Device:": "设备：",
    "N/A": "N/A",
    "VRAM:": "显存：",
    "Runtime:": "运行时：",
    "Precision:": "精度：",

    # === QueueListPreview ===
    "Queue": "队列",
    "Clear Completed": "清除已完成",
    "No items in queue": "队列中没有项目",
    "Queued": "排队中",
    "processing": "处理中",
    "completed": "已完成",
    "{0} items ({1} completed)": "{0} 项（{1} 已完成）",

    # === QueueToolbar ===
    "Add": "添加",
    "Remove": "移除",

    # === QueueList ===
    "Task Queue": "任务队列",
    "Clear Completed": "清除已完成",
    "queued": "排队中",

    # === ProgressBar ===
    "FPS: 0.0": "帧率：0.0",
    "FPS: {fps:.1f}": "帧率：{fps:.1f}",
    "ETA: --": "预计：--",
    "ETA: {eta}": "预计：{eta}",
    "Scene cuts: 0": "场景切换：0",
    "Scene cuts: {cuts}": "场景切换：{cuts}",
    "Skipped: 0": "跳过：0",
    "Skipped: {skipped}": "跳过：{skipped}",

    # === TaskLog ===
    "Log": "日志",

    # === GPUMonitor ===
    "Hardware Monitor": "硬件监控",
    "VRAM: N/A": "显存：N/A",
    "VRAM: {used:.1f} / {total:.1f} GB": "显存：{used:.1f} / {total:.1f} GB",
    "GPU Util: N/A": "GPU 利用率：N/A",
    "GPU Util: {util:.0f}%": "GPU 利用率：{util:.0f}%",
    "Temp: N/A": "温度：N/A",
    "Temp: {temp}°C": "温度：{temp}°C",

    # === SettingsDialog ===
    "Settings": "设置",
    "Performance": "性能",
    "Proxy": "代理",
    "Dependencies": "依赖",
    "Inference Settings": "推理设置",
    "Thread Count:": "线程数：",
    "GPU Precision:": "GPU 精度：",
    "Inference Streams:": "推理流数：",
    "Enable torch.compile": "启用 torch.compile",
    "Proxy Configuration": "代理配置",
    "Proxy Type:": "代理类型：",
    "Host:": "主机：",
    "Port:": "端口：",
    "Username:": "用户名：",
    "Password:": "密码：",
    "OK": "确定",
    "Cancel": "取消",
    "External Dependencies": "外部依赖",
    "e.g., 127.0.0.1": "例如：127.0.0.1",
    "Checking...": "检查中...",
    "FFmpeg:": "FFmpeg：",
    "VapourSynth:": "VapourSynth：",
    "Python:": "Python：",
    "PyTorch:": "PyTorch：",
    "Not found": "未找到",
    "Not installed": "未安装",

    # === BenchmarkDialog ===
    "Device:": "设备：",
    "Model:": "模型：",
    "Resolution:": "分辨率：",
    "Run Benchmark": "运行测试",
    "720p (1280×720)": "720p (1280×720)",
    "1080p (1920×1080)": "1080p (1920×1080)",
    "4K (3840×2160)": "4K (3840×2160)",
    "Benchmark functionality coming soon": "性能测试功能即将上线",
    "Auto (Best Available)": "自动（最佳可用）",

    # === AboutDialog ===
    "About %1": "关于 %1",
    "Version: %1": "版本：%1",
    "A PyQt6 desktop application for AI-powered video frame interpolation and upscaling.": "基于 PyQt6 的 AI 视频插帧和超分辨率桌面应用。",
    "License: %1": "许可证：%1",
    "Links:": "链接：",
    "GitHub": "GitHub",
    "RIFE": "RIFE",
    "VSGAN-tensorrt": "VSGAN-tensorrt",
    "Close": "关闭",
}


class DictTranslator(QTranslator):
    """QTranslator backed by a Python dict — no .qm files needed.

    Overrides translate() so that QObject.tr() automatically looks up
    the source text in the provided dictionary.
    """

    def __init__(self, translations: dict[str, str], parent: QObject | None = None):
        super().__init__(parent)
        self._translations: dict[str, str] = translations

    def translate(  # noqa: reportImplicitOverride
        self,
        context: str,
        sourceText: str,
        disambiguation: str | None = None,
        n: int = -1,
    ) -> str:
        """Look up sourceText in the translation dict.

        Args:
            context: Qt class name (ignored — all contexts share one dict)
            sourceText: The original English string to translate
            disambiguation: Optional context hint (ignored)
            n: Plural count (not used in dict-based translation)

        Returns:
            Translated string, or sourceText if no translation found
        """
        if not sourceText:
            return sourceText

        # Direct lookup
        result = self._translations.get(sourceText)
        if result is not None:
            return result

        # Fallback: return original
        return sourceText


def install_translator(language: str = "zh_CN") -> DictTranslator | None:
    """Install a DictTranslator for the given language into QApplication.

    Args:
        language: Language code (currently only zh_CN has translations)

    Returns:
        The installed DictTranslator, or None if language not supported
    """
    translation_dicts = {
        "zh_CN": ZH_CN_TRANSLATIONS,
        # zh_TW can be added later
    }

    if language not in translation_dicts:
        return None

    app = QCoreApplication.instance()
    if app is None:
        return None

    translator = DictTranslator(translation_dicts[language])
    _ = app.installTranslator(translator)
    return translator