# AGENTS.md — Config 配置层

`ConfigFacade` 封装 8 个域配置模块，提供统一访问入口。

> 本文件是项目分层 AGENTS.md 的第二层。上一层：[`VFI-gui/AGENTS.md`](../../AGENTS.md)

---

## 结构

```
core/config/
├── base_config.py       # BaseConfig(ABC) — 所有域配置的基类
├── config_facade.py     # ConfigFacade — 统一入口，组合 8 个域配置
├── pipeline_config.py   # 插值管线配置
├── ui_config.py         # UI 布局/主题配置
├── network_config.py    # 网络/代理配置
├── output_config.py     # 输出格式配置
├── runtime_config.py    # Runtime 环境选择配置
├── paths_config.py      # 路径配置
├── performance_config.py# 性能参数配置
└── vapoursynth_config.py# VapourSynth 配置
```

## 架构模式

### ConfigFacade（组合模式）

```python
config = ConfigFacade()
# 推荐：直接访问域配置
pipeline = config.pipeline.get_interpolation_config()
# 兼容旧式：点号字符串
value = config.get("pipeline.interpolation.model_type")
```

- `ConfigFacade` 是**唯一对外暴露的入口**
- 内部组合 8 个 `BaseConfig` 子类，每个负责自己的配置域
- 所有域配置共享一个 `ExportImportManager` 实例（高效 IO）

### BaseConfig（模板方法模式）

```python
class PipelineConfig(BaseConfig):
    def _load_defaults(self) -> None:
        self._settings = {"interpolation": {"model_type": "rife", ...}}
```

- 子类只需实现 `_load_defaults()`
- `load()` / `save()` 由基类统一处理（JSON 序列化）
- `_io_manager` 提供去抖保存（debounce 0.5s）

## 约束

- ❌ 不允许在 `ConfigFacade` 外部直接实例化域配置类
- ❌ 不允许在域配置间交叉引用（每个域独立）
- ✅ 新增配置域：继承 `BaseConfig`，在 `ConfigFacade.__init__()` 注册
- ✅ 访问配置：始终通过 `ConfigFacade.xxx.get_xxx()` 方法
