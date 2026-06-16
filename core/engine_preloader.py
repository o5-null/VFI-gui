"""Engine Preloader: pre-initialize inference engines in background threads.

This module provides EnginePreloader that preloads inference engines at startup
to reduce first-inference latency. Engines are loaded in background threads
using threading.Thread (NOT QThread - this is a core module, not UI).

Architecture:
    EnginePreloader → EngineManager (query engines) → InProcessBackend (create/initialize)
    
Preloading时机:
    - 启动时 (preload_all)
    - 用户选择引擎时 (preload_on_demand)
    - 空闲时 (preload_all)

Thread Safety:
    - threading.Thread for background loading
    - threading.Event for stop signaling
    - threading.Lock for _backends dict access

Usage:
    from core.engine_preloader import engine_preloader
    
    # Preload all engines at startup
    engine_preloader.preload_all()
    
    # Check if engine is ready
    if engine_preloader.is_preloaded("auto_torch_cuda0"):
        backend = engine_preloader.get_preloaded_backend("auto_torch_cuda0")
    
    # Stop preloading on shutdown
    engine_preloader.stop()
"""

from __future__ import annotations

import threading

from loguru import logger

from core.types import BackendConfig, EngineStatus
from core.engine_manager import EngineManager, EngineInstance, engine_manager
from core.backends.inprocess_backend import InProcessBackend
from core.backends.base_backend import BackendFactory
from core.events import preloader_started, preloader_engine_loaded, preloader_finished


class EnginePreloader:
    """引擎预加载器：启动时预初始化引擎，减少首次推理延迟。

    在后台线程中预加载所有配置的引擎，避免首次推理时的加载延迟。
    使用 threading.Thread 进行后台加载（非 QThread）。

    Attributes:
        _backends: 已预加载的后端实例字典 {engine_id: InProcessBackend}
        _stop_event: 停止信号事件
        _lock: 线程安全锁
        _preload_thread: 当前预加载线程
        _is_preloading: 是否正在预加载

    Example:
        preloader = EnginePreloader()
        preloader.preload_all()  # 启动时预加载所有引擎
        
        # 获取预加载的后端
        backend = preloader.get_preloaded_backend("auto_torch_cuda0")
        
        # 关闭时停止预加载
        preloader.stop()
    """

    def __init__(self, manager: EngineManager | None = None):
        """初始化引擎预加载器。

        Args:
            manager: EngineManager 实例（默认使用全局 singleton）
        """
        self._manager: EngineManager = manager or engine_manager
        self._backends: dict[str, InProcessBackend] = {}
        self._stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._is_preloading: bool = False
        self._on_demand_threads: dict[str, threading.Thread] = {}

    def preload_all(self) -> None:
        """预加载所有配置的引擎。

        启动后台线程加载所有 EngineManager 中注册的引擎。
        不阻塞调用线程 - 立即返回，加载在后台进行。

        流程:
            1. 获取所有引擎列表
            2. 发射 preloader_started 事件
            3. 后台线程遍历每个引擎:
               a. 设置状态为 LOADING
               b. 创建 InProcessBackend
               c. 调用 initialize() 和 load_model()
               d. 成功则标记 READY，失败则标记 ERROR
               e. 发射 preloader_engine_loaded 事件
            4. 发射 preloader_finished 事件
        """
        if self._is_preloading:
            logger.warning("Preload already in progress, skipping")
            return

        engines = self._manager.list_engines()
        if not engines:
            logger.info("No engines to preload")
            return

        # Reset stop event for new preload cycle
        self._stop_event.clear()
        self._is_preloading = True

        # Emit start event
        total = len(engines)
        _ = preloader_started.send(self, total=total)
        logger.info(f"Starting preload of {total} engines")

        # Start background thread
        self._preload_thread = threading.Thread(
            target=self._preload_all_worker,
            args=(engines,),
            daemon=True,
        )
        self._preload_thread.start()

    def _preload_all_worker(self, engines: list[EngineInstance]) -> None:
        """后台线程: 加载所有引擎。

        Args:
            engines: 要加载的引擎列表
        """
        loaded_count = 0
        failed_count = 0

        for engine in engines:
            # Check for stop signal
            if self._stop_event.is_set():
                logger.info("Preload stopped by stop signal")
                break

            success = self._load_single_engine(engine)
            if success:
                loaded_count += 1
            else:
                failed_count += 1

            # Emit per-engine event
            _ = preloader_engine_loaded.send(
                self,
                engine_id=engine.engine_id,
                success=success,
            )

        # Mark as complete
        self._is_preloading = False

        # Emit finished event
        _ = preloader_finished.send(self, loaded=loaded_count, failed=failed_count)
        logger.info(f"Preload complete: {loaded_count} loaded, {failed_count} failed")

    def _load_single_engine(self, engine: EngineInstance) -> bool:
        """加载单个引擎。

        Args:
            engine: 要加载的引擎实例

        Returns:
            True if loaded successfully, False otherwise
        """
        engine_id = engine.engine_id

        # Skip if already preloaded
        if self.is_preloaded(engine_id):
            logger.debug(f"Engine '{engine_id}' already preloaded, skipping")
            return True

        # Skip subprocess engines - they cannot be preloaded in-process
        if engine.execution_mode == "subprocess":
            logger.debug(
                f"Engine '{engine_id}' is subprocess mode, cannot preload in-process"
            )
            return True

        # Set status to LOADING
        _ = self._manager.update_status(engine_id, EngineStatus.LOADING)

        try:
            # Create BackendConfig from engine config
            config = BackendConfig(
                backend_type=engine.backend_type,
                device=engine.gpu_device,
                precision=str(engine.model_config.get("precision", "fp16")),
                models_dir="models",
                temp_dir="temp",
                output_dir="output",
            )

            # Create InProcessBackend via BackendFactory
            wrapped_backend = BackendFactory.create(engine.backend_type, config)
            backend = InProcessBackend(config, wrapped_backend=wrapped_backend)

            # Initialize backend
            if not backend.initialize():
                _ = self._manager.update_status(engine_id, EngineStatus.ERROR)
                logger.error(f"Failed to initialize engine '{engine_id}'")
                return False

            # Load model if model_config is provided
            if engine.model_config:
                if not backend.load_model(engine.model_config):
                    _ = self._manager.update_status(engine_id, EngineStatus.ERROR)
                    logger.error(f"Failed to load model for engine '{engine_id}'")
                    return False

            # Store preloaded backend
            with self._lock:
                self._backends[engine_id] = backend

            # Set status to READY
            _ = self._manager.update_status(engine_id, EngineStatus.READY)
            logger.info(f"Engine '{engine_id}' preloaded successfully")
            return True

        except Exception as e:
            _ = self._manager.update_status(engine_id, EngineStatus.ERROR)
            logger.error(f"Error preloading engine '{engine_id}': {e}")
            return False

    def preload_on_demand(self, engine_id: str) -> None:
        """按需加载单个引擎。

        在后台线程中加载指定引擎，不阻塞调用线程。

        Args:
            engine_id: 要加载的引擎 ID
        """
        # Check if already preloaded
        if self.is_preloaded(engine_id):
            logger.debug(f"Engine '{engine_id}' already preloaded")
            return

        # Check if engine exists
        engine = self._manager.get_engine(engine_id)
        if engine is None:
            logger.warning(f"Engine '{engine_id}' not found in EngineManager")
            return

        # Check if already loading in on-demand thread
        if engine_id in self._on_demand_threads:
            existing_thread = self._on_demand_threads[engine_id]
            if existing_thread.is_alive():
                logger.debug(f"Engine '{engine_id}' already loading on-demand")
                return

        # Start background thread for single engine
        thread = threading.Thread(
            target=self._preload_on_demand_worker,
            args=(engine,),
            daemon=True,
        )
        self._on_demand_threads[engine_id] = thread
        thread.start()
        logger.info(f"Started on-demand preload for engine '{engine_id}'")

    def _preload_on_demand_worker(self, engine: EngineInstance) -> None:
        """后台线程: 按需加载单个引擎。

        Args:
            engine: 要加载的引擎实例
        """
        engine_id = engine.engine_id
        success = self._load_single_engine(engine)

        # Emit event
        _ = preloader_engine_loaded.send(self, engine_id=engine_id, success=success)

        # Clean up thread reference
        if engine_id in self._on_demand_threads:
            del self._on_demand_threads[engine_id]

    def stop(self) -> None:
        """停止预加载。

        设置停止信号，等待后台线程结束（带超时）。
        """
        # Signal stop
        self._stop_event.set()

        # Wait for preload_all thread
        if self._preload_thread and self._preload_thread.is_alive():
            self._preload_thread.join(timeout=5.0)
            if self._preload_thread.is_alive():
                logger.warning("Preload thread did not stop within timeout")

        # Wait for on-demand threads
        for engine_id, thread in list(self._on_demand_threads.items()):
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        f"On-demand thread for '{engine_id}' did not stop within timeout"
                    )

        self._is_preloading = False
        logger.info("Engine preloader stopped")

    def get_preloaded_backend(self, engine_id: str) -> InProcessBackend | None:
        """获取预加载的后端实例。

        Args:
            engine_id: 引擎 ID

        Returns:
            InProcessBackend 实例，如果未预加载则返回 None
        """
        with self._lock:
            return self._backends.get(engine_id)

    def is_preloaded(self, engine_id: str) -> bool:
        """检查引擎是否已预加载。

        Args:
            engine_id: 引擎 ID

        Returns:
            True 如果引擎已预加载并准备就绪
        """
        with self._lock:
            backend = self._backends.get(engine_id)
            if backend is None:
                return False
            return backend.engine_status == EngineStatus.READY

    def is_preloading(self) -> bool:
        """检查是否正在预加载。

        Returns:
            True 如果预加载正在进行
        """
        return self._is_preloading

    def get_preloaded_count(self) -> int:
        """获取已预加载的引擎数量。

        Returns:
            已成功预加载的引擎数量
        """
        with self._lock:
            return len([
                b for b in self._backends.values()
                if b.engine_status == EngineStatus.READY
            ])

    def clear_preloaded(self) -> None:
        """清除所有预加载的后端。

        调用每个后端的 cleanup() 并清空缓存。
        """
        with self._lock:
            for engine_id, backend in self._backends.items():
                try:
                    backend.cleanup()
                    _ = self._manager.update_status(engine_id, EngineStatus.IDLE)
                except Exception as e:
                    logger.warning(f"Error cleaning up backend '{engine_id}': {e}")

            self._backends.clear()
            logger.info("All preloaded backends cleared")


# Singleton instance
engine_preloader = EnginePreloader()


__all__ = [
    "EnginePreloader",
    "engine_preloader",
]