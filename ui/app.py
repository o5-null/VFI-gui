"""VFIApp - Dependency Injection Container for VFI-gui.

Creates and manages all Core components, ViewModels, and Controllers.
Provides centralized access via viewmodels and controllers properties.

Usage:
    from ui.app import VFIApp
    
    app = VFIApp()
    
    # Access ViewModels
    pipeline_vm = app.viewmodels.pipeline
    task_vm = app.viewmodels.task
    
    # Access Controllers
    processing_ctrl = app.controllers.processing
    
    # Start processing
    processing_ctrl.start_task(video_path, pipeline_vm.to_pipeline_config())
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from loguru import logger

# Core components
from core.config_provider import get_config
from core.config.config_facade import ConfigFacade
from core.model_manager import ModelManager
from core.model_selection import ModelSelectionManager
from core.queue_manager import QueueManager
from core.task_orchestrator import TaskOrchestrator

# ViewModels (lazy import to avoid circular dependencies)
if TYPE_CHECKING:
    from ui.viewmodels import (
        ViewModelContainer,
        CodecViewModel,
        PipelineViewModel,
        TaskViewModel,
        QueueViewModel,
        DeviceViewModel,
    )
    from ui.controllers import (
        ControllerContainer,
        ProcessingController,
        QueueController,
        SettingsController,
    )


class VFIApp:
    """Dependency Injection Container for VFI-gui application.
    
    This class:
    - Creates all Core components (ConfigFacade, ModelManager, etc.)
    - Creates all ViewModels in dependency order
    - Creates all Controllers
    - Provides centralized access via viewmodels and controllers properties
    
    Dependency Order:
        1. Core: ConfigFacade -> ModelManager -> ModelSelectionManager
        2. Core: QueueManager -> TaskOrchestrator
        3. ViewModels: CodecVM -> PipelineVM -> TaskVM -> QueueVM -> DeviceVM
        4. Controllers: ProcessingCtrl -> QueueCtrl -> SettingsCtrl
    
    Usage:
        app = VFIApp()
        
        # Access ViewModels
        app.viewmodels.pipeline  # PipelineViewModel
        app.viewmodels.task      # TaskViewModel
        app.viewmodels.queue     # QueueViewModel
        app.viewmodels.device    # DeviceViewModel
        app.viewmodels.codec     # CodecViewModel
        
        # Access Controllers
        app.controllers.processing  # ProcessingController
        app.controllers.queue       # QueueController
        app.controllers.settings    # SettingsController
    """
    
    def __init__(self):
        """Initialize VFIApp with all dependencies."""
        logger.info("Initializing VFIApp...")
        
        # ====================
        # 1. Create Core Components
        # ====================
        
        self._config: ConfigFacade = get_config()
        logger.debug("ConfigFacade created")
        
        self._model_manager: ModelManager = ModelManager(self._config)
        logger.debug("ModelManager created")
        
        self._model_selection: ModelSelectionManager = ModelSelectionManager(
            self._config,
            self._model_manager,
        )
        logger.debug("ModelSelectionManager created")
        
        self._queue_manager: QueueManager = QueueManager()
        logger.debug("QueueManager created")
        
        self._orchestrator: TaskOrchestrator = TaskOrchestrator(self._config)
        logger.debug("TaskOrchestrator created")
        
        # ====================
        # 2. Create ViewModels (in dependency order)
        # ====================
        
        # CodecViewModel first (no dependencies)
        from ui.viewmodels.codec_viewmodel import CodecViewModel
        self._codec_vm = CodecViewModel(self._config)
        logger.debug("CodecViewModel created")
        
        # PipelineViewModel (depends on CodecViewModel and ModelSelectionManager)
        from ui.viewmodels.pipeline_viewmodel import PipelineViewModel
        self._pipeline_vm = PipelineViewModel(
            self._config,
            self._model_selection,
            self._codec_vm,
        )
        logger.debug("PipelineViewModel created")
        
        # TaskViewModel (no dependencies)
        from ui.viewmodels.task_viewmodel import TaskViewModel
        self._task_vm = TaskViewModel()
        logger.debug("TaskViewModel created")
        
        # QueueViewModel (depends on QueueManager)
        from ui.viewmodels.queue_viewmodel import QueueViewModel
        self._queue_vm = QueueViewModel(self._queue_manager)
        logger.debug("QueueViewModel created")
        
        # DeviceViewModel (no dependencies, uses global managers)
        from ui.viewmodels.device_viewmodel import DeviceViewModel
        self._device_vm = DeviceViewModel()
        logger.debug("DeviceViewModel created")
        
        # ====================
        # 3. Create Controllers
        # ====================
        
        # ProcessingController (depends on Orchestrator, TaskVM, PipelineVM)
        from ui.controllers.processing_controller import ProcessingController
        self._processing_ctrl = ProcessingController(
            self._orchestrator,
            self._task_vm,
            self._pipeline_vm,
        )
        logger.debug("ProcessingController created")
        
        # QueueController (depends on QueueManager)
        from ui.controllers.queue_controller import QueueController
        self._queue_ctrl = QueueController(self._queue_manager)
        logger.debug("QueueController created")
        
        # SettingsController (depends on ConfigFacade)
        from ui.controllers.settings_controller import SettingsController
        self._settings_ctrl = SettingsController(self._config)
        logger.debug("SettingsController created")
        
        logger.info("VFIApp initialized successfully")
    
    # ====================
    # Properties - ViewModels
    # ====================
    
    @property
    def viewmodels(self) -> "ViewModelContainer":
        """Get ViewModel container."""
        from ui.viewmodels import ViewModelContainer
        return ViewModelContainer(
            pipeline=self._pipeline_vm,
            task=self._task_vm,
            queue=self._queue_vm,
            device=self._device_vm,
            codec=self._codec_vm,
        )
    
    @property
    def pipeline_vm(self) -> "PipelineViewModel":
        """Get PipelineViewModel directly."""
        return self._pipeline_vm
    
    @property
    def task_vm(self) -> "TaskViewModel":
        """Get TaskViewModel directly."""
        return self._task_vm
    
    @property
    def queue_vm(self) -> "QueueViewModel":
        """Get QueueViewModel directly."""
        return self._queue_vm
    
    @property
    def device_vm(self) -> "DeviceViewModel":
        """Get DeviceViewModel directly."""
        return self._device_vm
    
    @property
    def codec_vm(self) -> "CodecViewModel":
        """Get CodecViewModel directly."""
        return self._codec_vm
    
    # ====================
    # Properties - Controllers
    # ====================
    
    @property
    def controllers(self) -> "ControllerContainer":
        """Get Controller container."""
        from ui.controllers import ControllerContainer
        return ControllerContainer(
            processing=self._processing_ctrl,
            queue=self._queue_ctrl,
            settings=self._settings_ctrl,
        )
    
    @property
    def processing_ctrl(self) -> "ProcessingController":
        """Get ProcessingController directly."""
        return self._processing_ctrl
    
    @property
    def queue_ctrl(self) -> "QueueController":
        """Get QueueController directly."""
        return self._queue_ctrl
    
    @property
    def settings_ctrl(self) -> "SettingsController":
        """Get SettingsController directly."""
        return self._settings_ctrl
    
    # ====================
    # Properties - Core Components
    # ====================
    
    @property
    def config(self) -> ConfigFacade:
        """Get ConfigFacade."""
        return self._config
    
    @property
    def model_manager(self) -> ModelManager:
        """Get ModelManager."""
        return self._model_manager
    
    @property
    def model_selection(self) -> ModelSelectionManager:
        """Get ModelSelectionManager."""
        return self._model_selection
    
    @property
    def queue_manager(self) -> QueueManager:
        """Get QueueManager."""
        return self._queue_manager
    
    @property
    def orchestrator(self) -> TaskOrchestrator:
        """Get TaskOrchestrator."""
        return self._orchestrator
    
    # ====================
    # Lifecycle Methods
    # ====================
    
    def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        logger.info("Shutting down VFIApp...")
        
        # Stop device polling
        self._device_vm.stop_polling()
        
        # Shutdown orchestrator
        self._orchestrator.shutdown()
        
        # Save configuration
        self._config.save()
        
        logger.info("VFIApp shutdown complete")
    
    def start_device_polling(self) -> None:
        """Start device monitoring."""
        self._device_vm.start_polling()
        logger.debug("Device polling started")
    
    def stop_device_polling(self) -> None:
        """Stop device monitoring."""
        self._device_vm.stop_polling()
        logger.debug("Device polling stopped")
    
    def refresh_models(self) -> None:
        """Refresh model list."""
        self._model_selection.refresh()
        self._pipeline_vm.refresh_models()
        logger.debug("Models refreshed")
    
    def save_config(self) -> None:
        """Save all configuration."""
        self._pipeline_vm.persist()
        self._config.save()
        logger.debug("Configuration saved")


# Global app instance (singleton pattern)
_app_instance: Optional[VFIApp] = None


def get_app() -> VFIApp:
    """Get the global VFIApp instance.
    
    Creates the instance on first call, returns the same instance thereafter.
    
    Returns:
        VFIApp instance
    """
    global _app_instance
    if _app_instance is None:
        _app_instance = VFIApp()
    return _app_instance


def reset_app() -> None:
    """Reset the global VFIApp instance.
    
    Forces creation of a new instance on next get_app() call.
    Useful for testing.
    """
    global _app_instance
    if _app_instance is not None:
        _app_instance.shutdown()
    _app_instance = None


__all__ = ["VFIApp", "get_app", "reset_app"]