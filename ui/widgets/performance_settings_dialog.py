"""性能设置对话框。

提供用户界面来配置多线程推理和其他性能相关设置。
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QGroupBox,
    QMessageBox,
    QWidget,
    QCheckBox,
    QComboBox,
)
from PyQt6.QtCore import Qt

from core import tr


class PerformanceSettingsDialog(QDialog):
    """性能设置对话框。
    
    允许用户配置多线程推理、缓存清理间隔等性能相关设置。
    """
    
    def __init__(self, parent: Optional[QWidget] = None, settings_controller=None):
        super().__init__(parent)
        
        self._settings_controller = settings_controller
        self._setup_ui()
        self._load_settings()
        
        self.setWindowTitle(tr("Performance Settings"))
        self.setMinimumWidth(450)
    
    def _setup_ui(self):
        """设置界面。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 说明标签
        info_label = QLabel(tr(
            "Configure performance settings for video processing.\n"
            "Multi-threaded inference can significantly improve processing speed "
            "on high-resolution videos."
        ))
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(info_label)
        
        # 推理设置组
        inference_group = QGroupBox(tr("Inference Settings"))
        inference_layout = QFormLayout(inference_group)
        inference_layout.setSpacing(10)
        
        # 启用多线程推理
        self.enable_threading_check = QCheckBox(tr("Enable multi-threaded inference"))
        self.enable_threading_check.setToolTip(tr(
            "Use multiple threads for frame interpolation. "
            "Recommended for high-resolution videos."
        ))
        self.enable_threading_check.stateChanged.connect(self._on_threading_changed)
        inference_layout.addRow(self.enable_threading_check)
        
        # 推理线程数
        self.inference_threads_spin = QSpinBox()
        self.inference_threads_spin.setRange(1, 16)
        self.inference_threads_spin.setSingleStep(1)
        self.inference_threads_spin.setToolTip(tr(
            "Number of worker threads for inference. "
            "More threads = faster processing but more VRAM usage.\n"
            "Recommended: 2-4 for 1080p, 1-2 for 4K+"
        ))
        inference_layout.addRow(tr("Inference Threads:"), self.inference_threads_spin)
        
        # 任务队列大小
        self.task_queue_spin = QSpinBox()
        self.task_queue_spin.setRange(10, 500)
        self.task_queue_spin.setSingleStep(10)
        self.task_queue_spin.setToolTip(tr(
            "Maximum number of tasks in the queue. "
            "Larger queue = better throughput but more memory usage."
        ))
        inference_layout.addRow(tr("Task Queue Size:"), self.task_queue_spin)
        
        layout.addWidget(inference_group)
        
        # 缓存设置组
        cache_group = QGroupBox(tr("Cache Settings"))
        cache_layout = QFormLayout(cache_group)
        cache_layout.setSpacing(10)
        
        # 缓存清理间隔
        self.cache_interval_spin = QSpinBox()
        self.cache_interval_spin.setRange(1, 100)
        self.cache_interval_spin.setSingleStep(1)
        self.cache_interval_spin.setSuffix(tr(" frames"))
        self.cache_interval_spin.setToolTip(tr(
            "Clear CUDA cache every N frames to prevent OOM errors. "
            "Lower value = less VRAM usage but slower processing."
        ))
        cache_layout.addRow(tr("Clear Cache Every:"), self.cache_interval_spin)
        
        layout.addWidget(cache_group)
        
        # 取消设置组
        cancel_group = QGroupBox(tr("Cancellation Settings"))
        cancel_layout = QFormLayout(cancel_group)
        cancel_layout.setSpacing(10)
        
        # 强制终止选项
        self.force_terminate_check = QCheckBox(tr("Force terminate on cancel"))
        self.force_terminate_check.setToolTip(tr(
            "Immediately force terminate worker threads when cancelling. "
            "Faster cancellation but may leave resources in inconsistent state."
        ))
        cancel_layout.addRow(self.force_terminate_check)
        
        # 取消超时
        self.cancel_timeout_spin = QSpinBox()
        self.cancel_timeout_spin.setRange(1, 30)
        self.cancel_timeout_spin.setSingleStep(1)
        self.cancel_timeout_spin.setSuffix(tr(" seconds"))
        self.cancel_timeout_spin.setToolTip(tr(
            "Timeout before forcing thread termination during cancellation."
        ))
        cancel_layout.addRow(tr("Cancel Timeout:"), self.cancel_timeout_spin)
        
        layout.addWidget(cancel_group)
        
        # 推荐设置按钮
        quick_layout = QHBoxLayout()
        
        self.low_end_btn = QPushButton(tr("Low-End PC"))
        self.low_end_btn.setToolTip(tr("Settings for low-end GPUs (4-6GB VRAM)"))
        self.low_end_btn.clicked.connect(self._apply_low_end_settings)
        quick_layout.addWidget(self.low_end_btn)
        
        self.mid_range_btn = QPushButton(tr("Mid-Range PC"))
        self.mid_range_btn.setToolTip(tr("Settings for mid-range GPUs (8-12GB VRAM)"))
        self.mid_range_btn.clicked.connect(self._apply_mid_range_settings)
        quick_layout.addWidget(self.mid_range_btn)
        
        self.high_end_btn = QPushButton(tr("High-End PC"))
        self.high_end_btn.setToolTip(tr("Settings for high-end GPUs (16GB+ VRAM)"))
        self.high_end_btn.clicked.connect(self._apply_high_end_settings)
        quick_layout.addWidget(self.high_end_btn)
        
        layout.addLayout(quick_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_btn = QPushButton(tr("OK"))
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _on_threading_changed(self, state):
        """多线程启用状态改变。"""
        enabled = state == Qt.CheckState.Checked.value
        self.inference_threads_spin.setEnabled(enabled)
        self.task_queue_spin.setEnabled(enabled)
    
    def _apply_low_end_settings(self):
        """应用低端配置。"""
        self.enable_threading_check.setChecked(False)
        self.inference_threads_spin.setValue(1)
        self.task_queue_spin.setValue(50)
        self.cache_interval_spin.setValue(5)
        self.force_terminate_check.setChecked(True)
        self.cancel_timeout_spin.setValue(2)
    
    def _apply_mid_range_settings(self):
        """应用中端配置。"""
        self.enable_threading_check.setChecked(True)
        self.inference_threads_spin.setValue(2)
        self.task_queue_spin.setValue(100)
        self.cache_interval_spin.setValue(10)
        self.force_terminate_check.setChecked(False)
        self.cancel_timeout_spin.setValue(5)
    
    def _apply_high_end_settings(self):
        """应用高端配置。"""
        self.enable_threading_check.setChecked(True)
        self.inference_threads_spin.setValue(4)
        self.task_queue_spin.setValue(200)
        self.cache_interval_spin.setValue(20)
        self.force_terminate_check.setChecked(False)
        self.cancel_timeout_spin.setValue(10)
    
    def _load_settings(self):
        """加载设置。"""
        if not self._settings_controller:
            # 使用默认设置
            self._apply_mid_range_settings()
            return
        
        # 加载多线程设置
        enable_threading = self._settings_controller.get_setting(
            "performance.inference.enable_threading", True
        )
        self.enable_threading_check.setChecked(enable_threading)
        
        inference_threads = self._settings_controller.get_setting(
            "performance.inference.threads", 2
        )
        self.inference_threads_spin.setValue(inference_threads)
        
        task_queue_size = self._settings_controller.get_setting(
            "performance.inference.task_queue_size", 100
        )
        self.task_queue_spin.setValue(task_queue_size)
        
        # 加载缓存设置
        cache_interval = self._settings_controller.get_setting(
            "performance.cache.clear_interval", 10
        )
        self.cache_interval_spin.setValue(cache_interval)
        
        # 加载取消设置
        force_terminate = self._settings_controller.get_setting(
            "performance.cancellation.force_terminate", False
        )
        self.force_terminate_check.setChecked(force_terminate)
        
        cancel_timeout = self._settings_controller.get_setting(
            "performance.cancellation.timeout", 5
        )
        self.cancel_timeout_spin.setValue(cancel_timeout)
    
    def _on_ok(self):
        """点击确定按钮。"""
        if self._settings_controller:
            # 保存设置
            self._settings_controller.set_setting(
                "performance.inference.enable_threading",
                self.enable_threading_check.isChecked()
            )
            self._settings_controller.set_setting(
                "performance.inference.threads",
                self.inference_threads_spin.value()
            )
            self._settings_controller.set_setting(
                "performance.inference.task_queue_size",
                self.task_queue_spin.value()
            )
            self._settings_controller.set_setting(
                "performance.cache.clear_interval",
                self.cache_interval_spin.value()
            )
            self._settings_controller.set_setting(
                "performance.cancellation.force_terminate",
                self.force_terminate_check.isChecked()
            )
            self._settings_controller.set_setting(
                "performance.cancellation.timeout",
                self.cancel_timeout_spin.value()
            )
            
            # 保存到文件
            self._settings_controller.save_settings()
        
        self.accept()
    
    def get_settings(self) -> dict:
        """获取当前设置。
        
        Returns:
            设置字典
        """
        return {
            "enable_threading": self.enable_threading_check.isChecked(),
            "inference_threads": self.inference_threads_spin.value(),
            "task_queue_size": self.task_queue_spin.value(),
            "cache_interval": self.cache_interval_spin.value(),
            "force_terminate": self.force_terminate_check.isChecked(),
            "cancel_timeout": self.cancel_timeout_spin.value(),
        }
