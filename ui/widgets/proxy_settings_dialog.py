"""代理设置对话框。

提供用户界面来配置网络代理，用于加速 GitHub 等站点的下载。
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QGroupBox,
    QMessageBox,
    QWidget,
)
from PyQt6.QtCore import Qt

from core import tr
from core.network import NetworkManager, ProxyConfig


class ProxySettingsDialog(QDialog):
    """代理设置对话框。
    
    允许用户配置 HTTP/HTTPS 代理，用于加速 GitHub 等站点的下载。
    支持常见的代理工具如 Clash、v2rayN、Shadowsocks 等。
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._network_manager = NetworkManager()
        self._setup_ui()
        self._load_settings()
        
        self.setWindowTitle(tr("Proxy Settings"))
        self.setMinimumWidth(450)
    
    def _setup_ui(self):
        """设置界面。"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 说明标签
        info_label = QLabel(tr(
            "Configure proxy settings to accelerate downloads from GitHub and other sites.\n"
            "Common proxy tools: Clash (7890), v2rayN (10808), Shadowsocks (1080)"
        ))
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(info_label)
        
        # 代理设置组
        proxy_group = QGroupBox(tr("Proxy Configuration"))
        proxy_layout = QFormLayout(proxy_group)
        proxy_layout.setSpacing(10)
        
        # HTTP 代理
        self.http_proxy_input = QLineEdit()
        self.http_proxy_input.setPlaceholderText(tr("e.g., http://127.0.0.1:7890"))
        proxy_layout.addRow(tr("HTTP Proxy:"), self.http_proxy_input)
        
        # HTTPS 代理
        self.https_proxy_input = QLineEdit()
        self.https_proxy_input.setPlaceholderText(tr("e.g., http://127.0.0.1:7890"))
        proxy_layout.addRow(tr("HTTPS Proxy:"), self.https_proxy_input)
        
        # 不使用代理的地址
        self.no_proxy_input = QLineEdit()
        self.no_proxy_input.setPlaceholderText(tr("e.g., localhost,127.0.0.1,*.local"))
        proxy_layout.addRow(tr("No Proxy for:"), self.no_proxy_input)
        
        layout.addWidget(proxy_group)
        
        # 网络设置组
        network_group = QGroupBox(tr("Network Settings"))
        network_layout = QFormLayout(network_group)
        network_layout.setSpacing(10)
        
        # 超时设置
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 1800)
        self.timeout_spin.setSingleStep(30)
        self.timeout_spin.setSuffix(tr(" seconds"))
        network_layout.addRow(tr("Download Timeout:"), self.timeout_spin)
        
        # 重试次数
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(1, 10)
        network_layout.addRow(tr("Max Retries:"), self.retry_spin)
        
        layout.addWidget(network_group)
        
        # 快速设置按钮
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel(tr("Quick Setup:")))
        
        clash_btn = QPushButton(tr("Clash (7890)"))
        clash_btn.clicked.connect(self._set_clash_proxy)
        quick_layout.addWidget(clash_btn)
        
        v2ray_btn = QPushButton(tr("v2rayN (10808)"))
        v2ray_btn.clicked.connect(self._set_v2ray_proxy)
        quick_layout.addWidget(v2ray_btn)
        
        ss_btn = QPushButton(tr("SS (1080)"))
        ss_btn.clicked.connect(self._set_ss_proxy)
        quick_layout.addWidget(ss_btn)
        
        quick_layout.addStretch()
        layout.addLayout(quick_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.test_btn = QPushButton(tr("Test Connection"))
        self.test_btn.clicked.connect(self._test_connection)
        button_layout.addWidget(self.test_btn)
        
        self.clear_btn = QPushButton(tr("Clear"))
        self.clear_btn.clicked.connect(self._clear_settings)
        button_layout.addWidget(self.clear_btn)
        
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton(tr("Save"))
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
    
    def _load_settings(self):
        """加载当前设置。"""
        proxy_config = self._network_manager.proxy_config
        
        self.http_proxy_input.setText(proxy_config.http_proxy or "")
        self.https_proxy_input.setText(proxy_config.https_proxy or "")
        self.no_proxy_input.setText(proxy_config.no_proxy or "")
        
        # 从配置加载超时和重试设置
        try:
            from core.config_provider import get_config
            config = get_config()
            timeout = config.network.get_timeout()
            retries = config.network.get_max_retries()
            self.timeout_spin.setValue(timeout)
            self.retry_spin.setValue(retries)
        except Exception:
            self.timeout_spin.setValue(300)
            self.retry_spin.setValue(3)
    
    def _set_clash_proxy(self):
        """设置 Clash 默认代理。"""
        self.http_proxy_input.setText("http://127.0.0.1:7890")
        self.https_proxy_input.setText("http://127.0.0.1:7890")
    
    def _set_v2ray_proxy(self):
        """设置 v2rayN 默认代理。"""
        self.http_proxy_input.setText("http://127.0.0.1:10808")
        self.https_proxy_input.setText("http://127.0.0.1:10808")
    
    def _set_ss_proxy(self):
        """设置 Shadowsocks 默认代理。"""
        self.http_proxy_input.setText("http://127.0.0.1:1080")
        self.https_proxy_input.setText("http://127.0.0.1:1080")
    
    def _clear_settings(self):
        """清除所有代理设置。"""
        self.http_proxy_input.clear()
        self.https_proxy_input.clear()
        self.no_proxy_input.clear()
        self.timeout_spin.setValue(300)
        self.retry_spin.setValue(3)
    
    def _test_connection(self):
        """测试代理连接。"""
        import urllib.request
        import ssl
        
        http_proxy = self.http_proxy_input.text().strip()
        https_proxy = self.https_proxy_input.text().strip()
        
        proxy_config = ProxyConfig(
            http_proxy=http_proxy or None,
            https_proxy=https_proxy or None,
        )
        
        # 测试 GitHub 连接
        test_url = "https://github.com"
        
        try:
            from core.network import build_proxy_handler
            opener = build_proxy_handler(proxy_config)
            
            request = urllib.request.Request(
                test_url,
                headers={"User-Agent": "Mozilla/5.0"},
                method="HEAD"
            )
            
            with opener.open(request, timeout=10) as response:
                if response.status == 200:
                    QMessageBox.information(
                        self,
                        tr("Connection Test"),
                        tr("Successfully connected to GitHub!\nProxy is working correctly.")
                    )
                else:
                    QMessageBox.warning(
                        self,
                        tr("Connection Test"),
                        tr(f"Received status {response.status} from GitHub.")
                    )
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("Connection Test Failed"),
                tr(f"Failed to connect to GitHub:\n{str(e)}")
            )
    
    def _save_settings(self):
        """保存设置。"""
        http_proxy = self.http_proxy_input.text().strip()
        https_proxy = self.https_proxy_input.text().strip()
        no_proxy = self.no_proxy_input.text().strip()
        timeout = self.timeout_spin.value()
        retries = self.retry_spin.value()
        
        # 更新网络管理器
        self._network_manager.set_proxy(
            http_proxy=http_proxy or None,
            https_proxy=https_proxy or None,
            no_proxy=no_proxy or None,
        )
        
        # 保存超时和重试设置
        try:
            from core.config_provider import get_config
            config = get_config()
            config.network.set_timeout(timeout)
            config.network.set_max_retries(retries)
        except Exception as e:
            QMessageBox.warning(
                self,
                tr("Warning"),
                tr(f"Failed to save network settings: {e}")
            )
        
        QMessageBox.information(
            self,
            tr("Settings Saved"),
            tr("Proxy settings have been saved successfully.")
        )
        
        self.accept()
