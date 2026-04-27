"""统一网络通信模块，支持代理设置。

提供统一的 HTTP 请求接口，支持：
- 自动代理检测和配置
- 下载进度回调
- 重试机制
- 超时设置
"""

import os
import ssl
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

from loguru import logger


@dataclass
class ProxyConfig:
    """代理配置类。"""
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """从环境变量读取代理配置。"""
        return cls(
            http_proxy=os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
            https_proxy=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
            no_proxy=os.environ.get("NO_PROXY") or os.environ.get("no_proxy"),
        )

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "ProxyConfig":
        """从设置字典创建代理配置。"""
        return cls(
            http_proxy=settings.get("http_proxy"),
            https_proxy=settings.get("https_proxy"),
            no_proxy=settings.get("no_proxy"),
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        """转换为字典。"""
        return {
            "http_proxy": self.http_proxy,
            "https_proxy": self.https_proxy,
            "no_proxy": self.no_proxy,
        }

    def is_configured(self) -> bool:
        """检查是否配置了代理。"""
        return self.http_proxy is not None or self.https_proxy is not None


def get_default_proxy_config() -> ProxyConfig:
    """获取默认代理配置。
    
    优先从配置文件中读取，如果没有则使用环境变量。
    
    Returns:
        ProxyConfig 实例
    """
    try:
        from core.config_provider import get_config
        config = get_config()
        proxy_settings = config.network.get_proxy_config()
        if proxy_settings:
            return ProxyConfig.from_settings(proxy_settings)
    except Exception as e:
        logger.debug(f"Failed to load proxy config from settings: {e}")
    
    return ProxyConfig.from_env()


def create_ssl_context() -> ssl.SSLContext:
    """创建 SSL 上下文，允许 GitHub 等站点的 HTTPS 连接。"""
    context = ssl.create_default_context()
    return context


def build_proxy_handler(proxy_config: Optional[ProxyConfig] = None) -> urllib.request.OpenerDirector:
    """构建带代理支持的 URL 打开器。
    
    Args:
        proxy_config: 代理配置，为 None 则使用默认配置
        
    Returns:
        配置好的 OpenerDirector
    """
    if proxy_config is None:
        proxy_config = get_default_proxy_config()
    
    handlers = []
    
    # 配置代理
    proxies = {}
    if proxy_config.http_proxy:
        proxies["http"] = proxy_config.http_proxy
        logger.debug(f"Using HTTP proxy: {proxy_config.http_proxy}")
    if proxy_config.https_proxy:
        proxies["https"] = proxy_config.https_proxy
        logger.debug(f"Using HTTPS proxy: {proxy_config.https_proxy}")
    
    if proxies:
        proxy_handler = urllib.request.ProxyHandler(proxies)
        handlers.append(proxy_handler)
    
    # HTTPS 处理器
    https_handler = urllib.request.HTTPSHandler(context=create_ssl_context())
    handlers.append(https_handler)
    
    opener = urllib.request.build_opener(*handlers)
    return opener


def download_file(
    url: str,
    target_path: Path,
    proxy_config: Optional[ProxyConfig] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    timeout: int = 300,
    chunk_size: int = 8192,
    headers: Optional[Dict[str, str]] = None,
) -> Path:
    """下载文件到指定路径。
    
    Args:
        url: 下载 URL
        target_path: 保存路径
        proxy_config: 代理配置，为 None 则使用默认配置
        progress_callback: 进度回调函数，参数为 (block_num, block_size, total_size)
        timeout: 超时时间（秒）
        chunk_size: 下载块大小
        headers: 额外的请求头
        
    Returns:
        保存的文件路径
        
    Raises:
        urllib.error.URLError: 网络错误
        IOError: 文件写入错误
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建请求
    request = urllib.request.Request(url)
    
    # 添加默认 User-Agent
    request.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0")
    
    # 添加自定义请求头
    if headers:
        for key, value in headers.items():
            request.add_header(key, value)
    
    # 获取 opener
    opener = build_proxy_handler(proxy_config)
    
    logger.info(f"Downloading: {url}")
    logger.debug(f"Target: {target_path}")
    if proxy_config and proxy_config.is_configured():
        logger.debug(f"Using proxy for download")
    
    try:
        with opener.open(request, timeout=timeout) as response:
            total_size = response.headers.get("Content-Length")
            total_size = int(total_size) if total_size else -1
            
            downloaded = 0
            block_num = 0
            
            with open(target_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    block_num += 1
                    
                    if progress_callback:
                        progress_callback(block_num, chunk_size, total_size)
        
        logger.info(f"Downloaded: {target_path} ({downloaded} bytes)")
        return target_path
        
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error {e.code}: {e.reason}")
        raise
    except urllib.error.URLError as e:
        logger.error(f"URL error: {e.reason}")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def download_with_retry(
    urls: List[str],
    target_path: Path,
    proxy_config: Optional[ProxyConfig] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    timeout: int = 300,
    max_retries: int = 3,
) -> Path:
    """尝试从多个 URL 下载，支持重试。
    
    Args:
        urls: URL 列表，按优先级排序
        target_path: 保存路径
        proxy_config: 代理配置
        progress_callback: 进度回调
        timeout: 超时时间
        max_retries: 每个 URL 的最大重试次数
        
    Returns:
        保存的文件路径
        
    Raises:
        RuntimeError: 所有 URL 都下载失败
    """
    errors = []
    
    for url in urls:
        for attempt in range(max_retries):
            try:
                return download_file(
                    url,
                    target_path,
                    proxy_config=proxy_config,
                    progress_callback=progress_callback,
                    timeout=timeout,
                )
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 * (attempt + 1))  # 指数退避
    
    error_summary = "\n".join(errors)
    raise RuntimeError(f"Failed to download from all URLs:\n{error_summary}")


def urlretrieve_with_proxy(
    url: str,
    filename: Path,
    proxy_config: Optional[ProxyConfig] = None,
    reporthook: Optional[Callable[[int, int, int], None]] = None,
    timeout: int = 300,
) -> Path:
    """兼容 urllib.request.urlretrieve 接口的下载函数。
    
    Args:
        url: 下载 URL
        filename: 保存路径
        proxy_config: 代理配置
        reporthook: 进度回调，与 urlretrieve 接口兼容
        timeout: 超时时间
        
    Returns:
        保存的文件路径
    """
    return download_file(
        url,
        filename,
        proxy_config=proxy_config,
        progress_callback=reporthook,
        timeout=timeout,
    )


class NetworkManager:
    """网络管理器，提供统一的网络请求管理。"""
    
    _instance: Optional["NetworkManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._proxy_config: Optional[ProxyConfig] = None
        self._load_proxy_config()
    
    def _load_proxy_config(self):
        """从配置加载代理设置。"""
        try:
            from core.config_provider import get_config
            config = get_config()
            proxy_settings = config.network.get_proxy_config()
            if proxy_settings:
                self._proxy_config = ProxyConfig.from_settings(proxy_settings)
                logger.info(f"Loaded proxy config from settings")
            else:
                self._proxy_config = ProxyConfig.from_env()
                if self._proxy_config.is_configured():
                    logger.info(f"Loaded proxy config from environment")
        except Exception as e:
            logger.debug(f"Failed to load proxy config: {e}")
            self._proxy_config = ProxyConfig()
    
    @property
    def proxy_config(self) -> ProxyConfig:
        """获取当前代理配置。"""
        if self._proxy_config is None:
            self._proxy_config = ProxyConfig()
        return self._proxy_config
    
    def set_proxy(self, http_proxy: Optional[str] = None, https_proxy: Optional[str] = None, no_proxy: Optional[str] = None):
        """设置代理。
        
        Args:
            http_proxy: HTTP 代理地址
            https_proxy: HTTPS 代理地址
            no_proxy: 不使用代理的地址列表
        """
        self._proxy_config = ProxyConfig(
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            no_proxy=no_proxy,
        )
        
        # 保存到配置
        try:
            from core.config_provider import get_config
            config = get_config()
            proxy_dict = self._proxy_config.to_dict()
            config.network.set_proxy_config(
                http_proxy=proxy_dict.get("http_proxy", ""),
                https_proxy=proxy_dict.get("https_proxy", ""),
                no_proxy=proxy_dict.get("no_proxy", ""),
            )
            logger.info(f"Saved proxy config to settings")
        except Exception as e:
            logger.debug(f"Failed to save proxy config: {e}")
    
    def clear_proxy(self):
        """清除代理设置。"""
        self._proxy_config = ProxyConfig()
        try:
            from core.config_provider import get_config
            config = get_config()
            config.network.set_proxy_config("", "", "")
            logger.info(f"Cleared proxy config")
        except Exception as e:
            logger.debug(f"Failed to clear proxy config: {e}")
    
    def download(
        self,
        url: str,
        target_path: Path,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        timeout: int = 300,
    ) -> Path:
        """下载文件。
        
        Args:
            url: 下载 URL
            target_path: 保存路径
            progress_callback: 进度回调
            timeout: 超时时间
            
        Returns:
            保存的文件路径
        """
        return download_file(
            url,
            target_path,
            proxy_config=self._proxy_config,
            progress_callback=progress_callback,
            timeout=timeout,
        )
    
    def download_multiple(
        self,
        urls: List[str],
        target_path: Path,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        timeout: int = 300,
    ) -> Path:
        """从多个 URL 尝试下载。
        
        Args:
            urls: URL 列表
            target_path: 保存路径
            progress_callback: 进度回调
            timeout: 超时时间
            
        Returns:
            保存的文件路径
        """
        return download_with_retry(
            urls,
            target_path,
            proxy_config=self._proxy_config,
            progress_callback=progress_callback,
            timeout=timeout,
        )


def get_network_manager() -> NetworkManager:
    """获取网络管理器单例。"""
    return NetworkManager()
