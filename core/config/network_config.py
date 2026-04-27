"""Network-specific configuration."""

from typing import Optional

from core.config.base_config import BaseConfig


DEFAULT_NETWORK_SETTINGS = {
    "proxy": {
        "http_proxy": "",
        "https_proxy": "",
        "no_proxy": "",
    },
    "timeout": 300,
    "max_retries": 3,
}


class NetworkConfig(BaseConfig):
    """Configuration for network-related settings.
    
    Manages proxy settings, timeouts, and retry policies.
    """

    def _load_defaults(self) -> None:
        """Load default network settings."""
        self._settings = DEFAULT_NETWORK_SETTINGS.copy()

    def get_proxy_config(self) -> dict:
        """Get proxy configuration.
        
        Returns:
            Dictionary with http_proxy, https_proxy, no_proxy
        """
        return self._settings.get("proxy", DEFAULT_NETWORK_SETTINGS["proxy"]).copy()

    def set_proxy_config(
        self,
        http_proxy: str = "",
        https_proxy: str = "",
        no_proxy: str = "",
    ) -> None:
        """Set proxy configuration."""
        self._settings["proxy"] = {
            "http_proxy": http_proxy,
            "https_proxy": https_proxy,
            "no_proxy": no_proxy,
        }
        self.save()

    def get_timeout(self) -> int:
        """Get download timeout in seconds."""
        return self._settings.get("timeout", 300)

    def set_timeout(self, timeout: int) -> None:
        """Set download timeout."""
        self._settings["timeout"] = timeout
        self.save()

    def get_max_retries(self) -> int:
        """Get maximum retry attempts."""
        return self._settings.get("max_retries", 3)

    def set_max_retries(self, retries: int) -> None:
        """Set maximum retry attempts."""
        self._settings["max_retries"] = retries
        self.save()

    def get_proxies_dict(self) -> Optional[dict]:
        """Get proxies in requests format.
        
        Returns:
            Proxies dict for requests library or None if not configured
        """
        proxy_config = self.get_proxy_config()
        http_proxy = proxy_config.get("http_proxy", "")
        https_proxy = proxy_config.get("https_proxy", "")
        
        if not http_proxy and not https_proxy:
            return None
            
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
            
        return proxies if proxies else None
