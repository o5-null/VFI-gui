"""Serializers for different data formats."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class BaseSerializer(ABC):
    """Abstract base class for data serializers."""

    @abstractmethod
    def serialize(self, data: Dict[str, Any], indent: Optional[int] = None) -> str:
        """Serialize data to string.

        Args:
            data: Data to serialize
            indent: Indentation level for pretty printing

        Returns:
            Serialized string
        """
        pass

    @abstractmethod
    def deserialize(self, content: str) -> Dict[str, Any]:
        """Deserialize string to data.

        Args:
            content: String content to deserialize

        Returns:
            Deserialized data
        """
        pass

    def save(self, data: Dict[str, Any], filepath: Path, indent: Optional[int] = None) -> None:
        """Save data to file.

        Args:
            data: Data to save
            filepath: Target file path
            indent: Indentation level for pretty printing
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        content = self.serialize(data, indent)
        filepath.write_text(content, encoding="utf-8")
        logger.debug(f"Data saved to {filepath}")

    def load(self, filepath: Path) -> Dict[str, Any]:
        """Load data from file.

        Args:
            filepath: Source file path

        Returns:
            Loaded data
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = filepath.read_text(encoding="utf-8")
        return self.deserialize(content)


class JsonSerializer(BaseSerializer):
    """JSON format serializer."""

    def serialize(self, data: Dict[str, Any], indent: Optional[int] = 2) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)

    def deserialize(self, content: str) -> Dict[str, Any]:
        """Deserialize JSON string to data."""
        return json.loads(content)


class YamlSerializer(BaseSerializer):
    """YAML format serializer."""

    def __init__(self):
        try:
            import yaml
            self._yaml = yaml
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("PyYAML not installed, YAML support disabled")

    def serialize(self, data: Dict[str, Any], indent: Optional[int] = 2) -> str:
        """Serialize data to YAML string."""
        if not self._available:
            raise ImportError("PyYAML is required for YAML serialization")

        # YAML doesn't use indent parameter like JSON
        return self._yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def deserialize(self, content: str) -> Dict[str, Any]:
        """Deserialize YAML string to data."""
        if not self._available:
            raise ImportError("PyYAML is required for YAML deserialization")

        return self._yaml.safe_load(content) or {}


class TomlSerializer(BaseSerializer):
    """TOML format serializer."""

    def __init__(self):
        try:
            import tomli
            import tomli_w
            self._tomli = tomli
            self._tomli_w = tomli_w
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("tomli/tomli_w not installed, TOML support disabled")

    def serialize(self, data: Dict[str, Any], indent: Optional[int] = None) -> str:
        """Serialize data to TOML string."""
        if not self._available:
            raise ImportError("tomli_w is required for TOML serialization")

        return self._tomli_w.dumps(data)

    def deserialize(self, content: str) -> Dict[str, Any]:
        """Deserialize TOML string to data."""
        if not self._available:
            raise ImportError("tomli is required for TOML deserialization")

        return self._tomli.loads(content.encode("utf-8"))


class SerializerFactory:
    """Factory for creating serializers based on file extension."""

    _serializers = {
        ".json": JsonSerializer,
        ".yaml": YamlSerializer,
        ".yml": YamlSerializer,
        ".toml": TomlSerializer,
    }

    @classmethod
    def get_serializer(cls, filepath: Path) -> BaseSerializer:
        """Get appropriate serializer for file path.

        Args:
            filepath: File path to determine serializer for

        Returns:
            Serializer instance
        """
        ext = filepath.suffix.lower()
        serializer_class = cls._serializers.get(ext, JsonSerializer)
        return serializer_class()

    @classmethod
    def register_serializer(cls, extension: str, serializer_class: type) -> None:
        """Register a new serializer for an extension.

        Args:
            extension: File extension (e.g., '.json')
            serializer_class: Serializer class to register
        """
        cls._serializers[extension.lower()] = serializer_class
