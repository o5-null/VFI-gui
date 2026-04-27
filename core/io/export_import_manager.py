"""Centralized export/import manager for VFI-gui.

This module provides a unified interface for all data export and import operations,
including configuration, presets, batch queues, and processing results.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from loguru import logger

from core.io.serializers import BaseSerializer, SerializerFactory, JsonSerializer
from core.io.async_io import AsyncFileHandler
from core.io.data_validator import DataValidator, ValidationResult, SchemaValidator, FieldValidator


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


class ExportType(Enum):
    """Types of data that can be exported."""
    CONFIG = "config"
    PRESET = "preset"
    QUEUE = "queue"
    RESULTS = "results"
    BACKUP = "backup"


@dataclass
class ExportMetadata:
    """Metadata for exported data."""
    version: str = "1.0"
    export_type: str = ""
    export_date: str = ""
    description: str = ""
    source_app: str = "VFI-gui"

    def __post_init__(self):
        if not self.export_date:
            self.export_date = datetime.now().isoformat()


@dataclass
class ExportOptions:
    """Options for export operations."""
    format: ExportFormat = ExportFormat.JSON
    indent: int = 2
    include_metadata: bool = True
    compress: bool = False
    validate: bool = True
    async_save: bool = False
    debounce_delay: float = 0.0


class ExportImportManager:
    """Centralized manager for all export/import operations.

    This class provides:
    - Unified interface for saving/loading different data types
    - Async IO support for non-blocking operations
    - Data validation and transformation
    - Debounced saves to reduce disk writes
    - Batch operations for multiple files

    Example:
        >>> manager = ExportImportManager()
        >>> # Export configuration
        >>> manager.export_config(config_data, Path("config.json"))
        >>> # Import with validation
        >>> data = manager.import_data(Path("preset.yaml"), validate=True)
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize export/import manager.

        Args:
            base_path: Base directory for exports/imports
        """
        self.base_path = base_path or Path.cwd()
        self._async_handler = AsyncFileHandler()
        self._validator = DataValidator()
        self._pending_saves: Dict[str, Dict[str, Any]] = {}
        self._schemas_registered = False

        self._register_default_schemas()

    def _register_default_schemas(self) -> None:
        """Register default validation schemas."""
        if self._schemas_registered:
            return

        # Config schema
        config_schema = SchemaValidator([
            FieldValidator("pipeline", required=False, field_type=dict),
            FieldValidator("output", required=False, field_type=dict),
            FieldValidator("vapoursynth", required=False, field_type=dict),
            FieldValidator("paths", required=False, field_type=dict),
            FieldValidator("ui", required=False, field_type=dict),
        ])
        self._validator.register_schema("config", config_schema)

        # Preset schema
        preset_schema = SchemaValidator([
            FieldValidator("name", required=True, field_type=str),
            FieldValidator("description", required=False, field_type=str),
            FieldValidator("settings", required=True, field_type=dict),
            FieldValidator("category", required=False, field_type=str, default="general"),
        ])
        self._validator.register_schema("preset", preset_schema)

        # Queue schema
        queue_schema = SchemaValidator([
            FieldValidator("items", required=True, field_type=list),
            FieldValidator("settings", required=False, field_type=dict),
        ])
        self._validator.register_schema("queue", queue_schema)

        self._schemas_registered = True
        logger.debug("Default schemas registered")

    def export(
        self,
        data: Dict[str, Any],
        filepath: Union[str, Path],
        options: Optional[ExportOptions] = None,
        schema_name: Optional[str] = None,
    ) -> bool:
        """Export data to file.

        Args:
            data: Data to export
            filepath: Target file path
            options: Export options
            schema_name: Optional schema for validation

        Returns:
            True if export succeeded
        """
        options = options or ExportOptions()
        filepath = Path(filepath)

        try:
            # Validate if requested
            if options.validate and schema_name:
                result, processed_data = self._validator.validate(data, schema_name)
                if not result.is_valid:
                    for level, msg in result.messages:
                        logger.warning(f"Validation {level.value}: {msg}")
                    if any(l == level for l, _ in result.messages if l.value == "error"):
                        return False
                data = processed_data

            # Add metadata if requested
            if options.include_metadata:
                data = self._add_metadata(data, filepath)

            # Get serializer
            serializer = SerializerFactory.get_serializer(filepath)

            # Debounced async save
            if options.debounce_delay > 0:
                self._async_handler.debounced_save(
                    data, filepath, options.debounce_delay, serializer, options.indent
                )
                return True

            # Regular async save
            if options.async_save:
                asyncio.create_task(self._async_handler.save(data, filepath, serializer, options.indent))
                return True

            # Synchronous save
            serializer.save(data, filepath, options.indent)
            logger.info(f"Exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def import_data(
        self,
        filepath: Union[str, Path],
        validate: bool = False,
        schema_name: Optional[str] = None,
        strip_metadata: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Import data from file.

        Args:
            filepath: Source file path
            validate: Whether to validate imported data
            schema_name: Schema name for validation
            strip_metadata: Whether to remove metadata fields

        Returns:
            Imported data or None if failed
        """
        filepath = Path(filepath)

        try:
            serializer = SerializerFactory.get_serializer(filepath)
            data = serializer.load(filepath)

            # Strip metadata if present
            if strip_metadata and "_metadata" in data:
                data = {k: v for k, v in data.items() if k != "_metadata"}

            # Validate if requested
            if validate and schema_name:
                result, processed_data = self._validator.validate(data, schema_name)
                if not result.is_valid:
                    for level, msg in result.messages:
                        logger.warning(f"Validation {level.value}: {msg}")
                    if any(l == level for l, _ in result.messages if l.value == "error"):
                        return None
                data = processed_data

            logger.info(f"Imported from {filepath}")
            return data

        except FileNotFoundError:
            logger.debug(f"Config file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None

    async def export_async(
        self,
        data: Dict[str, Any],
        filepath: Union[str, Path],
        options: Optional[ExportOptions] = None,
    ) -> bool:
        """Asynchronously export data to file.

        Args:
            data: Data to export
            filepath: Target file path
            options: Export options

        Returns:
            True if export succeeded
        """
        options = options or ExportOptions()
        filepath = Path(filepath)

        try:
            if options.include_metadata:
                data = self._add_metadata(data, filepath)

            serializer = SerializerFactory.get_serializer(filepath)
            await self._async_handler.save(data, filepath, serializer, options.indent)
            logger.info(f"Async exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Async export failed: {e}")
            return False

    async def import_async(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Asynchronously import data from file.

        Args:
            filepath: Source file path

        Returns:
            Imported data or None if failed
        """
        filepath = Path(filepath)

        try:
            serializer = SerializerFactory.get_serializer(filepath)
            data = await self._async_handler.load(filepath, serializer)
            logger.info(f"Async imported from {filepath}")
            return data

        except Exception as e:
            logger.error(f"Async import failed: {e}")
            return None

    def export_batch(
        self,
        items: List[tuple[Dict[str, Any], Path]],
        options: Optional[ExportOptions] = None,
    ) -> List[tuple[Path, bool]]:
        """Export multiple items.

        Args:
            items: List of (data, filepath) tuples
            options: Export options

        Returns:
            List of (filepath, success) tuples
        """
        options = options or ExportOptions()
        results = []

        for data, filepath in items:
            success = self.export(data, filepath, options)
            results.append((filepath, success))

        return results

    async def export_batch_async(
        self,
        items: List[tuple[Dict[str, Any], Path]],
        options: Optional[ExportOptions] = None,
    ) -> List[tuple[Path, bool]]:
        """Asynchronously export multiple items.

        Args:
            items: List of (data, filepath) tuples
            options: Export options

        Returns:
            List of (filepath, success) tuples
        """
        options = options or ExportOptions()
        tasks = []

        for data, filepath in items:
            task = self.export_async(data, filepath, options)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return [(filepath, success) for (_, filepath), success in zip(items, results)]

    def export_config(
        self,
        config_data: Dict[str, Any],
        filepath: Optional[Union[str, Path]] = None,
        options: Optional[ExportOptions] = None,
    ) -> bool:
        """Export configuration data.

        Args:
            config_data: Configuration dictionary
            filepath: Target file path (defaults to base_path/config.json)
            options: Export options

        Returns:
            True if export succeeded
        """
        if filepath is None:
            filepath = self.base_path / "config.json"

        options = options or ExportOptions(
            format=ExportFormat.JSON,
            validate=True,
            debounce_delay=0.5,
        )

        return self.export(config_data, filepath, options, schema_name="config")

    def import_config(
        self,
        filepath: Optional[Union[str, Path]] = None,
        validate: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Import configuration data.

        Args:
            filepath: Source file path (defaults to base_path/config.json)
            validate: Whether to validate imported data

        Returns:
            Configuration data or None if failed
        """
        if filepath is None:
            filepath = self.base_path / "config.json"

        return self.import_data(filepath, validate=validate, schema_name="config")

    def export_preset(
        self,
        preset_data: Dict[str, Any],
        filepath: Union[str, Path],
        options: Optional[ExportOptions] = None,
    ) -> bool:
        """Export a preset.

        Args:
            preset_data: Preset dictionary
            filepath: Target file path
            options: Export options

        Returns:
            True if export succeeded
        """
        options = options or ExportOptions(
            format=ExportFormat.YAML,
            validate=True,
        )

        return self.export(preset_data, filepath, options, schema_name="preset")

    def import_preset(
        self,
        filepath: Union[str, Path],
        validate: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Import a preset.

        Args:
            filepath: Source file path
            validate: Whether to validate imported data

        Returns:
            Preset data or None if failed
        """
        return self.import_data(filepath, validate=validate, schema_name="preset")

    def _add_metadata(
        self,
        data: Dict[str, Any],
        filepath: Path,
    ) -> Dict[str, Any]:
        """Add metadata to data.

        Args:
            data: Original data
            filepath: Target file path

        Returns:
            Data with metadata added
        """
        metadata = ExportMetadata(
            export_type=filepath.stem,
        )

        result = data.copy()
        result["_metadata"] = asdict(metadata)
        return result

    def register_schema(self, name: str, schema: SchemaValidator) -> None:
        """Register a custom validation schema.

        Args:
            name: Schema name
            schema: Schema validator
        """
        self._validator.register_schema(name, schema)

    def validate_data(
        self,
        data: Dict[str, Any],
        schema_name: str,
    ) -> ValidationResult:
        """Validate data against a schema.

        Args:
            data: Data to validate
            schema_name: Registered schema name

        Returns:
            Validation result
        """
        result, _ = self._validator.validate(data, schema_name)
        return result

    def shutdown(self) -> None:
        """Shutdown the manager and cleanup resources."""
        self._async_handler.shutdown()
        logger.debug("ExportImportManager shutdown complete")
