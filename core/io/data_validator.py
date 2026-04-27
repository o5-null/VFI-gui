"""Data validation and transformation utilities."""

from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

from loguru import logger


T = TypeVar('T')


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    messages: List[tuple[ValidationLevel, str]]

    def __init__(self):
        self.is_valid = True
        self.messages = []

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.is_valid = False
        self.messages.append((ValidationLevel.ERROR, message))

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.messages.append((ValidationLevel.WARNING, message))

    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.messages.append((ValidationLevel.INFO, message))

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one."""
        self.is_valid = self.is_valid and other.is_valid
        self.messages.extend(other.messages)
        return self


class FieldValidator:
    """Validator for individual fields."""

    def __init__(
        self,
        name: str,
        required: bool = False,
        field_type: Optional[type] = None,
        default: Any = None,
        choices: Optional[List[Any]] = None,
        custom_validator: Optional[Callable[[Any], Optional[str]]] = None,
    ):
        """Initialize field validator.

        Args:
            name: Field name
            required: Whether field is required
            field_type: Expected type
            default: Default value
            choices: Allowed values
            custom_validator: Custom validation function
        """
        self.name = name
        self.required = required
        self.field_type = field_type
        self.default = default
        self.choices = choices
        self.custom_validator = custom_validator

    def validate(self, value: Any) -> tuple[bool, Any, Optional[str]]:
        """Validate a field value.

        Returns:
            Tuple of (is_valid, processed_value, error_message)
        """
        # Check required
        if value is None:
            if self.required:
                return False, None, f"Required field '{self.name}' is missing"
            return True, self.default, None

        # Check type
        if self.field_type and not isinstance(value, self.field_type):
            try:
                value = self.field_type(value)
            except (ValueError, TypeError):
                return False, value, f"Field '{self.name}' must be of type {self.field_type.__name__}"

        # Check choices
        if self.choices and value not in self.choices:
            return False, value, f"Field '{self.name}' must be one of {self.choices}"

        # Custom validation
        if self.custom_validator:
            error = self.custom_validator(value)
            if error:
                return False, value, error

        return True, value, None


class SchemaValidator:
    """Schema-based data validator."""

    def __init__(self, fields: List[FieldValidator]):
        """Initialize schema validator.

        Args:
            fields: List of field validators
        """
        self.fields = {f.name: f for f in fields}

    def validate(self, data: Dict[str, Any]) -> tuple[ValidationResult, Dict[str, Any]]:
        """Validate data against schema.

        Args:
            data: Data to validate

        Returns:
            Tuple of (validation_result, processed_data)
        """
        result = ValidationResult()
        processed = {}

        # Validate known fields
        for name, validator in self.fields.items():
            value = data.get(name)
            is_valid, processed_value, error = validator.validate(value)

            if is_valid:
                processed[name] = processed_value
            else:
                result.add_error(error)

        # Check for unknown fields
        unknown_fields = set(data.keys()) - set(self.fields.keys())
        for field in unknown_fields:
            result.add_warning(f"Unknown field '{field}' will be ignored")

        return result, processed


class DataTransformer:
    """Data transformation utilities."""

    @staticmethod
    def flatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary.

        Args:
            data: Nested dictionary
            separator: Key separator

        Returns:
            Flattened dictionary
        """
        result = {}

        def _flatten(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}{separator}{key}" if prefix else key
                    _flatten(value, new_key)
            else:
                result[prefix] = obj

        _flatten(data)
        return result

    @staticmethod
    def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """Unflatten dictionary with dot-notation keys.

        Args:
            data: Flat dictionary with dot-notation keys
            separator: Key separator

        Returns:
            Nested dictionary
        """
        result = {}

        for key, value in data.items():
            parts = key.split(separator)
            current = result

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return result

    @staticmethod
    def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DataTransformer.merge_dicts(result[key], value)
            else:
                result[key] = value

        return result


class DataValidator:
    """Main data validation class."""

    def __init__(self):
        self._schemas: Dict[str, SchemaValidator] = {}
        self._transformer = DataTransformer()

    def register_schema(self, name: str, schema: SchemaValidator) -> None:
        """Register a validation schema.

        Args:
            name: Schema name
            schema: Schema validator
        """
        self._schemas[name] = schema
        logger.debug(f"Registered schema: {name}")

    def validate(self, data: Dict[str, Any], schema_name: str) -> tuple[ValidationResult, Dict[str, Any]]:
        """Validate data against a registered schema.

        Args:
            data: Data to validate
            schema_name: Name of registered schema

        Returns:
            Tuple of (validation_result, processed_data)
        """
        if schema_name not in self._schemas:
            raise ValueError(f"Unknown schema: {schema_name}")

        return self._schemas[schema_name].validate(data)

    def transform(
        self,
        data: Dict[str, Any],
        operation: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Transform data.

        Args:
            data: Data to transform
            operation: Transformation operation ('flatten', 'unflatten', 'merge')
            **kwargs: Additional arguments for transformation

        Returns:
            Transformed data
        """
        if operation == "flatten":
            return self._transformer.flatten_dict(data, kwargs.get("separator", "."))
        elif operation == "unflatten":
            return self._transformer.unflatten_dict(data, kwargs.get("separator", "."))
        elif operation == "merge":
            other = kwargs.get("other", {})
            return self._transformer.merge_dicts(data, other)
        else:
            raise ValueError(f"Unknown transformation: {operation}")
