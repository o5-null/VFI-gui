"""Tests for core/result_validator.py — inference result validation."""

from __future__ import annotations

import torch

from core.result_validator import ResultValidator
from core.types import InferenceResult


def _make_result(output_frame: torch.Tensor, success: bool = True, error: str = "") -> InferenceResult:
    """Create an InferenceResult for testing."""
    return InferenceResult(
        output_frame=output_frame,
        success=success,
        error=error or None,
    )


class TestResultValidator:
    """ResultValidator check correctness."""

    def setup_method(self):
        self.validator = ResultValidator()

    def test_valid_float32_result(self):
        """A clean float32 result passes all checks."""
        frame = torch.rand(3, 64, 64, dtype=torch.float32)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is True
        assert validation.error is None

    def test_valid_float16_result(self):
        """Float16 inference output is accepted."""
        frame = torch.rand(3, 64, 64, dtype=torch.float16)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is True

    def test_failed_result_invalid(self):
        """A result with success=False is immediately invalid."""
        result = _make_result(torch.zeros(0), success=False, error="OOM")
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is False
        assert validation.error == "OOM"

    def test_nan_values_rejected(self):
        """Result containing NaN fails validation."""
        frame = torch.full((3, 64, 64), float("nan"), dtype=torch.float32)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is False

    def test_inf_values_rejected(self):
        """Result containing Inf fails validation."""
        frame = torch.full((3, 64, 64), float("inf"), dtype=torch.float32)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is False

    def test_uint8_dtype_rejected(self):
        """Result with uint8 dtype fails (expects float32/16)."""
        frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is False

    def test_empty_frame_rejected(self):
        """Empty frame (0-dim) fails shape check."""
        frame = torch.zeros(0)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is False

    def test_value_range_exceeded(self):
        """Frame values significantly outside [0,1] are rejected."""
        frame = torch.full((3, 64, 64), 2.0, dtype=torch.float32)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is False

    def test_value_range_edge_allowed(self):
        """Frame values at exactly 1.0 are accepted."""
        frame = torch.ones(3, 64, 64, dtype=torch.float32)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is True

    def test_value_range_negative_allowed(self):
        """Slightly negative values (-0.05) are within tolerance."""
        frame = torch.full((3, 64, 64), -0.05, dtype=torch.float32)
        result = _make_result(frame)
        validation = self.validator.validate(None, result)  # type: ignore[arg-type]
        assert validation.valid is True
