"""Tests for core/types.py — enumerations, dataclasses, and type definitions."""

from __future__ import annotations

import torch

from core.types import (
    BackendType,
    InferenceRequest,
    InferenceResult,
    InferenceStrategy,
    ProcessedFrameData,
    SubTaskState,
    TaskState,
)


class TestEnums:
    """Enum correctness and value uniqueness."""

    def test_task_state_values(self):
        """TaskState covers the full lifecycle."""
        expected = {"pending", "loading", "processing", "completed", "failed", "cancelled"}
        actual = {s.value for s in TaskState}
        assert actual == expected

    def test_task_state_transitions(self):
        """State enum members are accessible by name."""
        assert TaskState.PENDING.value == "pending"
        assert TaskState.PROCESSING.value == "processing"
        assert TaskState.COMPLETED.value == "completed"

    def test_subtask_state_values(self):
        """SubTaskState covers scheduling states."""
        assert SubTaskState.PENDING.value == "pending"
        assert SubTaskState.RUNNING.value == "running"
        assert SubTaskState.COMPLETED.value == "completed"

    def test_backend_type_values(self):
        """BackendType covers all supported backends."""
        types = {b.value for b in BackendType}
        assert "torch" in types
        assert "tensorrt" in types
        assert "onnx" in types
        assert "ncnn" in types
        assert "directml" in types

    def test_inference_strategy_values(self):
        """InferenceStrategy covers all parallelism approaches."""
        strategies = {s.value for s in InferenceStrategy}
        assert "batch" in strategies
        assert "cuda_streams" in strategies
        assert "multi_model" in strategies
        assert "serial" in strategies


class TestDataclasses:
    """Dataclass construction and field defaults."""

    def test_inference_request_minimal(self):
        """InferenceRequest with required fields only."""
        f0 = torch.rand(3, 64, 64)
        f1 = torch.rand(3, 64, 64)
        req = InferenceRequest(
            frame0=f0,
            frame1=f1,
            timestep=0.5,
            model_config={},
        )
        assert req.timestep == 0.5
        assert req.model_config == {}

    def test_inference_request_full(self):
        """InferenceRequest with all fields."""
        f0 = torch.rand(3, 64, 64)
        f1 = torch.rand(3, 64, 64)
        req = InferenceRequest(
            frame0=f0,
            frame1=f1,
            timestep=0.25,
            model_config={"model_type": "film", "precision": "fp16"},
        )
        assert req.timestep == 0.25
        assert req.model_config["model_type"] == "film"

    def test_inference_result_success(self):
        """InferenceResult with successful inference."""
        frame = torch.rand(3, 64, 64)
        result = InferenceResult(
            output_frame=frame,
            success=True,
        )
        assert result.success is True
        assert result.error is None
        assert result.inference_time_ms == 0.0  # default

    def test_inference_result_failure(self):
        """InferenceResult with failure."""
        result = InferenceResult(
            output_frame=torch.zeros(0),
            success=False,
            error="CUDA OOM",
        )
        assert result.success is False
        assert result.error == "CUDA OOM"

    def test_processed_frame_data(self):
        """ProcessedFrameData holds output frame + metadata."""
        data = ProcessedFrameData(
            data=torch.rand(3, 64, 64),
            source_frame_idx=42,
        )
        assert data.source_frame_idx == 42
        assert data.interpolated is False

    def test_processed_frame_data_interpolated(self):
        """ProcessedFrameData with interpolation metadata."""
        data = ProcessedFrameData(
            data=torch.rand(3, 64, 64),
            source_frame_idx=1,
            interpolated=True,
            interpolation_ratio=0.5,
        )
        assert data.interpolated is True
        assert data.interpolation_ratio == 0.5
