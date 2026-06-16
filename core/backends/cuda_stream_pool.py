"""CUDA Stream Pool for single-model multi-stream parallel inference.

This module provides a pool of CUDA streams for parallel inference on a single
model. Multiple inference requests can be processed concurrently using different
CUDA streams while sharing the same model weights.

Architecture:
    - Single model instance (one copy of weights)
    - Multiple CUDA streams for parallel execution
    - Thread-safe round-robin stream selection
    - Zero IO principle: only works with tensor data, no file handling

Usage:
    pool = CUDAStreamPool(model, num_streams=4, device="cuda:0")
    result = pool.infer(request)
    pool.cleanup()
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, List, Optional

import torch

from core.types import InferenceRequest, InferenceResult


class CUDAStreamPool:
    """Pool of CUDA streams for parallel inference on a single model.

    Provides efficient parallel inference by using multiple CUDA streams
    with a single model instance. Each stream can execute inference
    concurrently while sharing the same model weights.

    Attributes:
        _model: The model instance for inference
        _num_streams: Number of CUDA streams in the pool
        _streams: List of CUDA stream objects
        _device: Target device for inference
        _stream_index: Current stream index for round-robin selection
        _index_lock: Lock for thread-safe stream index management
    """

    def __init__(
        self,
        model: Any,
        num_streams: int = 4,
        device: str = "cuda:0",
    ) -> None:
        """Initialize the CUDA stream pool.

        Args:
            model: Model instance with interpolate() method.
                   Must accept: model.interpolate(frame0, frame1, timestep=timestep)
                   Must return: torch.Tensor [C, H, W]
            num_streams: Number of CUDA streams to create (default: 4)
            device: Target device string (default: "cuda:0")

        Raises:
            RuntimeError: If CUDA is not available
            ValueError: If num_streams is less than 1
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for CUDAStreamPool")

        if num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {num_streams}")

        self._model = model
        self._num_streams = num_streams
        self._device = torch.device(device)

        # Create CUDA streams
        self._streams: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device=self._device)
            for _ in range(num_streams)
        ]

        # Round-robin stream selection with thread-safe index
        self._stream_index: int = 0
        self._index_lock: Lock = Lock()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute single inference request using a CUDA stream.

        Selects a stream in round-robin fashion and executes inference
        within that stream's context. Moves output to CPU and converts
        fp16 to float32 before returning.

        Args:
            request: InferenceRequest containing:
                - frame0: First input frame tensor [C, H, W] float32
                - frame1: Second input frame tensor [C, H, W] float32
                - timestep: Interpolation position in [0, 1]
                - model_config: Model-specific configuration (unused here)

        Returns:
            InferenceResult containing:
                - output_frame: Interpolated frame tensor [C, H, W] float32 on CPU
                - success: Whether inference succeeded
                - error: Error message (on failure)
                - inference_time_ms: Wall-clock inference time in milliseconds
        """
        # Select stream via round-robin with thread-safe index
        with self._index_lock:
            stream_idx = self._stream_index
            self._stream_index = (self._stream_index + 1) % self._num_streams

        stream = self._streams[stream_idx]

        # Prepare input tensors on target device
        frame0 = request.frame0.to(self._device)
        frame1 = request.frame1.to(self._device)
        timestep = request.timestep

        start_time = time.perf_counter()

        try:
            # Execute inference in stream context
            with torch.cuda.stream(stream):
                with torch.inference_mode():
                    output = self._model.interpolate(
                        frame0,
                        frame1,
                        timestep=timestep,
                    )

            # Synchronize this stream to ensure output is ready
            stream.synchronize()

            # Move output to CPU
            output_cpu = output.cpu()

            # Convert fp16 to float32 if needed
            if output_cpu.dtype == torch.float16:
                output_cpu = output_cpu.float()

            inference_time_ms = (time.perf_counter() - start_time) * 1000

            return InferenceResult(
                output_frame=output_cpu,
                success=True,
                error=None,
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"CUDAStreamPool inference error: {str(e)}"

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=error_msg,
                inference_time_ms=inference_time_ms,
            )

    def infer_batch(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResult]:
        """Execute multiple inference requests.

        Processes each request sequentially using the stream pool.
        For true batch parallelism, consider using batch inference
        on the model directly.

        Args:
            requests: List of InferenceRequest objects

        Returns:
            List of InferenceResult objects, one per request
        """
        results: List[InferenceResult] = []
        for request in requests:
            result = self.infer(request)
            results.append(result)
        return results

    def synchronize_all(self) -> None:
        """Synchronize all streams in the pool.

        Ensures all pending operations on all streams are complete.
        Useful before switching contexts or cleanup.
        """
        for stream in self._streams:
            stream.synchronize()

    def cleanup(self) -> None:
        """Clean up resources and synchronize all streams.

        Synchronizes all streams to ensure pending operations complete
        before cleanup. This prevents potential errors when the model
        or device resources are released.
        """
        self.synchronize_all()

        # Clear stream references
        self._streams.clear()
        self._num_streams = 0

    def get_stream_count(self) -> int:
        """Get the number of streams in the pool.

        Returns:
            Number of CUDA streams
        """
        return self._num_streams

    def get_device(self) -> torch.device:
        """Get the target device for inference.

        Returns:
            torch.device instance
        """
        return self._device

    def __repr__(self) -> str:
        return (
            f"CUDAStreamPool("
            f"streams={self._num_streams}, "
            f"device={self._device})"
        )

    def __enter__(self) -> "CUDAStreamPool":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit - cleanup resources."""
        self.cleanup()


__all__ = ["CUDAStreamPool"]