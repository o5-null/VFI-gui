"""Subprocess inference backend: stdin/stdout JSON communication.

This backend executes inference in a separate process, communicating via
stdin/stdout pipes with JSON messages. Frame data is encoded as base64
strings for transmission.

Use cases:
    - ncnn-vulkan: External executable, natural subprocess
    - VapourSynth: vspipe runs as independent process

Architecture:
    SubProcessBackend(BaseBackend) → spawns and manages a child process
    - Sends inference requests as JSON + base64 frame data via stdin
    - Receives inference results as JSON + base64 frame data via stdout
    - GPU memory isolation (separate process space)
    - Crash isolation (child crash doesn't kill parent)

Communication protocol:
    Request:  {"type": "infer", "request_id": "...", "frame0": "<base64>",
              "frame1": "<base64>", "timestep": 0.5, "model_config": {...}}
    Response: {"type": "result", "request_id": "...", "output_frame": "<base64>",
              "success": true, "inference_time_ms": 12.5}
    Control:  {"type": "load_model", "model_config": {...}}
              {"type": "unload_model"}
              {"type": "shutdown"}
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import time
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
from loguru import logger

from core.types import (
    BackendType,
    BackendConfig,
    ProcessingConfig,
    InferenceRequest,
    InferenceResult,
    EngineStatus,
)
from .base_backend import BaseBackend


@dataclass
class EngineMessage:
    """JSON message for subprocess communication.

    Attributes:
        type: Message type (infer, load_model, unload_model, shutdown, result, error)
        request_id: Unique request identifier for correlation
        frame0: Base64-encoded first frame data (for infer requests)
        frame1: Base64-encoded second frame data (for infer requests)
        timestep: Interpolation position [0, 1]
        model_config: Model-specific configuration
        output_frame: Base64-encoded output frame data (for results)
        success: Whether the operation succeeded
        error: Error message (on failure)
        inference_time_ms: Wall-clock inference time in milliseconds
    """

    type: str  # "infer" | "load_model" | "unload_model" | "shutdown" | "result" | "error"
    request_id: str = ""
    frame0: str = ""  # base64-encoded
    frame1: str = ""  # base64-encoded
    timestep: float = 0.5
    model_config: Dict[str, Any] = field(default_factory=dict)
    output_frame: str = ""  # base64-encoded
    success: bool = True
    error: str = ""
    inference_time_ms: float = 0.0

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = {}

    def to_json(self) -> str:
        """Serialize to JSON string (one line, newline-terminated)."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> EngineMessage:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _frame_to_base64(frame: Any) -> str:
    """Encode frame data to base64 string for transmission.

    Handles both numpy arrays and torch tensors.

    Args:
        frame: Frame data (numpy array or torch tensor)

    Returns:
        Base64-encoded string
    """
    if isinstance(frame, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, frame)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # Try torch tensor
    try:
        import torch
        if isinstance(frame, torch.Tensor):
            numpy_frame = frame.cpu().numpy()
            buf = io.BytesIO()
            np.save(buf, numpy_frame)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        pass

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _base64_to_tensor(b64_str: str) -> Any:
    """Decode base64 string to tensor.

    Returns numpy array or torch tensor depending on availability.

    Args:
        b64_str: Base64-encoded frame data

    Returns:
        Decoded frame as numpy array or torch tensor
    """
    raw = base64.b64decode(b64_str)
    buf = io.BytesIO(raw)
    numpy_frame = np.load(buf)

    try:
        import torch
        return torch.from_numpy(numpy_frame)
    except ImportError:
        return numpy_frame


class SubProcessBackend(BaseBackend):
    """Subprocess inference backend: stdin/stdout JSON communication.

    Spawns a child process for inference, communicating via JSON messages
    over stdin/stdout pipes. Frame data is encoded as base64 for
    serialization.

    Features:
        - GPU memory isolation (separate process space)
        - Crash isolation (child crash doesn't kill parent)
        - Supports external executables (ncnn-vulkan, vspipe)
        - Thread-safe message sending via lock

    Limitations:
        - IPC overhead (serialization/deserialization + base64)
        - Higher latency per frame than InProcessBackend
        - Frame data size limited by pipe buffer
    """

    # Backend metadata
    BACKEND_TYPE = BackendType.NCNN
    BACKEND_NAME = "SubProcess"
    BACKEND_DESCRIPTION = "Subprocess inference backend (stdin/stdout JSON)"

    # Supported features
    SUPPORTS_INTERPOLATION = True
    SUPPORTS_UPSCALING = False
    SUPPORTS_SCENE_DETECTION = False
    SUPPORTED_MODELS: Dict[str, List[str]] = {}

    def __init__(
        self,
        config: BackendConfig,
        parent: Optional[object] = None,
    ):
        """Initialize the subprocess backend.

        Args:
            config: Backend configuration (must include 'executable' in extra)
            parent: Parent object for signal handling
        """
        super().__init__(config, parent)
        self._process: Optional[subprocess.Popen] = None
        self._send_lock = threading.Lock()
        self._request_counter = 0
        self._pending_requests: Dict[str, EngineMessage] = {}
        self._engine_status: EngineStatus = EngineStatus.IDLE
        self._executable: str = config.extra.get("executable", "")

    @property
    def engine_status(self) -> EngineStatus:
        """Get the current engine lifecycle status."""
        return self._engine_status

    def initialize(self) -> bool:
        """Initialize the subprocess backend by spawning the child process.

        Returns:
            True if the child process started successfully
        """
        if not self._executable:
            logger.error("No executable specified in config.extra['executable']")
            self._engine_status = EngineStatus.ERROR
            return False

        self._engine_status = EngineStatus.LOADING
        try:
            self._process = subprocess.Popen(
                [self._executable],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered for real-time communication
            )

            if self._process.poll() is not None:
                stderr_output = ""
                if self._process.stderr:
                    stderr_output = self._process.stderr.read().decode(
                        errors="replace"
                    )
                logger.error(
                    f"Child process exited immediately: {stderr_output}"
                )
                self._engine_status = EngineStatus.ERROR
                return False

            self._is_initialized = True
            self._engine_status = EngineStatus.READY
            logger.info(
                f"SubProcessBackend initialized: executable={self._executable}, "
                f"pid={self._process.pid}"
            )
            return True

        except FileNotFoundError as e:
            logger.error(f"Executable not found: {self._executable}")
            self._engine_status = EngineStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to start subprocess: {e}")
            self._engine_status = EngineStatus.ERROR
            return False

    def load_model(self, model_config: Dict[str, Any]) -> bool:
        """Load model in the child process.

        Sends a load_model message to the child process and waits for
        acknowledgment.

        Args:
            model_config: Model configuration dict

        Returns:
            True if model loaded successfully
        """
        if not self._is_process_alive():
            logger.error("Child process not running")
            return False

        self._engine_status = EngineStatus.LOADING
        try:
            msg = EngineMessage(
                type="load_model",
                model_config=model_config,
            )
            response = self._send_message(msg)

            if response.success:
                self._engine_status = EngineStatus.READY
                logger.info(f"Model loaded in subprocess: {model_config.get('model_type', 'unknown')}")
                return True
            else:
                self._engine_status = EngineStatus.ERROR
                logger.error(f"Model load failed in subprocess: {response.error}")
                return False

        except Exception as e:
            self._engine_status = EngineStatus.ERROR
            logger.error(f"Model load communication error: {e}")
            return False

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Single frame-pair inference via subprocess pipe.

        Encodes frame data as base64, sends JSON message via stdin,
        and reads JSON response from stdout.

        Args:
            request: Inference request with frame data

        Returns:
            Inference result with interpolated frame
        """
        if not self._is_process_alive():
            return InferenceResult(
                output_frame=np.array([]),
                success=False,
                error="Child process not running",
            )

        if self._engine_status not in (EngineStatus.READY, EngineStatus.RUNNING):
            return InferenceResult(
                output_frame=np.array([]),
                success=False,
                error=f"Engine not ready: status={self._engine_status.value}",
            )

        self._engine_status = EngineStatus.RUNNING

        try:
            # Encode frame data to base64
            frame0_b64 = _frame_to_base64(request.frame0)
            frame1_b64 = _frame_to_base64(request.frame1)

            # Generate unique request ID
            self._request_counter += 1
            request_id = f"req_{self._request_counter}"

            # Send inference request
            msg = EngineMessage(
                type="infer",
                request_id=request_id,
                frame0=frame0_b64,
                frame1=frame1_b64,
                timestep=request.timestep,
                model_config=request.model_config,
            )

            response = self._send_message(msg)

            if response.success:
                # Decode output frame from base64
                output_frame = _base64_to_tensor(response.output_frame)

                self._engine_status = EngineStatus.READY
                return InferenceResult(
                    output_frame=output_frame,
                    success=True,
                    inference_time_ms=response.inference_time_ms,
                )
            else:
                self._engine_status = EngineStatus.READY
                return InferenceResult(
                    output_frame=np.array([]),
                    success=False,
                    error=response.error,
                )

        except Exception as e:
            self._engine_status = EngineStatus.ERROR
            logger.error(f"Subprocess inference error: {e}")
            return InferenceResult(
                output_frame=np.array([]),
                success=False,
                error=str(e),
            )

    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch inference via subprocess pipe.

        Sends multiple inference requests sequentially to the child process.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results
        """
        # SubProcess doesn't have native batch support
        # Process each request individually
        results: List[InferenceResult] = []
        for req in requests:
            result = self.infer(req)
            results.append(result)
        return results

    def process(self, video_path: str, processing_config: ProcessingConfig,
                progress_callback=None, stage_callback=None, log_callback=None,
                image_sequence_frames=None):
        """Process a video file (legacy interface — not supported for subprocess).

        SubProcessBackend does not support the legacy process() interface.
        Use infer() for frame-by-frame inference via subprocess pipes.
        """
        from core.types import ProcessingResult
        return ProcessingResult(
            success=False,
            error_message="SubProcessBackend does not support legacy process() interface. Use infer() instead.",
        )

    def cancel(self) -> None:
        """Cancel the current processing operation."""
        if self._is_process_alive():
            try:
                msg = EngineMessage(type="shutdown")
                self._send_message(msg, timeout=2.0)
            except Exception:
                pass

        self._terminate_process()
        self._engine_status = EngineStatus.IDLE

    def cleanup(self) -> None:
        """Clean up resources and terminate the child process."""
        self._terminate_process()
        self._engine_status = EngineStatus.IDLE
        self._is_initialized = False
        self._pending_requests.clear()

    def _send_message(
        self, msg: EngineMessage, timeout: float = 30.0
    ) -> EngineMessage:
        """Send a JSON message to the child process and wait for response.

        Thread-safe: uses a lock to prevent concurrent writes to stdin.

        Args:
            msg: Message to send
            timeout: Maximum time to wait for response (seconds)

        Returns:
            Response message from child process

        Raises:
            RuntimeError: If child process is not running or communication fails
        """
        if not self._is_process_alive():
            raise RuntimeError("Child process not running")

        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        with self._send_lock:
            # Send message
            json_str = msg.to_json() + "\n"
            self._process.stdin.write(json_str.encode("utf-8"))
            self._process.stdin.flush()

            # Read response
            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("Child process closed stdout unexpectedly")

            response_str = response_line.decode("utf-8").strip()
            if not response_str:
                raise RuntimeError("Empty response from child process")

            return EngineMessage.from_json(response_str)

    def _is_process_alive(self) -> bool:
        """Check if the child process is still running.

        Returns:
            True if the process is alive
        """
        if self._process is None:
            return False
        return self._process.poll() is None

    def _terminate_process(self) -> None:
        """Terminate the child process gracefully."""
        if self._process is None:
            return

        try:
            # Try graceful shutdown first
            if self._process.stdin:
                shutdown_msg = EngineMessage(type="shutdown")
                json_str = shutdown_msg.to_json() + "\n"
                self._process.stdin.write(json_str.encode("utf-8"))
                self._process.stdin.flush()

            # Wait briefly for graceful exit
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                # Force terminate
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()

            logger.info(f"Child process terminated: pid={self._process.pid}")

        except Exception as e:
            logger.warning(f"Error terminating child process: {e}")
            try:
                self._process.kill()
            except Exception:
                pass
        finally:
            self._process = None
