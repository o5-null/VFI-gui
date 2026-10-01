"""ONNX Runtime backend for video frame interpolation.

This backend uses ONNX Runtime to run exported VFI models (ONNX format) through
a per-``model_type`` adapter dispatch layer (see
``core/backends/adapters/`` and ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md`` §5/§6).
It is the preferred replacement for the legacy torch/CUDA inference path.

Model contract:
    The adapter for ``model_config["model_type"]`` decides how many inputs the
    graph takes, how frames/timestep are packed, and how the raw outputs are
    post-processed. The backend reads **all** input/output names from the ONNX
    session and feeds :meth:`ModelAdapter.build_feeds` output verbatim, then
    collects **all** outputs into :meth:`ModelAdapter.postprocess`.

    Input names are never hardcoded; the session's declared names are always
    used (falling back to adapter defaults only when the graph declares none).

Output: ``[3, H, W]`` float32 (unpadded to the source resolution), returned as a
torch tensor ``[3, H, W]``.

Execution providers:
    By default the backend picks the first available accelerator among CUDA and
    DirectML, falling back to CPU. The ordered provider list can be forced via
    ``BackendConfig.extra["execution_providers"]`` (a list of provider-name
    strings); unavailable names are skipped with a warning and CPU is always the
    final fallback. TensorRT EP is opt-in only (pass it explicitly) and never
    auto-selected, to avoid slow engine builds.

Core constraint:
    Backend 不接触文件路径，只接收 numpy/tensor 数据。
    Backend 不自主决定 IO 时机，由 TaskScheduler 调度。
    Backend 不直接写文件，推理结果返回给 TaskScheduler。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.types import (
    BackendType,
    BackendConfig,
    InferenceRequest,
    InferenceResult,
)
from core.models.asset_resolver import checkpoint_to_version, resolve_onnx_path
from .base_backend import BaseBackend
from .adapters import (
    AdapterContext,
    AdapterError,
    ModelAdapter,
    get_adapter,
    register_builtin_adapters,
    supported_models,
)

# Ensure the adapter registry is populated before deriving SUPPORTED_MODELS.
register_builtin_adapters()

# Ordered candidate providers (accelerators before CPU). TensorRT is listed for
# completeness but is opt-in only — it is never auto-selected by default.
_PROVIDER_PRIORITY = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

# Accelerators considered for the default (no-override) selection.
_DEFAULT_ACCELERATORS = ["CUDAExecutionProvider", "DmlExecutionProvider"]


class OnnxBackend(BaseBackend):
    """ONNX Runtime-based video processing backend.

    Runs exported RIFE ONNX models through ONNX Runtime, using the CUDA
    execution provider when available and falling back to CPU otherwise.
    Input/output tensors are torch tensors to match the pipeline contract;
    conversion to/from numpy is done lazily inside the inference methods.
    """

    # Backend metadata
    BACKEND_TYPE = BackendType.ONNX
    BACKEND_NAME = "ONNX Runtime"
    BACKEND_DESCRIPTION = "ONNX Runtime inference backend for exported RIFE models"

    # Supported features
    SUPPORTS_INTERPOLATION = True
    SUPPORTS_UPSCALING = False
    SUPPORTS_SCENE_DETECTION = False

    # Supported models — derived from the adapter registry (spec §5/§6).
    # No cain/sepconv (they have no PyTorch/ONNX implementation).
    SUPPORTED_MODELS = supported_models()

    def __init__(
        self,
        config: BackendConfig,
        parent=None,
    ):
        super().__init__(config, parent)
        self._session = None
        self._adapter: Optional[ModelAdapter] = None
        self._model_type: str = ""
        # All input/output names as declared by the loaded ONNX graph.
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        # Back-compat convenience handles (first input/output name).
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._in_channels: Optional[int] = None
        # Fully-static spatial (H, W) of a 4D input, or None if the graph's
        # spatial axes are dynamic. Static-size assets (e.g. FILM/FLAVR 256x256)
        # reject any other resolution at the ORT layer, so we pre-check.
        self._fixed_hw: Optional[Tuple[int, int]] = None
        self._cancelled = False
        self._loaded = False

    def initialize(self) -> bool:
        """Initialize the ONNX Runtime backend.

        Only checks that ``onnxruntime`` is importable and logs the version /
        available providers. The session itself is created in ``load_model``.
        """
        try:
            import onnxruntime as ort

            logger.info(f"ONNX Runtime version: {ort.__version__}")
            logger.info(f"Available providers: {ort.get_available_providers()}")
            self._is_initialized = True
            return True
        except ImportError as e:
            logger.error(f"ONNX Runtime not available (pip install onnxruntime): {e}")
            return False

    def load_model(self, model_config: Dict[str, Any]) -> bool:
        """Load an ONNX model and create the inference session.

        Validates support first (``is_model_supported``), resolving the ONNX
        path through the single-source-of-truth :mod:`core.models.asset_resolver`.

        Args:
            model_config: {
                "model_type": "rife",
                "model_version": "4.22",   # version token (or checkpoint name)
                "onnx_path": "/path/to/model.onnx",  # optional override
                "checkpoint_path": "/path/to/model.onnx",  # optional override
            }

        Returns:
            True if the session was created successfully; False otherwise
            (unsupported model, unresolvable asset, or session error).
        """
        try:
            import onnxruntime as ort

            model_type = str(model_config.get("model_type") or "").strip().lower()
            if not model_type:
                logger.error("load_model: model_config missing 'model_type'")
                return False

            # Normalize a checkpoint name to its version token (spec §7) before
            # the support check, so both token and checkpoint inputs are accepted.
            raw_version = str(model_config.get("model_version") or "")
            version = checkpoint_to_version(model_type, raw_version) or raw_version

            if not self.is_model_supported(model_type, version):
                logger.error(
                    f"Model '{model_type}' version '{version}' is not supported "
                    f"by the ONNX backend"
                )
                return False

            explicit = model_config.get("onnx_path") or model_config.get("checkpoint_path")
            path = resolve_onnx_path(
                model_type, version, str(self._config.models_dir), explicit=explicit
            )
            if path is None or not path.exists():
                logger.error(
                    f"ONNX model file not found: {path} "
                    f"(model_type={model_type}, version={version})"
                )
                return False

            # Session options
            sess_options = ort.SessionOptions()

            # Build the ordered provider list (override or auto-select).
            available = ort.get_available_providers()
            logger.info(f"Available providers: {available}")
            providers = self._build_providers(available)
            logger.info(f"Selected providers: {providers}")

            self._session = ort.InferenceSession(
                str(path), sess_options, providers=providers
            )

            # Read EVERY input/output name dynamically from the session — the
            # adapter decides which feeds/outputs are actually used.
            inputs = self._session.get_inputs()
            self._input_names = [i.name for i in inputs]
            self._output_names = [o.name for o in self._session.get_outputs()]

            # Record a fully-static spatial size, if the graph has one (M2).
            self._fixed_hw = self._detect_fixed_hw(inputs)

            # Derive the channel count from the first static 4D NCHW input
            # (symbolic/dynamic dims fall back to None and let the adapter decide).
            self._in_channels = None
            for node in inputs:
                shape = node.shape
                if len(shape) == 4 and isinstance(shape[1], int) and shape[1] > 0:
                    self._in_channels = int(shape[1])
                    break

            self._input_name = self._input_names[0] if self._input_names else None
            self._output_name = self._output_names[0] if self._output_names else None

            # Adapter dispatch for this model type (spec §6).
            self._model_type = model_type
            self._adapter = get_adapter(model_type)

            self._loaded = True
            self._cancelled = False
            logger.info(
                f"ONNX model loaded: {path} | model_type={model_type} | "
                f"providers={self._session.get_providers()} | "
                f"channels={self._in_channels} | "
                f"inputs={self._input_names} | outputs={self._output_names} | "
                f"fixed_hw={self._fixed_hw}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False

    @staticmethod
    def _detect_fixed_hw(nodes: List[Any]) -> Optional[Tuple[int, int]]:
        """Return a fully-static ``(H, W)`` from the graph's 4D inputs.

        Only inputs whose spatial axes are *both* static integers qualify. If
        every 4D input is dynamic (or there is no 4D input) the result is
        ``None`` (M2 — see docs/ONNX_TRT_MULTI_MODEL_SPEC.md).
        """
        fixed: Optional[Tuple[int, int]] = None
        for node in nodes:
            shape = list(getattr(node, "shape", []) or [])
            if len(shape) != 4:
                continue
            hh, ww = shape[2], shape[3]
            if isinstance(hh, int) and isinstance(ww, int) and hh > 0 and ww > 0:
                hw = (int(hh), int(ww))
                if fixed is None:
                    fixed = hw
                elif fixed != hw:
                    logger.warning(
                        f"Graph declares differing static spatial sizes "
                        f"{fixed} vs {hw}; using {fixed}"
                    )
        return fixed

    @staticmethod
    def _parse_device_id(device: str) -> int:
        """Parse a device id from a device string like ``"cuda:0"``."""
        if device and ":" in device:
            try:
                return int(device.split(":", 1)[1])
            except ValueError:
                return 0
        return 0

    def _build_providers(self, available: List[str]) -> List[Any]:
        """Build the ordered ONNX Runtime provider list.

        If ``config.extra["execution_providers"]`` is set, its order is used
        verbatim; names not present in ``available`` are skipped with a warning.
        Otherwise the first available accelerator from the default priority is
        selected. CPU is always appended as the final fallback.

        Args:
            available: Provider names reported by ``ort.get_available_providers()``.

        Returns:
            List of provider entries (``(name, options)`` tuples or bare strings).
        """
        override = self._config.extra.get("execution_providers")

        if override:
            requested: List[str] = []
            for name in override:
                if name in available:
                    requested.append(name)
                else:
                    logger.warning(
                        f"Requested execution provider '{name}' is not available; skipping"
                    )
        else:
            requested = [name for name in _DEFAULT_ACCELERATORS if name in available]

        # CPU is always present as the final fallback.
        if "CPUExecutionProvider" not in requested:
            requested.append("CPUExecutionProvider")

        return [self._provider_entry(name) for name in requested]

    def _provider_entry(self, name: str) -> Any:
        """Return an ORT provider entry: options tuple, or bare string if none."""
        if name == "CUDAExecutionProvider":
            return (name, {"device_id": self._parse_device_id(self._config.device)})
        # Empty options -> use the bare string form (ORT convention).
        return name

    @staticmethod
    def _to_numpy_frame(frame: Any) -> "Any":
        """Convert a torch tensor / numpy ``[3, H, W]`` to contiguous float32 numpy."""
        import numpy as np

        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(frame, dtype=np.float32))

    def _build_extra(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect pass-through adapter options from backend/model config.

        ``scale`` / ``fastmode`` / ``ensemble`` are recorded by the RIFE adapter
        (they are baked into the exported graph); ``in_channels`` seeds RIFE's
        channel layout. Nothing here is backend-specific.
        """
        extra: Dict[str, Any] = {}
        for source in (self._config.extra or {}, model_config or {}):
            for key in ("scale", "fastmode", "ensemble", "in_channels"):
                if key in source and source[key] is not None:
                    extra.setdefault(key, source[key])
        if self._in_channels is not None:
            extra.setdefault("in_channels", self._in_channels)
        return extra

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Single frame-pair inference through the per-type adapter pipeline.

        Pipeline (spec §6): frames -> ``expand_frames`` -> ``align_shape`` ->
        ``build_feeds`` -> ``session.run`` (ALL outputs) -> ``postprocess``.

        Args:
            request: InferenceRequest with frame0, frame1, timestep, model_config

        Returns:
            InferenceResult with interpolated frame [3, H, W] float32 on CPU.
        """
        if self._session is None or self._adapter is None:
            import torch

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error="Model not loaded. Call load_model() first.",
            )

        start_time = time.perf_counter()
        try:
            import numpy as np
            import torch

            adapter = self._adapter
            model_config = request.model_config or {}

            # Frames as adapter-friendly numpy [3, H, W] float32.
            f0 = self._to_numpy_frame(request.frame0)
            f1 = self._to_numpy_frame(request.frame1)
            h, w = int(f0.shape[-2]), int(f0.shape[-1])

            # M2: static-size graphs must not receive other resolutions — fail
            # with a clear message instead of letting ORT raise INVALID_ARGUMENT.
            if self._fixed_hw is not None and (h, w) != self._fixed_hw:
                fh, fw = self._fixed_hw
                return InferenceResult(
                    output_frame=torch.empty(0),
                    success=False,
                    error=(
                        f"模型 '{self._model_type}' 仅支持 {fh}x{fw} 输入，"
                        f"当前 {h}x{w}；请用该分辨率或改用动态轴导出的 ONNX"
                    ),
                )

            frames = adapter.expand_frames([f0, f1])
            pad_hw = adapter.align_shape(h, w)

            ctx = AdapterContext(
                input_names=list(self._input_names),
                output_names=list(self._output_names),
                src_hw=(h, w),
                pad_hw=pad_hw,
                in_channels=self._in_channels,
                device=self._config.get_device(),
                extra=self._build_extra(model_config),
            )

            feeds = adapter.build_feeds(frames, float(request.timestep), ctx)

            # m1: some adapters (ATM/MoMo) only support t=0.5 and simply ignore
            # other timesteps; surface that rather than silently returning 0.5.
            if ctx.extra.get("only_t05") and abs(float(request.timestep) - 0.5) > 1e-6:
                return InferenceResult(
                    output_frame=torch.empty(0),
                    success=False,
                    error=f"模型 '{self._model_type}' 仅支持 timestep=0.5",
                )

            out_names = adapter.output_names(ctx)

            # Collect ALL declared outputs, then let the adapter post-process.
            raw_outputs = self._session.run(out_names, feeds)
            post = adapter.postprocess(list(raw_outputs), ctx)  # [3, H, W] float32

            frame = torch.from_numpy(
                np.ascontiguousarray(post, dtype=np.float32)
            ).float()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return InferenceResult(
                output_frame=frame,
                success=True,
                inference_time_ms=elapsed_ms,
            )
        except AdapterError as e:
            logger.error(f"ONNX adapter error: {e}")
            import torch

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            import torch

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=str(e),
            )

    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch inference.

        ONNX Runtime runs one inference per request here; this is a simple
        loop over ``infer``. Batch kernels are not assumed from the export.

        Args:
            requests: List of InferenceRequest objects.

        Returns:
            List of InferenceResult, one per request.
        """
        return [self.infer(req) for req in requests]

    def cancel(self) -> None:
        """Cancel the current processing operation."""
        self._cancelled = True

    def unload_model(self) -> None:
        """Release the ONNX session. Safe to call multiple times."""
        self._session = None
        self._adapter = None
        self._model_type = ""
        self._input_names = []
        self._output_names = []
        self._input_name = None
        self._output_name = None
        self._in_channels = None
        self._fixed_hw = None
        self._loaded = False

    def cleanup(self) -> None:
        """Clean up resources."""
        self.unload_model()
