"""TensorRT-RTX backend for video frame interpolation.

This backend runs exported RIFE ONNX models through the native ``tensorrt_rtx``
API (NOT the ONNX Runtime TensorRT execution provider). It is the second
accelerator engine alongside the ONNX Runtime backend.

Pipeline:
    1. ``load_model()`` resolves and parses the ONNX graph only to discover the
       input/output tensor names and channel count. No engine is built yet
       (frame size is unknown at load time).
    2. ``infer()`` lazily builds (or loads from disk cache) a static-shape engine
       for the actual (adapter-aligned) input shapes and runs
       ``execute_async_v3``.

Engine cache:
    Built engines are serialized to ``{models_dir}/trt_rtx_cache/<key>.engine``.
    ``<key>`` is derived from the ONNX stem and the adapter-aligned input
    shapes. Re-runs with the same shape reuse the cached engine (skipping the
    multi-hundred-ms build).

Input contract:
    The adapter for ``model_config["model_type"]`` builds the feed tensors
    (see :mod:`core.backends.adapters` and
    ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md`` §5/§6). The backend binds **every**
    engine input/output tensor; it never assumes a single ``[1, C, H, W]`` I/O.

Core constraint:
    Backend 不接触文件路径（除加载模型资产本身），只接收 numpy/tensor 数据。
    Backend 不自主决定 IO 时机，由 TaskScheduler 调度。
    Backend 不直接写文件（引擎缓存除外），推理结果返回给 TaskScheduler。

NOTE: no module-level ``import torch`` / ``import tensorrt_rtx`` (see
core/backends/AGENTS.md) — all such imports are done lazily inside methods.
"""

from __future__ import annotations

import time
from pathlib import Path
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

# Workspace pool size for the builder (1 GiB).
_WORKSPACE_BYTES = 1 << 30

# Subdirectory (under models_dir) for serialized engine cache files.
_CACHE_SUBDIR = "trt_rtx_cache"


class TensorRTRTXBackend(BaseBackend):
    """TensorRT-RTX based video processing backend (native tensorrt_rtx API).

    Runs exported RIFE ONNX models via TensorRT-RTX with a lazily built,
    disk-cached, static-shape engine. Input/output tensors are torch tensors to
    match the pipeline contract.
    """

    # Backend metadata
    BACKEND_TYPE = BackendType.TENSORRT_RTX
    BACKEND_NAME = "TensorRT-RTX"
    BACKEND_DESCRIPTION = "Native TensorRT-RTX inference backend for exported RIFE models"

    # Supported features
    SUPPORTS_INTERPOLATION = True
    SUPPORTS_UPSCALING = False
    SUPPORTS_SCENE_DETECTION = False

    # Supported models — derived from the adapter registry (spec §5/§6).
    SUPPORTED_MODELS = supported_models()

    def __init__(
        self,
        config: BackendConfig,
        parent=None,
    ):
        super().__init__(config, parent)
        self._engine = None
        self._engine_key: Optional[str] = None
        # Execution context + output buffers are expensive to create, so they
        # are built once per engine/shape and reused (see _ensure_context).
        self._context = None
        self._ctx_engine = None
        self._ctx_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
        self._out_tensors: List[Any] = []
        self._trt_logger = None
        self._onnx_path: Optional[Path] = None
        self._adapter: Optional[ModelAdapter] = None
        self._model_type = ""
        self._in_names: List[str] = []
        self._out_names: List[str] = []
        # Back-compat convenience aliases (first input/output).
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._in_channels: Optional[int] = None
        # Fully-static spatial (H, W) of a 4D input, or None if dynamic (M2).
        self._fixed_hw: Optional[Tuple[int, int]] = None
        self._cancelled = False
        self._loaded = False

    def initialize(self) -> bool:
        """Initialize the TensorRT-RTX backend.

        Only checks that ``tensorrt_rtx`` is importable and logs its version.
        The engine is created later in ``infer()`` (via ``_ensure_engine``).
        """
        try:
            import tensorrt_rtx as trt

            version = getattr(trt, "__version__", "unknown")
            logger.info(f"TensorRT-RTX version: {version}")
            self._is_initialized = True
            return True
        except ImportError as e:
            logger.error(
                f"TensorRT-RTX not available (pip install tensorrt-rtx): {e}"
            )
            return False

    def load_model(self, model_config: Dict[str, Any]) -> bool:
        """Resolve and parse the ONNX model (no engine build).

        Parsing only discovers the input/output tensor names and channel count.
        The actual engine is built lazily in :meth:`_ensure_engine` once the
        aligned frame size is known.

        Args:
            model_config: {
                "model_type": "rife",
                "model_version": "4.26",             # version token
                "onnx_path": "/path/to/model.onnx",  # optional override
                "checkpoint_path": "/path/to/model.onnx",  # optional override
            }

        Returns:
            True if the graph was parsed successfully.
        """
        model_type = str(model_config.get("model_type", "") or "").lower()
        if not model_type:
            logger.error("TensorRT-RTX load_model: model_config missing 'model_type'")
            return False

        # Normalize a checkpoint name to its version token (spec §7) before the
        # support check, so both token and checkpoint inputs are accepted.
        raw_version = str(model_config.get("model_version", "") or "")
        version = checkpoint_to_version(model_type, raw_version) or raw_version

        # Validate against the adapter registry first (cain/sepconv are rejected).
        if not self.is_model_supported(model_type, version):
            logger.error(
                f"TensorRT-RTX does not support model '{model_type}' version '{version}'"
            )
            return False

        # m3: TensorRT-RTX executes only on CUDA; refuse non-CUDA devices early
        # (execute_async_v3 cannot run against a CPU/other-device context).
        device = str(self._config.get_device() or "")
        if not self._is_cuda_device(device):
            logger.error(
                f"TensorRT-RTX requires a CUDA device, got '{device}'. "
                f"Use the ONNX backend for non-CUDA execution."
            )
            return False

        try:
            import tensorrt_rtx as trt

            explicit = model_config.get("onnx_path") or model_config.get("checkpoint_path")
            path = resolve_onnx_path(
                model_type, version, str(self._config.models_dir), explicit=explicit
            )
            if path is None or not path.exists():
                logger.error(f"ONNX model file not found: {path}")
                return False

            self._trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(self._trt_logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, self._trt_logger)

            with open(path, "rb") as f:
                ok = parser.parse(f.read())
            if not ok:
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                logger.error(f"Failed to parse ONNX model: {path}")
                return False

            # Discover ALL input/output names dynamically from the parsed graph.
            self._in_names = [
                network.get_input(i).name for i in range(network.num_inputs)
            ]
            self._out_names = [
                network.get_output(i).name for i in range(network.num_outputs)
            ]
            if not self._in_names or not self._out_names:
                logger.error(f"ONNX graph has no inputs/outputs: {path}")
                return False

            # Channel count: first 4D input with a static channel dimension.
            self._in_channels = None
            for i in range(network.num_inputs):
                shape = tuple(network.get_input(i).shape)
                if len(shape) == 4 and int(shape[1]) > 0:
                    self._in_channels = int(shape[1])
                    break

            # Back-compat aliases.
            self._input_name = self._in_names[0]
            self._output_name = self._out_names[0]

            # Record a fully-static spatial size, if the graph has one (M2).
            self._fixed_hw = self._detect_fixed_hw(network)

            self._adapter = get_adapter(model_type)
            self._model_type = model_type
            self._onnx_path = path

            # Ensure the engine-cache directory exists and log it.
            cache_dir = self._cache_dir()
            logger.info(f"TensorRT-RTX engine cache dir: {cache_dir}")

            self._loaded = True
            self._cancelled = False
            logger.info(
                f"TensorRT-RTX model loaded: {path} | adapter={model_type} | "
                f"channels={self._in_channels} | "
                f"inputs={self._in_names} outputs={self._out_names} | "
                f"fixed_hw={self._fixed_hw}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load TensorRT-RTX model: {e}")
            return False

    @staticmethod
    def _is_cuda_device(device: str) -> bool:
        """True if ``device`` refers to a CUDA torch device (m3)."""
        return str(device or "").strip().lower().startswith("cuda")

    @staticmethod
    def _detect_fixed_hw(network) -> Optional[Tuple[int, int]]:
        """Return a fully-static ``(H, W)`` from the network's 4D inputs (M2).

        TensorRT reports dynamic dims as ``-1``; only inputs with both spatial
        dims positive qualify. ``None`` means every 4D input is dynamic.
        """
        fixed: Optional[Tuple[int, int]] = None
        for i in range(network.num_inputs):
            shape = tuple(int(d) for d in network.get_input(i).shape)
            if len(shape) != 4:
                continue
            hh, ww = shape[2], shape[3]
            if hh > 0 and ww > 0:
                hw = (hh, ww)
                if fixed is None:
                    fixed = hw
                elif fixed != hw:
                    logger.warning(
                        f"Network declares differing static spatial sizes "
                        f"{fixed} vs {hw}; using {fixed}"
                    )
        return fixed

    def _cache_dir(self) -> Path:
        """Return (and create) the engine cache directory."""
        cache_dir = Path(self._config.models_dir) / _CACHE_SUBDIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @staticmethod
    def _engine_key(
        stem: str, input_shapes: Dict[str, Tuple[int, ...]]
    ) -> str:
        """Build a filesystem-safe engine cache key from aligned input shapes.

        Single NCHW input -> ``<stem>_<C>_<H>x<W>`` (spec §6). Additional
        tensors (e.g. FILM's ``timestep``) contribute their flattened dims so
        different contracts never collide.
        """
        names = sorted(input_shapes)
        primary = names[0]
        for n in names:
            if len(input_shapes[n]) == 4:
                primary = n
                break
        pshape = tuple(int(d) for d in input_shapes[primary])
        if len(pshape) == 4:
            _, c, h, w = pshape
            key = f"{stem}_{c}_{h}x{w}"
        else:
            key = f"{stem}_" + "x".join(str(d) for d in pshape)
        if len(names) > 1:
            extra = "_".join(
                "x".join(str(int(d)) for d in input_shapes[n])
                for n in names
                if n != primary
            )
            key = f"{key}_{extra}"
        return key

    def _ensure_engine(self, input_shapes: Dict[str, Tuple[int, ...]]):
        """Return a cached/serialized engine for the given aligned input shapes.

        The engine key derives from the ONNX stem and all input shapes. In-memory
        reuse is checked first, then the on-disk cache, and finally the engine is
        built (static min=opt=max per input) and serialized to disk.

        Args:
            input_shapes: ``{input_name: tuple(shape)}`` for every engine input.

        Returns:
            A deserialized ``ICudaEngine``.
        """
        import tensorrt_rtx as trt

        stem = self._onnx_path.stem if self._onnx_path else "model"
        key = self._engine_key(stem, input_shapes)

        # 1. In-memory reuse.
        if self._engine is not None and self._engine_key == key:
            return self._engine

        if self._trt_logger is None:
            self._trt_logger = trt.Logger(trt.Logger.WARNING)

        cache_file = self._cache_dir() / f"{key}.engine"
        runtime = trt.Runtime(self._trt_logger)

        # 2. On-disk cache hit.
        if cache_file.exists():
            try:
                blob = cache_file.read_bytes()
                engine = runtime.deserialize_cuda_engine(blob)
                if engine is not None:
                    logger.info(f"TensorRT-RTX engine cache HIT: {cache_file.name}")
                    self._engine = engine
                    self._engine_key = key
                    return engine
                logger.warning(
                    f"Engine cache deserialize returned None, rebuilding: {cache_file}"
                )
            except Exception as e:
                logger.warning(f"Engine cache load failed, rebuilding: {e}")

        # 3. Build + serialize.
        logger.info(
            f"Building TensorRT-RTX engine for shapes {input_shapes} (key={key})"
        )
        builder = trt.Builder(self._trt_logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, self._trt_logger)
        with open(self._onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parse failed while building engine")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE_BYTES)
        profile = builder.create_optimization_profile()
        # Static min = opt = max = actual shape, for every input.
        for name, shape in input_shapes.items():
            static_shape = tuple(int(d) for d in shape)
            try:
                profile.set_shape(name, static_shape, static_shape, static_shape)
            except Exception as e:  # static tensors may reject redundant profiles
                logger.debug(f"Optimization profile skip for '{name}': {e}")
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        blob = bytes(serialized)

        # Persist for subsequent runs.
        try:
            cache_file.write_bytes(blob)
            logger.info(f"TensorRT-RTX engine cache MISS -> built & saved: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to write engine cache {cache_file}: {e}")

        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")

        self._engine = engine
        self._engine_key = key
        return engine

    def _read_engine_io(self, engine) -> None:
        """Populate ALL input/output names from an engine using the tensor API.

        The engine may expose names in a different order than the parsed network,
        so we compare the sets before overwriting and warn (never block) on a
        mismatch (m2).
        """
        import tensorrt_rtx as trt

        in_names: List[str] = []
        out_names: List[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            dtype = engine.get_tensor_dtype(name)
            if dtype != trt.float32:
                logger.warning(f"Tensor '{name}' has non-float32 dtype: {dtype}")
            if mode == trt.TensorIOMode.OUTPUT:
                out_names.append(name)
            else:
                in_names.append(name)

        # m2: engine I/O names should match what load_model parsed from the graph.
        if self._in_names and set(self._in_names) != set(in_names):
            logger.warning(
                f"Engine input names {in_names} differ from parsed network "
                f"inputs {self._in_names}; using engine names"
            )
        if self._out_names and set(self._out_names) != set(out_names):
            logger.warning(
                f"Engine output names {out_names} differ from parsed network "
                f"outputs {self._out_names}; using engine names"
            )

        self._in_names = in_names
        self._out_names = out_names
        self._input_name = in_names[0] if in_names else None
        self._output_name = out_names[0] if out_names else None

    def _ensure_context(
        self, input_shapes: Dict[str, Tuple[int, ...]], device: str
    ):
        """Return a cached execution context (+ output buffers) for ``input_shapes``.

        ``create_execution_context()`` in TensorRT-RTX is very expensive (several
        seconds the first time), so the context and its output GPU buffers are
        built once per engine/shape and reused by every subsequent inference.

        Args:
            input_shapes: ``{input_name: tuple(shape)}`` for every engine input.
            device: torch device string (e.g. ``"cuda:0"``); used for output
                buffers so XPU/other devices are not hardcoded to CUDA.

        Returns:
            A ready-to-use ``IExecutionContext`` with all output buffers bound.
        """
        import torch

        engine = self._ensure_engine(input_shapes)
        if (
            self._context is None
            or self._ctx_engine is not engine
            or self._ctx_shapes != input_shapes
        ):
            self._read_engine_io(engine)
            ctx = engine.create_execution_context()
            torch_device = torch.device(device)
            # Set input shapes only for dynamic dims (static inputs reject it).
            for name, shape in input_shapes.items():
                try:
                    cur = tuple(ctx.get_tensor_shape(name))
                    if any(int(d) < 0 for d in cur):
                        ctx.set_input_shape(name, tuple(int(d) for d in shape))
                except Exception as e:
                    logger.debug(f"set_input_shape skip for '{name}': {e}")
            # Allocate + bind every output buffer.
            out_tensors: List[Any] = []
            for name in self._out_names:
                out_shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
                out_t = torch.empty(out_shape, dtype=torch.float32, device=torch_device)
                ctx.set_tensor_address(name, out_t.data_ptr())
                out_tensors.append(out_t)

            self._context = ctx
            self._ctx_engine = engine
            self._ctx_shapes = dict(input_shapes)
            self._out_tensors = out_tensors
        return self._context

    @staticmethod
    def _to_numpy_frame(frame) -> "Any":
        """Convert a torch/numpy frame to a contiguous float32 numpy array."""
        import numpy as np

        if hasattr(frame, "detach"):
            arr = frame.detach().cpu().numpy()
        else:
            arr = np.asarray(frame)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _build_extra(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge adapter-relevant knobs from config/model_config into ctx.extra."""
        extra: Dict[str, Any] = {}
        for key in ("scale", "fastmode", "ensemble", "in_channels"):
            if key in self._config.extra:
                extra[key] = self._config.extra[key]
            if key in model_config:
                extra[key] = model_config[key]
        return extra

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Single frame-pair inference via TensorRT-RTX.

        Runs the adapter pipeline (expand -> align -> build_feeds -> postprocess)
        and binds **every** engine input/output tensor (spec §6).

        Args:
            request: InferenceRequest with frame0, frame1, timestep, model_config

        Returns:
            InferenceResult with interpolated frame [3, H, W] float32 on CPU.
        """
        if not self._loaded or self._onnx_path is None or self._adapter is None:
            import torch

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error="Model not loaded. Call load_model() first.",
            )

        # m3: guard against non-CUDA execution (execute_async_v3 is CUDA-only).
        device_str = str(self._config.get_device() or "")
        if not self._is_cuda_device(device_str):
            import torch

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=(
                    f"TensorRT-RTX requires a CUDA device, got '{device_str}'. "
                    f"Use the ONNX backend for non-CUDA execution."
                ),
            )

        start_time = time.perf_counter()
        try:
            import numpy as np
            import torch

            f0 = self._to_numpy_frame(request.frame0)
            f1 = self._to_numpy_frame(request.frame1)
            _, h, w = f0.shape

            # M2: static-size graphs must not receive other resolutions.
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

            adapter = self._adapter
            frames = adapter.expand_frames([f0, f1])
            pad_hw = adapter.align_shape(h, w)
            ctx = AdapterContext(
                input_names=list(self._in_names),
                output_names=list(self._out_names),
                src_hw=(h, w),
                pad_hw=pad_hw,
                in_channels=self._in_channels,
                device=self._config.get_device(),
                extra=self._build_extra(request.model_config),
            )
            feeds = adapter.build_feeds(frames, request.timestep, ctx)
            if not feeds:
                raise AdapterError("adapter.build_feeds returned no feeds")

            # m1: ATM/MoMo only support t=0.5; do not silently return a 0.5 frame.
            if ctx.extra.get("only_t05") and abs(float(request.timestep) - 0.5) > 1e-6:
                return InferenceResult(
                    output_frame=torch.empty(0),
                    success=False,
                    error=f"模型 '{self._model_type}' 仅支持 timestep=0.5",
                )

            input_shapes = {
                name: tuple(int(d) for d in np.asarray(arr).shape)
                for name, arr in feeds.items()
            }

            # Reuse the cached context/output buffers for these shapes; only the
            # input addresses move between calls.
            exec_ctx = self._ensure_context(input_shapes, ctx.device)
            device = torch.device(ctx.device)
            in_tensors = []
            for name, arr in feeds.items():
                t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)
                exec_ctx.set_tensor_address(name, t.data_ptr())
                in_tensors.append(t)  # keep alive until after synchronize

            if device.type == "cuda":
                exec_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
            else:
                exec_ctx.execute_async_v3(0)

            # Collect ALL output buffers -> adapter.postprocess.
            raw_outputs = [t.float().cpu().numpy() for t in self._out_tensors]
            post = adapter.postprocess(raw_outputs, ctx)
            frame = torch.from_numpy(np.ascontiguousarray(post, dtype=np.float32)).float()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return InferenceResult(
                output_frame=frame,
                success=True,
                inference_time_ms=elapsed_ms,
            )
        except AdapterError as e:
            logger.error(f"TensorRT-RTX adapter error: {e}")
            import torch

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"TensorRT-RTX inference error: {e}")
            import torch

            return InferenceResult(
                output_frame=torch.empty(0),
                success=False,
                error=str(e),
            )

    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch inference.

        Runs one inference per request (RIFE engines here are static single-frame
        graphs). Batch kernels are not assumed.

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
        """Release the engine/context. Safe to call multiple times."""
        self._engine = None
        self._engine_key = None
        self._context = None
        self._ctx_engine = None
        self._ctx_shapes = None
        self._out_tensors = []
        self._adapter = None
        self._model_type = ""
        self._in_names = []
        self._out_names = []
        self._input_name = None
        self._output_name = None
        self._in_channels = None
        self._fixed_hw = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up resources."""
        self.unload_model()
