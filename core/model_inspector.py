"""Deep model inspection module for extracting metadata and structural information.

This module provides comprehensive inspection capabilities for various model file
formats without requiring GPU execution. All heavy dependencies are lazily imported
to ensure import safety.

Supported formats:
- PyTorch checkpoints (.pth, .pt, .ckpt, .bin)
- ONNX models (.onnx)
- TensorRT engines (.engine)
- Safetensors (.safetensors)
- GGUF models (.gguf)
- GGML models (.ggml)
"""

from __future__ import annotations

import struct
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, BinaryIO


class ModelFormat(Enum):
    """Supported model file formats."""
    PYTORCH_PTH = "pytorch_pth"
    PYTORCH_PT = "pytorch_pt"
    PYTORCH_CKPT = "pytorch_ckpt"
    PYTORCH_BIN = "pytorch_bin"
    ONNX = "onnx"
    TENSORRT_ENGINE = "tensorrt_engine"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    GGML = "ggml"
    UNKNOWN = "unknown"


@dataclass
class ModelFileInfo:
    """Basic file information for a model file."""
    path: str
    name: str
    extension: str
    size_mb: float
    created_time: datetime | None = None
    modified_time: datetime | None = None
    format_type: ModelFormat = ModelFormat.UNKNOWN


@dataclass
class TensorInfo:
    """Information about a single tensor in a model."""
    name: str
    shape: tuple[int, ...]
    dtype: str
    num_params: int
    size_bytes: int


@dataclass
class CheckpointInfo:
    """Deep inspection results for PyTorch checkpoint files."""
    format_specific_keys: list[str] = field(default_factory=list)
    model_state_dict_keys: list[str] = field(default_factory=list)
    optimizer_state_dict_keys: list[str] = field(default_factory=list)
    other_keys: list[str] = field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0
    tensor_details: list[TensorInfo] = field(default_factory=list)
    architecture_prefixes: list[str] = field(default_factory=list)


@dataclass
class ONNXModelInfo:
    """Deep inspection results for ONNX model files."""
    opset_version: int | None = None
    producer_name: str | None = None
    producer_version: str | None = None
    ir_version: int | None = None
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    nodes_count: int = 0
    op_types: list[str] = field(default_factory=list)


@dataclass
class SafetensorsInfo:
    """Deep inspection results for safetensors files."""
    metadata: dict[str, Any] = field(default_factory=dict)
    tensor_count: int = 0
    total_params: int = 0
    tensor_details: list[TensorInfo] = field(default_factory=list)


@dataclass
class GGUFInfo:
    """Deep inspection results for GGUF model files."""
    version: int = 0
    tensor_count: int = 0
    metadata_kv_count: int = 0
    architecture: str | None = None
    tensors: list[TensorInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionResult:
    """Complete inspection result for a model file."""
    file_info: ModelFileInfo
    checkpoint_info: CheckpointInfo | None = None
    onnx_info: ONNXModelInfo | None = None
    safetensors_info: SafetensorsInfo | None = None
    gguf_info: GGUFInfo | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# GGUF metadata value types (v3 spec)
GGUF_METADATA_TYPES = {
    0: "UINT8",
    1: "UINT16",
    2: "UINT32",
    3: "UINT64",
    4: "INT8",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "FLOAT32",
    9: "FLOAT64",
    10: "BOOL",
    11: "STRING",
    12: "ARRAY",
    13: "UINT64",  # Same as 3 in practice
}

# GGUF tensor data types
GGUF_TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_2",  # Removed in v3
    5: "Q4_3",  # Removed in v3
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
}


class ModelInspector:
    """Deep model inspector for extracting metadata and structural information.

    This class provides comprehensive inspection of model files without loading
    tensors into GPU memory. All operations are CPU-safe and handle errors gracefully.
    """

    # Extension to format mapping
    EXTENSION_MAP: dict[str, ModelFormat] = {
        ".pth": ModelFormat.PYTORCH_PTH,
        ".pt": ModelFormat.PYTORCH_PT,
        ".ckpt": ModelFormat.PYTORCH_CKPT,
        ".bin": ModelFormat.PYTORCH_BIN,
        ".onnx": ModelFormat.ONNX,
        ".engine": ModelFormat.TENSORRT_ENGINE,
        ".plan": ModelFormat.TENSORRT_ENGINE,
        ".safetensors": ModelFormat.SAFETENSORS,
        ".gguf": ModelFormat.GGUF,
        ".ggml": ModelFormat.GGML,
    }

    # Magic bytes for format detection
    MAGIC_BYTES: dict[bytes, ModelFormat] = {
        b"GGUF": ModelFormat.GGUF,
        b"GGML": ModelFormat.GGML,
    }

    # TensorRT header magic (derived from TRT binary analysis)
    TRT_MAGIC_OFFSET: int = 0

    # Common checkpoint wrapper keys
    CHECKPOINT_WRAPPERS: list[str] = [
        "model", "state_dict", "module", "model_state_dict",
        "state", "net", "network", "ema", "optimizer"
    ]

    def __init__(self) -> None:
        """Initialize the model inspector."""
        pass

    def inspect(self, path: str | Path) -> InspectionResult:
        """Inspect a model file and return structured information.

        This is the main entry point for model inspection. It detects the format
        and dispatches to the appropriate inspection method.

        Args:
            path: Path to the model file (str or Path object)

        Returns:
            InspectionResult containing all extracted information and any
            warnings/errors encountered during inspection.
        """
        path = Path(path)

        # Build basic file info
        file_info = self._build_file_info(path)

        # Detect format
        format_type = self.detect_format(path)
        file_info.format_type = format_type

        # Create result with file info
        result = InspectionResult(file_info=file_info)

        # Check if file exists
        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result

        # Check if file is readable
        if not path.is_file():
            result.errors.append(f"Path is not a file: {path}")
            return result

        # Dispatch to format-specific inspector
        try:
            if format_type in (ModelFormat.PYTORCH_PTH, ModelFormat.PYTORCH_PT,
                               ModelFormat.PYTORCH_CKPT, ModelFormat.PYTORCH_BIN):
                self._inspect_pytorch_internal(path, result)
            elif format_type == ModelFormat.ONNX:
                self._inspect_onnx_internal(path, result)
            elif format_type == ModelFormat.TENSORRT_ENGINE:
                self._inspect_tensorrt_internal(path, result)
            elif format_type == ModelFormat.SAFETENSORS:
                self._inspect_safetensors_internal(path, result)
            elif format_type == ModelFormat.GGUF:
                self._inspect_gguf_internal(path, result)
            elif format_type == ModelFormat.GGML:
                self._inspect_ggml_internal(path, result)
            else:
                result.warnings.append(f"Unknown format, cannot inspect deeply")
        except Exception as e:
            result.errors.append(f"Inspection failed: {str(e)}")

        return result

    def detect_format(self, path: Path) -> ModelFormat:
        """Detect model format from file extension and magic bytes.

        Uses both extension matching and binary header inspection to determine
        the model format.

        Args:
            path: Path to the model file

        Returns:
            ModelFormat enum value indicating detected format.
        """
        # Check extension first
        ext = path.suffix.lower()
        if ext in self.EXTENSION_MAP:
            format_from_ext = self.EXTENSION_MAP[ext]
        else:
            format_from_ext = ModelFormat.UNKNOWN

        # Try magic bytes detection for binary formats
        try:
            with open(path, "rb") as f:
                header = f.read(16)

            # Check GGUF magic
            if header[:4] == b"GGUF":
                return ModelFormat.GGUF

            # Check GGML magic
            if header[:4] == b"GGML":
                return ModelFormat.GGML

            # Check safetensors (first 8 bytes = header length as uint64)
            # Safetensors files start with JSON header length
            if ext == ".safetensors":
                try:
                    header_len = struct.unpack("<Q", header[:8])[0]
                    # Validate header length is reasonable (not too large)
                    if 0 < header_len < 10 * 1024 * 1024:  # < 10MB header
                        return ModelFormat.SAFETENSORS
                except struct.error:
                    pass

        except (IOError, OSError):
            # Can't read file, use extension-based detection
            pass

        return format_from_ext

    def _build_file_info(self, path: Path) -> ModelFileInfo:
        """Build basic file information from path.

        Args:
            path: Path to the model file

        Returns:
            ModelFileInfo with basic file metadata.
        """
        try:
            stat = path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            created_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
        except (IOError, OSError):
            size_mb = 0.0
            created_time = None
            modified_time = None

        return ModelFileInfo(
            path=str(path.resolve()),
            name=path.name,
            extension=path.suffix.lower(),
            size_mb=round(size_mb, 2),
            created_time=created_time,
            modified_time=modified_time,
        )

    def inspect_pytorch(self, path: Path) -> InspectionResult:
        """Inspect a PyTorch checkpoint file.

        Public wrapper for PyTorch inspection. Loads the checkpoint on CPU
        and extracts all tensor information.

        Args:
            path: Path to PyTorch checkpoint file

        Returns:
            InspectionResult with checkpoint_info populated.
        """
        result = InspectionResult(file_info=self._build_file_info(path))
        result.file_info.format_type = self.detect_format(path)

        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result

        self._inspect_pytorch_internal(path, result)
        return result

    def _inspect_pytorch_internal(
        self,
        path: Path,
        result: InspectionResult
    ) -> None:
        """Internal PyTorch checkpoint inspection implementation.

        Uses lazy import of torch to avoid dependency issues. Parses checkpoint
        structure and extracts tensor information.

        Args:
            path: Path to checkpoint file
            result: InspectionResult to populate
        """
        # Lazy import torch
        try:
            import torch
        except ImportError:
            result.errors.append("torch not installed, cannot inspect PyTorch files")
            return

        try:
            # Load checkpoint on CPU with weights_only=False for compatibility
            # Note: weights_only=True would be safer but breaks many legacy checkpoints
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as e:
            result.errors.append(f"Failed to load checkpoint: {str(e)}")
            return

        # Initialize checkpoint info
        checkpoint_info = CheckpointInfo()

        # Handle different checkpoint structures
        state_dict = None

        if isinstance(checkpoint, dict):
            # Check for common wrapper patterns
            for wrapper_key in self.CHECKPOINT_WRAPPERS:
                if wrapper_key in checkpoint:
                    value = checkpoint[wrapper_key]
                    if isinstance(value, dict):
                        # This is likely the state dict
                        checkpoint_info.format_specific_keys.append(wrapper_key)
                        if state_dict is None:
                            state_dict = value

            # Separate keys by type
            for key, value in checkpoint.items():
                if key in ("model_state_dict", "state_dict"):
                    if isinstance(value, dict):
                        checkpoint_info.model_state_dict_keys = list(value.keys())
                elif key == "optimizer_state_dict":
                    if isinstance(value, dict):
                        checkpoint_info.optimizer_state_dict_keys = list(value.keys())
                elif key not in self.CHECKPOINT_WRAPPERS:
                    checkpoint_info.other_keys.append(key)

            # If no wrapper found, treat entire dict as state dict
            if state_dict is None and isinstance(checkpoint, dict):
                # Check if this looks like a state dict (has tensor-like values)
                has_tensors = any(
                    self._is_tensor_like(v) for v in checkpoint.values()
                )
                if has_tensors:
                    state_dict = checkpoint
        elif self._is_tensor_like(checkpoint):
            # Single tensor saved directly
            state_dict = {"__single_tensor__": checkpoint}

        # Extract tensor information
        if state_dict is not None:
            self._extract_tensor_info(state_dict, checkpoint_info, result)

        result.checkpoint_info = checkpoint_info

    def _is_tensor_like(self, value: Any) -> bool:
        """Check if a value is tensor-like (torch.Tensor or numpy array).

        Args:
            value: Value to check

        Returns:
            True if value is tensor-like.
        """
        # Check torch tensor
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return True
        except ImportError:
            pass

        # Check numpy array
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return True
        except ImportError:
            pass

        return False

    def _extract_tensor_info(
        self,
        state_dict: dict[str, Any],
        checkpoint_info: CheckpointInfo,
        result: InspectionResult
    ) -> None:
        """Extract tensor information from a state dict.

        Args:
            state_dict: Dictionary of tensors
            checkpoint_info: CheckpointInfo to populate
            result: InspectionResult for warnings
        """
        # Lazy import torch
        try:
            import torch
        except ImportError:
            result.errors.append("torch not installed")
            return

        total_params = 0
        total_bytes = 0
        tensor_details: list[TensorInfo] = []
        prefixes: dict[str, int] = {}

        for name, tensor in state_dict.items():
            try:
                if isinstance(tensor, torch.Tensor):
                    shape = tuple(tensor.shape)
                    dtype = str(tensor.dtype).replace("torch.", "")
                    num_params = tensor.numel()
                    size_bytes = tensor.element_size() * num_params

                    tensor_info = TensorInfo(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        num_params=num_params,
                        size_bytes=size_bytes,
                    )
                    tensor_details.append(tensor_info)

                    total_params += num_params
                    total_bytes += size_bytes

                    # Extract architecture prefix (first part before first dot)
                    if "." in name:
                        prefix = name.split(".")[0]
                        prefixes[prefix] = prefixes.get(prefix, 0) + 1

                elif isinstance(tensor, (int, float, str, bool)):
                    # Non-tensor values, skip
                    pass
                else:
                    # Unknown type, try to handle numpy arrays
                    try:
                        import numpy as np
                        if isinstance(tensor, np.ndarray):
                            shape = tuple(tensor.shape)
                            dtype = str(tensor.dtype)
                            num_params = tensor.size
                            size_bytes = tensor.nbytes

                            tensor_info = TensorInfo(
                                name=name,
                                shape=shape,
                                dtype=dtype,
                                num_params=num_params,
                                size_bytes=size_bytes,
                            )
                            tensor_details.append(tensor_info)

                            total_params += num_params
                            total_bytes += size_bytes
                    except ImportError:
                        pass

            except Exception as e:
                result.warnings.append(f"Failed to process tensor '{name}': {str(e)}")

        checkpoint_info.total_params = total_params
        checkpoint_info.trainable_params = total_params  # Assume all trainable
        checkpoint_info.tensor_details = tensor_details
        checkpoint_info.architecture_prefixes = list(prefixes.keys())

    def inspect_onnx(self, path: Path) -> InspectionResult:
        """Inspect an ONNX model file.

        Public wrapper for ONNX inspection. Parses the model proto for
        inputs, outputs, nodes, and opset information.

        Args:
            path: Path to ONNX model file

        Returns:
            InspectionResult with onnx_info populated.
        """
        result = InspectionResult(file_info=self._build_file_info(path))
        result.file_info.format_type = ModelFormat.ONNX

        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result

        self._inspect_onnx_internal(path, result)
        return result

    def _inspect_onnx_internal(
        self,
        path: Path,
        result: InspectionResult
    ) -> None:
        """Internal ONNX model inspection implementation.

        Uses lazy import of onnx library to avoid dependency issues.

        Args:
            path: Path to ONNX file
            result: InspectionResult to populate
        """
        # Lazy import onnx
        try:
            import onnx
        except ImportError:
            result.warnings.append("onnx library not installed, limited inspection")
            # Try basic file read for header info
            self._parse_onnx_header_basic(path, result)
            return

        try:
            # Load model
            model = onnx.load(str(path))

            # Extract basic info
            onnx_info = ONNXModelInfo()

            # Opset version
            if hasattr(model, "opset_import") and model.opset_import:
                for opset in model.opset_import:
                    if opset.domain == "" or opset.domain == "ai.onnx":
                        onnx_info.opset_version = opset.version
                        break

            # Producer info
            if hasattr(model, "producer_name"):
                onnx_info.producer_name = model.producer_name
            if hasattr(model, "producer_version"):
                onnx_info.producer_version = model.producer_version
            if hasattr(model, "ir_version"):
                onnx_info.ir_version = model.ir_version

            # Parse graph
            if hasattr(model, "graph"):
                graph = model.graph

                # Inputs
                for input_tensor in graph.input:
                    shape = []
                    dtype = "unknown"
                    if hasattr(input_tensor, "type") and hasattr(input_tensor.type,
                                                                "tensor_type"):
                        tt = input_tensor.type.tensor_type
                        if tt.shape:
                            for dim in tt.shape.dim:
                                if dim.dim_param:
                                    shape.append(dim.dim_param)
                                elif dim.dim_value:
                                    shape.append(dim.dim_value)
                                else:
                                    shape.append(-1)
                        if tt.elem_type:
                            dtype = self._onnx_dtype_to_str(tt.elem_type)

                    onnx_info.inputs.append({
                        "name": input_tensor.name,
                        "shape": shape,
                        "dtype": dtype,
                    })

                # Outputs
                for output_tensor in graph.output:
                    shape = []
                    dtype = "unknown"
                    if hasattr(output_tensor, "type") and hasattr(output_tensor.type,
                                                                  "tensor_type"):
                        tt = output_tensor.type.tensor_type
                        if tt.shape:
                            for dim in tt.shape.dim:
                                if dim.dim_param:
                                    shape.append(dim.dim_param)
                                elif dim.dim_value:
                                    shape.append(dim.dim_value)
                                else:
                                    shape.append(-1)
                        if tt.elem_type:
                            dtype = self._onnx_dtype_to_str(tt.elem_type)

                    onnx_info.outputs.append({
                        "name": output_tensor.name,
                        "shape": shape,
                        "dtype": dtype,
                    })

                # Nodes
                onnx_info.nodes_count = len(graph.node)
                op_types: list[str] = []
                for node in graph.node:
                    if node.op_type not in op_types:
                        op_types.append(node.op_type)
                onnx_info.op_types = op_types

            result.onnx_info = onnx_info

        except Exception as e:
            result.errors.append(f"Failed to parse ONNX model: {str(e)}")

    def _onnx_dtype_to_str(self, dtype_num: int) -> str:
        """Convert ONNX dtype number to string.

        Args:
            dtype_num: ONNX tensor element type number

        Returns:
            String representation of dtype.
        """
        # ONNX tensor element types
        dtype_map = {
            0: "UNDEFINED",
            1: "FLOAT",
            2: "UINT8",
            3: "INT8",
            4: "UINT16",
            5: "INT16",
            6: "INT32",
            7: "INT64",
            8: "STRING",
            9: "BOOL",
            10: "FLOAT16",
            11: "DOUBLE",
            12: "UINT32",
            13: "UINT64",
            14: "COMPLEX64",
            15: "COMPLEX128",
            16: "BFLOAT16",
            17: "FLOAT8E3M2",
            18: "FLOAT8E4M3",
            19: "FLOAT8E5M2",
            20: "FLOAT8E5M2FN",
        }
        return dtype_map.get(dtype_num, f"UNKNOWN({dtype_num})")

    def _parse_onnx_header_basic(self, path: Path, result: InspectionResult) -> None:
        """Basic ONNX header parsing without onnx library.

        Reads the first few bytes to extract minimal information.

        Args:
            path: Path to ONNX file
            result: InspectionResult to populate
        """
        onnx_info = ONNXModelInfo()

        try:
            with open(path, "rb") as f:
                # Read first 1KB for basic header info
                header = f.read(1024)

            # ONNX uses protobuf format
            # Try to find readable strings in header
            readable_parts = []
            current_str = ""
            for byte in header:
                if 32 <= byte <= 126:  # ASCII printable
                    current_str += chr(byte)
                else:
                    if len(current_str) > 3:
                        readable_parts.append(current_str)
                    current_str = ""

            # Look for common ONNX markers
            for part in readable_parts:
                if "onnx" in part.lower():
                    result.warnings.append(f"Found ONNX marker: {part[:50]}")

        except Exception as e:
            result.warnings.append(f"Basic header read failed: {str(e)}")

        result.onnx_info = onnx_info

    def inspect_tensorrt(self, path: Path) -> InspectionResult:
        """Inspect a TensorRT engine file.

        Reads the binary header to extract profile names and basic metadata.

        Args:
            path: Path to TensorRT engine file

        Returns:
            InspectionResult with basic TensorRT info.
        """
        result = InspectionResult(file_info=self._build_file_info(path))
        result.file_info.format_type = ModelFormat.TENSORRT_ENGINE

        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result

        self._inspect_tensorrt_internal(path, result)
        return result

    def _inspect_tensorrt_internal(
        self,
        path: Path,
        result: InspectionResult
    ) -> None:
        """Internal TensorRT engine header inspection.

        TensorRT engines have a proprietary binary format. We parse the header
        to extract profile names and basic metadata without loading the full engine.

        Args:
            path: Path to TensorRT engine file
            result: InspectionResult to populate
        """
        try:
            with open(path, "rb") as f:
                # Read header (first 1KB is usually enough for metadata)
                header = f.read(1024)

            # TensorRT engine header structure (proprietary, reverse-engineered)
            # The header contains:
            # - Magic bytes (not standard, but identifiable patterns)
            # - Version info
            # - Profile names for optimization profiles

            # Try to extract readable strings (profile names, etc.)
            readable_parts: list[str] = []
            current_str = ""
            for byte in header:
                if 32 <= byte <= 126:  # ASCII printable range
                    current_str += chr(byte)
                else:
                    if len(current_str) > 4:
                        readable_parts.append(current_str)
                    current_str = ""

            if readable_parts:
                result.warnings.append(
                    f"TensorRT header contains: {readable_parts[:5]}"
                )

            # Look for common TensorRT patterns
            header_str = header.decode("latin-1", errors="ignore")

            # Try to find optimization profile markers
            # TRT engines often contain "input" profile names
            if "input" in header_str.lower():
                result.warnings.append("Found 'input' profile marker in header")

        except Exception as e:
            result.errors.append(f"Failed to read TensorRT header: {str(e)}")

    def inspect_safetensors(self, path: Path) -> InspectionResult:
        """Inspect a safetensors file.

        Parses only the JSON header, never loads tensor data.

        Args:
            path: Path to safetensors file

        Returns:
            InspectionResult with safetensors_info populated.
        """
        result = InspectionResult(file_info=self._build_file_info(path))
        result.file_info.format_type = ModelFormat.SAFETENSORS

        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result

        self._inspect_safetensors_internal(path, result)
        return result

    def _inspect_safetensors_internal(
        self,
        path: Path,
        result: InspectionResult
    ) -> None:
        """Internal safetensors header parsing.

        Safetensors format:
        - First 8 bytes: little-endian uint64 = header length (N)
        - Next N bytes: JSON header with tensor metadata
        - Rest: tensor data (NOT loaded)

        Args:
            path: Path to safetensors file
            result: InspectionResult to populate
        """
        try:
            with open(path, "rb") as f:
                # Read header length (first 8 bytes)
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    result.errors.append("File too small for safetensors format")
                    return

                header_len = struct.unpack("<Q", header_len_bytes)[0]

                # Sanity check header length
                if header_len <= 0:
                    result.errors.append("Invalid safetensors header length: 0")
                    return
                if header_len > 100 * 1024 * 1024:  # 100MB
                    result.warnings.append(
                        f"Large header ({header_len} bytes), may be corrupted"
                    )

                # Read JSON header
                header_json_bytes = f.read(header_len)
                if len(header_json_bytes) < header_len:
                    result.errors.append(
                        f"Could not read full header (got {len(header_json_bytes)}, "
                        f"expected {header_len})"
                    )
                    return

                # Parse JSON
                header_json = header_json_bytes.decode("utf-8")
                header_data = json.loads(header_json)

            # Parse header data
            safetensors_info = SafetensorsInfo()

            # Extract metadata if present
            if "__metadata__" in header_data:
                safetensors_info.metadata = header_data["__metadata__"]
                del header_data["__metadata__"]

            # Parse tensor entries
            total_params = 0
            tensor_details: list[TensorInfo] = []

            for tensor_name, tensor_meta in header_data.items():
                try:
                    # Safetensors tensor metadata format:
                    # {"dtype": "F32", "shape": [1, 2, 3], "data_offsets": [0, 100]}
                    dtype = tensor_meta.get("dtype", "unknown")
                    shape = tuple(tensor_meta.get("shape", []))
                    data_offsets = tensor_meta.get("data_offsets", [0, 0])

                    # Calculate size
                    num_params = 1
                    for dim in shape:
                        num_params *= dim if dim > 0 else 1

                    size_bytes = data_offsets[1] - data_offsets[0]

                    tensor_info = TensorInfo(
                        name=tensor_name,
                        shape=shape,
                        dtype=dtype,
                        num_params=num_params,
                        size_bytes=size_bytes,
                    )
                    tensor_details.append(tensor_info)

                    total_params += num_params

                except Exception as e:
                    result.warnings.append(
                        f"Failed to parse tensor '{tensor_name}': {str(e)}"
                    )

            safetensors_info.tensor_count = len(tensor_details)
            safetensors_info.total_params = total_params
            safetensors_info.tensor_details = tensor_details

            result.safetensors_info = safetensors_info

        except json.JSONDecodeError as e:
            result.errors.append(f"Failed to parse safetensors JSON header: {str(e)}")
        except struct.error as e:
            result.errors.append(f"Failed to parse safetensors header: {str(e)}")
        except Exception as e:
            result.errors.append(f"Safetensors inspection failed: {str(e)}")

    def inspect_gguf(self, path: Path) -> InspectionResult:
        """Inspect a GGUF model file.

        Parses the GGUF header to extract version, tensor count, metadata,
        and tensor information.

        Args:
            path: Path to GGUF file

        Returns:
            InspectionResult with gguf_info populated.
        """
        result = InspectionResult(file_info=self._build_file_info(path))
        result.file_info.format_type = ModelFormat.GGUF

        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result

        self._inspect_gguf_internal(path, result)
        return result

    def _inspect_gguf_internal(
        self,
        path: Path,
        result: InspectionResult
    ) -> None:
        """Internal GGUF header parsing implementation.

        GGUF format (v3):
        - Magic: "GGUF" (4 bytes)
        - Version: uint32 (4 bytes)
        - Tensor count: uint64 (8 bytes)
        - Metadata KV count: uint64 (8 bytes)
        - Metadata KV pairs
        - Tensor info

        Args:
            path: Path to GGUF file
            result: InspectionResult to populate
        """
        try:
            with open(path, "rb") as f:
                # Read magic
                magic = f.read(4)
                if magic != b"GGUF":
                    result.errors.append(f"Invalid GGUF magic: {magic}")
                    return

                # Read version
                version_bytes = f.read(4)
                version = struct.unpack("<I", version_bytes)[0]

                # Read tensor count
                tensor_count_bytes = f.read(8)
                tensor_count = struct.unpack("<Q", tensor_count_bytes)[0]

                # Read metadata KV count
                metadata_kv_count_bytes = f.read(8)
                metadata_kv_count = struct.unpack("<Q", metadata_kv_count_bytes)[0]

            gguf_info = GGUFInfo(
                version=version,
                tensor_count=tensor_count,
                metadata_kv_count=metadata_kv_count,
            )

            # Parse metadata and tensor info
            self._parse_gguf_metadata_and_tensors(path, gguf_info, result)

            # Extract architecture from metadata
            if "general.architecture" in gguf_info.metadata:
                gguf_info.architecture = gguf_info.metadata["general.architecture"]
            elif "architecture" in gguf_info.metadata:
                gguf_info.architecture = gguf_info.metadata["architecture"]

            result.gguf_info = gguf_info

        except struct.error as e:
            result.errors.append(f"Failed to parse GGUF header: {str(e)}")
        except Exception as e:
            result.errors.append(f"GGUF inspection failed: {str(e)}")

    def _parse_gguf_metadata_and_tensors(
        self,
        path: Path,
        gguf_info: GGUFInfo,
        result: InspectionResult
    ) -> None:
        """Parse GGUF metadata key-value pairs and tensor info.

        GGUF v3 metadata format:
        - Key length: uint64
        - Key string: bytes
        - Value type: uint32
        - Value data (type-specific)

        Args:
            path: Path to GGUF file
            gguf_info: GGUFInfo to populate
            result: InspectionResult for warnings
        """
        try:
            with open(path, "rb") as f:
                # Skip header (magic + version + tensor_count + metadata_kv_count)
                f.seek(4 + 4 + 8 + 8)

                # Parse metadata KV pairs
                metadata: dict[str, Any] = {}
                for _ in range(gguf_info.metadata_kv_count):
                    try:
                        key = self._read_gguf_string(f)
                        value_type = struct.unpack("<I", f.read(4))[0]
                        value = self._read_gguf_value(f, value_type, result)
                        metadata[key] = value
                    except Exception as e:
                        result.warnings.append(
                            f"Failed to parse metadata entry: {str(e)}"
                        )
                        break

                gguf_info.metadata = metadata

                # Parse tensor info
                tensors: list[TensorInfo] = []
                for _ in range(gguf_info.tensor_count):
                    try:
                        tensor_name = self._read_gguf_string(f)
                        n_dims = struct.unpack("<I", f.read(4))[0]

                        # Read dimensions
                        dims = []
                        for _ in range(n_dims):
                            dim = struct.unpack("<Q", f.read(8))[0]
                            dims.append(dim)

                        # Read type
                        tensor_type = struct.unpack("<I", f.read(4))[0]

                        # Read offset (not used but part of format)
                        _ = struct.unpack("<Q", f.read(8))[0]

                        # Calculate params and size
                        num_params = 1
                        for dim in dims:
                            num_params *= dim if dim > 0 else 1

                        # Get dtype string
                        dtype = GGUF_TENSOR_TYPES.get(tensor_type,
                                                       f"UNKNOWN({tensor_type})")

                        # Estimate size based on quantization type
                        size_bytes = self._estimate_gguf_tensor_size(
                            num_params, tensor_type
                        )

                        tensor_info = TensorInfo(
                            name=tensor_name,
                            shape=tuple(dims),
                            dtype=dtype,
                            num_params=num_params,
                            size_bytes=size_bytes,
                        )
                        tensors.append(tensor_info)

                    except Exception as e:
                        result.warnings.append(
                            f"Failed to parse tensor info: {str(e)}"
                        )
                        break

                gguf_info.tensors = tensors

        except Exception as e:
            result.warnings.append(f"Failed to parse GGUF metadata/tensors: {str(e)}")

    def _read_gguf_string(self, f: BinaryIO) -> str:
        """Read a GGUF string from file.

        GGUF string format:
        - Length: uint64
        - String bytes

        Args:
            f: File handle (BinaryIO)

        Returns:
            Decoded string.
        """
        length = struct.unpack("<Q", f.read(8))[0]
        string_bytes = f.read(length)
        return string_bytes.decode("utf-8")

    def _read_gguf_value(
        self,
        f: BinaryIO,
        value_type: int,
        result: InspectionResult
    ) -> Any:
        """Read a GGUF metadata value based on type.

        GGUF value types:
        - UINT8, UINT16, UINT32, UINT64: unsigned integers
        - INT8, INT16, INT32, INT64: signed integers
        - FLOAT32, FLOAT64: floats
        - BOOL: boolean
        - STRING: string
        - ARRAY: array of values

        Args:
            f: File handle
            value_type: Value type number
            result: InspectionResult for warnings

        Returns:
            Parsed value.
        """
        if value_type == 0:  # UINT8
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == 1:  # UINT16
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == 2:  # UINT32
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == 3:  # UINT64
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == 4:  # INT8
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == 5:  # INT16
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == 6:  # INT32
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == 7:  # INT64
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == 8:  # FLOAT32
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == 9:  # FLOAT64
            return struct.unpack("<d", f.read(8))[0]
        elif value_type == 10:  # BOOL
            return struct.unpack("<B", f.read(1))[0] != 0
        elif value_type == 11:  # STRING
            return self._read_gguf_string(f)
        elif value_type == 12:  # ARRAY
            # Read array type and count
            array_type = struct.unpack("<I", f.read(4))[0]
            array_count = struct.unpack("<Q", f.read(8))[0]

            # Read array values
            array_values = []
            for _ in range(array_count):
                try:
                    val = self._read_gguf_value(f, array_type, result)
                    array_values.append(val)
                except Exception as e:
                    result.warnings.append(f"Array parse error: {str(e)}")
                    break

            return array_values
        else:
            result.warnings.append(f"Unknown GGUF value type: {value_type}")
            return None

    def _estimate_gguf_tensor_size(
        self,
        num_params: int,
        tensor_type: int
    ) -> int:
        """Estimate tensor size in bytes based on quantization type.

        GGUF uses various quantization formats with different byte sizes.

        Args:
            num_params: Number of parameters
            tensor_type: Tensor type number

        Returns:
            Estimated size in bytes.
        """
        # Bytes per element for different quantization types
        # Approximate values based on GGUF spec
        bytes_per_element = {
            0: 4,   # F32
            1: 2,   # F16
            2: 0.5, # Q4_0
            3: 0.5, # Q4_1
            6: 0.625, # Q5_0
            7: 0.625, # Q5_1
            8: 1,   # Q8_0
            9: 1,   # Q8_1
            10: 0.25, # Q2_K
            11: 0.375, # Q3_K
            12: 0.5, # Q4_K
            13: 0.625, # Q5_K
            14: 0.75, # Q6_K
            15: 1,   # Q8_K
            24: 1,   # I8
            25: 2,   # I16
            26: 4,   # I32
            27: 8,   # I64
            28: 8,   # F64
        }

        bpe = bytes_per_element.get(tensor_type, 4)  # Default to F32
        return int(num_params * bpe)

    def inspect_ggml(self, path: Path) -> InspectionResult:
        """Inspect a legacy GGML model file.

        Basic header parse for legacy GGML format.

        Args:
            path: Path to GGML file

        Returns:
            InspectionResult with basic GGML info.
        """
        result = InspectionResult(file_info=self._build_file_info(path))
        result.file_info.format_type = ModelFormat.GGML

        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result

        self._inspect_ggml_internal(path, result)
        return result

    def _inspect_ggml_internal(
        self,
        path: Path,
        result: InspectionResult
    ) -> None:
        """Internal GGML header parsing implementation.

        Legacy GGML format (older, simpler than GGUF):
        - Magic: "GGML" (4 bytes)
        - Version: uint32 (varies by version)
        - Basic tensor info

        Args:
            path: Path to GGML file
            result: InspectionResult to populate
        """
        try:
            with open(path, "rb") as f:
                # Read magic
                magic = f.read(4)
                if magic != b"GGML":
                    result.warnings.append(f"Unexpected GGML magic: {magic}")

                # Try to read version (format varies)
                header = f.read(32)

                # GGML format is less standardized, provide basic info
                result.warnings.append(
                    "Legacy GGML format detected, limited inspection available"
                )

        except Exception as e:
            result.errors.append(f"Failed to read GGML header: {str(e)}")


# Convenience function for quick inspection
def inspect_model(path: str | Path) -> InspectionResult:
    """Quick inspection function for model files.

    Args:
        path: Path to model file

    Returns:
        InspectionResult with all extracted information.
    """
    inspector = ModelInspector()
    return inspector.inspect(path)