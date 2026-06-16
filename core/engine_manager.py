"""Engine Manager: lifecycle + GPU allocation + auto-configuration.

This module provides the EngineManager and EngineInstance classes for managing
inference engine instances. Each EngineInstance represents a configured backend
bound to a specific GPU device with a specific model.

Architecture:
    EngineManager → creates/manages EngineInstance objects
    EngineInstance → holds a BaseBackend subclass + GPU device binding
    InProcessBackend → direct Python function call (PyTorch, TensorRT)
    SubProcessBackend → stdin/stdout JSON communication (ncnn-vulkan, VapourSynth)

Design constraints:
    - EngineManager does NOT execute inference (only lifecycle management)
    - EngineManager delegates GPU detection to DeviceManager (authoritative source)
    - SubProcessBackend uses stdin/stdout pipes only (no sockets)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.types import BackendType, EngineStatus


@dataclass
class EngineInstance:
    """A configured inference engine bound to a GPU device.

    Attributes:
        engine_id: Unique identifier (e.g., "rife_cuda0")
        backend_type: Type of inference backend
        execution_mode: "inprocess" or "subprocess"
        gpu_device: Device string (e.g., "cuda:0", "xpu:0", "cpu")
        model_config: Model-specific configuration dict
        status: Current engine lifecycle state
    """

    engine_id: str
    backend_type: BackendType
    execution_mode: str  # "inprocess" | "subprocess"
    gpu_device: str  # "cuda:0" | "xpu:0" | "cpu"
    model_config: Dict[str, Any] = field(default_factory=dict)
    status: EngineStatus = EngineStatus.IDLE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        data = asdict(self)
        data["backend_type"] = self.backend_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EngineInstance:
        """Deserialize from dictionary (e.g., loaded from JSON)."""
        data = data.copy()
        data["backend_type"] = BackendType(data["backend_type"])
        data["status"] = EngineStatus(data.get("status", "idle"))
        return cls(**data)


# Mapping: BackendType → default execution mode
_BACKEND_EXECUTION_MODE: Dict[BackendType, str] = {
    BackendType.TORCH: "inprocess",
    BackendType.TENSORRT: "inprocess",
    BackendType.DIRECTML: "inprocess",
    BackendType.NCNN: "subprocess",
    BackendType.VAPOURSYNTH: "subprocess",
    BackendType.ONNX: "inprocess",
}


class EngineManager:
    """Engine manager: create/query/list engine instances.

    Responsibilities:
        - Create engine instances with GPU device binding
        - Auto-configure engines based on detected hardware
        - Persist engine configuration to engines.json
        - Query engine status

    NOT responsible for:
        - Executing inference (that's Backend's job)
        - Direct GPU context management (that's DeviceManager's job)
    """

    def __init__(self, config_dir: str = "config"):
        """Initialize the engine manager.

        Args:
            config_dir: Directory containing engine configuration files
        """
        self._engines: Dict[str, EngineInstance] = {}
        self._config_path = Path(config_dir) / "engines.json"
        self._load_engine_configs()

    def create_engine(
        self,
        engine_id: str,
        backend_type: BackendType,
        gpu_device: str,
        model_config: Optional[Dict[str, Any]] = None,
        execution_mode: Optional[str] = None,
    ) -> EngineInstance:
        """Create a new engine instance.

        Args:
            engine_id: Unique identifier for this engine
            backend_type: Type of inference backend
            gpu_device: Device string (e.g., "cuda:0", "xpu:0", "cpu")
            model_config: Model-specific configuration dict
            execution_mode: "inprocess" or "subprocess" (auto-detected if None)

        Returns:
            The created EngineInstance

        Raises:
            ValueError: If engine_id already exists
        """
        if engine_id in self._engines:
            raise ValueError(f"Engine '{engine_id}' already exists")

        if execution_mode is None:
            execution_mode = _BACKEND_EXECUTION_MODE.get(
                backend_type, "inprocess"
            )

        instance = EngineInstance(
            engine_id=engine_id,
            backend_type=backend_type,
            execution_mode=execution_mode,
            gpu_device=gpu_device,
            model_config=model_config or {},
            status=EngineStatus.IDLE,
        )

        self._engines[engine_id] = instance
        self._save_engine_configs()

        logger.info(
            f"Created engine '{engine_id}': "
            f"type={backend_type.value}, mode={execution_mode}, "
            f"device={gpu_device}"
        )
        return instance

    def remove_engine(self, engine_id: str) -> bool:
        """Remove an engine instance.

        Args:
            engine_id: ID of the engine to remove

        Returns:
            True if engine was removed, False if not found
        """
        if engine_id not in self._engines:
            logger.warning(f"Engine '{engine_id}' not found")
            return False

        del self._engines[engine_id]
        self._save_engine_configs()

        logger.info(f"Removed engine '{engine_id}'")
        return True

    def auto_configure(self) -> List[EngineInstance]:
        """Auto-detect available GPUs and create default engines.

        Detection priority: CUDA > ROCm > XPU > CPU

        Creates one engine per detected GPU device using the default
        configuration from engines.json.

        Returns:
            List of auto-configured EngineInstance objects
        """
        from core.device_manager import device_manager
        from core.device_type import DeviceType

        # Clear existing auto-configured engines
        auto_ids = [
            eid
            for eid, eng in self._engines.items()
            if eid.startswith("auto_")
        ]
        for eid in auto_ids:
            del self._engines[eid]

        # Load default configs from engines.json
        defaults = self._load_auto_configure_defaults()

        created: List[EngineInstance] = []

        # Detect available GPUs
        gpu_devices = device_manager.get_gpu_devices()

        if not gpu_devices:
            # CPU fallback
            default = defaults.get("torch", {})
            instance = self.create_engine(
                engine_id="auto_torch_cpu",
                backend_type=BackendType(default.get("backend_type", "torch")),
                gpu_device="cpu",
                model_config={
                    "model_type": default.get("model_type", "rife"),
                    "model_version": default.get("model_version", "4.22"),
                    "precision": default.get("precision", "fp16"),
                    "scale": default.get("scale", 1.0),
                },
                execution_mode="inprocess",
            )
            created.append(instance)
            logger.info("Auto-configured: CPU fallback engine")
            return created

        # Create one engine per GPU device
        for device_info in gpu_devices:
            # Map DeviceType to device string
            if device_info.device_type == DeviceType.CUDA:
                device_str = f"cuda:{device_info.device_id}"
                backend_key = "torch"
            elif device_info.device_type == DeviceType.ROCM:
                device_str = f"cuda:{device_info.device_id}"  # ROCm uses CUDA namespace
                backend_key = "torch"
            elif device_info.device_type == DeviceType.XPU:
                device_str = f"xpu:{device_info.device_id}"
                backend_key = "torch"
            else:
                continue

            default = defaults.get(backend_key, {})
            backend_type = BackendType(default.get("backend_type", "torch"))

            # Generate unique engine ID
            engine_id = f"auto_{backend_type.value}_{device_str.replace(':', '')}"

            instance = self.create_engine(
                engine_id=engine_id,
                backend_type=backend_type,
                gpu_device=device_str,
                model_config={
                    "model_type": default.get("model_type", "rife"),
                    "model_version": default.get("model_version", "4.22"),
                    "precision": default.get("precision", "fp16"),
                    "scale": default.get("scale", 1.0),
                },
                execution_mode=_BACKEND_EXECUTION_MODE.get(backend_type, "inprocess"),
            )
            created.append(instance)

            logger.info(
                f"Auto-configured: {engine_id} on {device_str} "
                f"({device_info.display_name})"
            )

        self._save_engine_configs()
        return created

    def get_engine(self, engine_id: str) -> Optional[EngineInstance]:
        """Get an engine instance by ID.

        Args:
            engine_id: Engine identifier

        Returns:
            EngineInstance if found, None otherwise
        """
        return self._engines.get(engine_id)

    def list_engines(self) -> List[EngineInstance]:
        """List all engine instances.

        Returns:
            List of all EngineInstance objects
        """
        return list(self._engines.values())

    def get_engines_by_device(self, gpu_device: str) -> List[EngineInstance]:
        """Get engines bound to a specific GPU device.

        Args:
            gpu_device: Device string to filter by

        Returns:
            List of EngineInstance objects on the specified device
        """
        return [
            eng for eng in self._engines.values()
            if eng.gpu_device == gpu_device
        ]

    def get_engines_by_type(self, backend_type: BackendType) -> List[EngineInstance]:
        """Get engines of a specific backend type.

        Args:
            backend_type: Backend type to filter by

        Returns:
            List of EngineInstance objects of the specified type
        """
        return [
            eng for eng in self._engines.values()
            if eng.backend_type == backend_type
        ]

    def update_status(self, engine_id: str, status: EngineStatus) -> bool:
        """Update an engine's status.

        Args:
            engine_id: Engine identifier
            status: New status

        Returns:
            True if updated, False if engine not found
        """
        engine = self._engines.get(engine_id)
        if engine is None:
            return False
        engine.status = status
        return True

    def _load_engine_configs(self) -> None:
        """Load engine configurations from engines.json."""
        if not self._config_path.exists():
            logger.debug(f"Engine config not found: {self._config_path}")
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            engines_data = data.get("engines", [])
            for eng_data in engines_data:
                try:
                    instance = EngineInstance.from_dict(eng_data)
                    self._engines[instance.engine_id] = instance
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to load engine config: {e}")

            logger.debug(f"Loaded {len(self._engines)} engine configs")

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load engine configs: {e}")

    def _load_auto_configure_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Load auto-configure defaults from engines.json.

        Returns:
            Dict mapping backend key to default config
        """
        if not self._config_path.exists():
            return {}

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("auto_configure_defaults", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_engine_configs(self) -> None:
        """Persist engine configurations to engines.json."""
        try:
            # Load existing file to preserve metadata and defaults
            existing_data: Dict[str, Any] = {}
            if self._config_path.exists():
                with open(self._config_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

            # Update engines list
            existing_data["engines"] = [
                eng.to_dict() for eng in self._engines.values()
            ]

            # Ensure metadata exists
            if "_metadata" not in existing_data:
                existing_data["_metadata"] = {
                    "version": "1.0",
                    "description": "Engine configuration for VFI-gui",
                    "source_app": "VFI-gui",
                }

            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self._engines)} engine configs")

        except OSError as e:
            logger.error(f"Failed to save engine configs: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all engines for display/logging.

        Returns:
            Dictionary with engine summary
        """
        return {
            "total_engines": len(self._engines),
            "engines": [
                {
                    "engine_id": eng.engine_id,
                    "backend_type": eng.backend_type.value,
                    "execution_mode": eng.execution_mode,
                    "gpu_device": eng.gpu_device,
                    "status": eng.status.value,
                    "model_type": eng.model_config.get("model_type", ""),
                    "model_version": eng.model_config.get("model_version", ""),
                }
                for eng in self._engines.values()
            ],
        }


# Singleton instance
engine_manager = EngineManager()


def main():
    """CLI entry point for EngineManager verification.

    Usage:
        python -m core.engine_manager --list --json
        python -m core.engine_manager --auto-configure --json
    """
    import argparse

    parser = argparse.ArgumentParser(description="VFI-gui Engine Manager CLI")
    parser.add_argument("--list", action="store_true", help="List all configured engines")
    parser.add_argument("--auto-configure", action="store_true", help="Auto-detect GPUs and create default engines")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    if args.auto_configure:
        engines = engine_manager.auto_configure()
        if args.json:
            print(json.dumps([eng.to_dict() for eng in engines], indent=2, ensure_ascii=False))
        else:
            for eng in engines:
                print(f"  {eng.engine_id}: type={eng.backend_type.value}, device={eng.gpu_device}, mode={eng.execution_mode}")
        return

    if args.list:
        engines = engine_manager.list_engines()
        if args.json:
            print(json.dumps([eng.to_dict() for eng in engines], indent=2, ensure_ascii=False))
        else:
            if not engines:
                print("No engines configured. Use --auto-configure to detect GPUs.")
                return
            for eng in engines:
                print(f"  {eng.engine_id}: type={eng.backend_type.value}, device={eng.gpu_device}, mode={eng.execution_mode}, status={eng.status.value}")
        return

    # Default: print summary
    summary = engine_manager.get_summary()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Total engines: {summary['total_engines']}")
        for eng in summary["engines"]:
            print(f"  {eng['engine_id']}: type={eng['backend_type']}, device={eng['gpu_device']}, status={eng['status']}")


if __name__ == "__main__":
    main()
