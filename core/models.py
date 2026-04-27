"""Unified model and TensorRT engine management.

This module provides a single unified manager for all model types:
- TensorRT engines (.engine)
- ONNX models (.onnx)
- PyTorch checkpoints (.pth, .pt, .pkl, .ckpt)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal


class ModelStatus(Enum):
    """Status of a model checkpoint."""
    INSTALLED = "installed"
    DOWNLOADABLE = "downloadable"
    PARTIAL = "partial"


@dataclass
class EngineInfo:
    """Information about a TensorRT engine file."""
    path: str
    name: str
    size_mb: float
    scale: Optional[int] = None
    precision: Optional[str] = None
    input_channels: Optional[int] = None

    @property
    def display_name(self) -> str:
        """Get a display-friendly name."""
        parts = [self.name]
        if self.scale:
            parts.append(f"{self.scale}x")
        if self.precision:
            parts.append(self.precision)
        return " - ".join(parts)


@dataclass
class CheckpointInfo:
    """Information about a model checkpoint."""
    name: str
    model_type: str
    path: Optional[Path] = None
    size_mb: float = 0.0
    status: ModelStatus = ModelStatus.DOWNLOADABLE
    download_urls: List[str] = field(default_factory=list)
    
    @property
    def display_name(self) -> str:
        """Get display-friendly name."""
        return self.name
    
    @property
    def is_installed(self) -> bool:
        """Check if checkpoint is installed."""
        return self.status == ModelStatus.INSTALLED


@dataclass
class OnnxInfo:
    """Information about an ONNX model file."""
    path: str
    name: str
    size_mb: float
    model_type: str = "scene_detect"

    @property
    def display_name(self) -> str:
        """Get display-friendly name."""
        return self.name


@dataclass
class ModelTypeInfo:
    """Information about a model type."""
    name: str
    display_name: str
    description: str = ""
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    
    @property
    def installed_count(self) -> int:
        """Count of installed checkpoints."""
        return sum(1 for cp in self.checkpoints if cp.is_installed)
    
    @property
    def total_count(self) -> int:
        """Total count of checkpoints."""
        return len(self.checkpoints)


# Model definitions with download URLs
MODEL_DEFINITIONS: Dict[str, Dict] = {
    "rife": {
        "display_name": "RIFE",
        "description": "Real-Time Intermediate Flow Estimation for Video Frame Interpolation",
        "checkpoints": {
            "rife47.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife47.pth",
                "https://huggingface.co/marduk191/rife/resolve/main/rife47.pth",
            ],
            "rife49.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife49.pth",
                "https://huggingface.co/marduk191/rife/resolve/main/rife49.pth",
            ],
            "rife417.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife417.pth",
            ],
            "rife426.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife426.pth",
            ],
            "sudo_rife4_269.662_testV1_scale1.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/sudo_rife4_269.662_testV1_scale1.pth",
            ],
        },
    },
    "film": {
        "display_name": "FILM",
        "description": "Frame Interpolation for Large Motion",
        "checkpoints": {
            "film_net_fp32.pt": [
                "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/film_net_fp32.pt",
            ],
        },
    },
    "amt": {
        "display_name": "AMT",
        "description": "All-Pairs Multi-Field Transforms for Efficient Frame Interpolation",
        "checkpoints": {
            "amt-g.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/amt-g.pth",
            ],
            "amt-s.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/amt-s.pth",
            ],
            "amt-l.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/amt-l.pth",
            ],
            "gopro_amt-s.pth": [
                "https://huggingface.co/lalala125/AMT/resolve/main/gopro_amt-s.pth",
            ],
        },
    },
    "ifrnet": {
        "display_name": "IFRNet",
        "description": "Intermediate Feature Refine Network for Efficient Frame Interpolation",
        "checkpoints": {
            "IFRNet_L_Vimeo90K.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/IFRNet_L_Vimeo90K.pth",
            ],
            "IFRNet_S_Vimeo90K.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/IFRNet_S_Vimeo90K.pth",
            ],
            "IFRNet_S_GoPro.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/IFRNet_S_GoPro.pth",
            ],
            "IFRNet_L_GoPro.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/IFRNet_L_GoPro.pth",
            ],
        },
    },
    "ifunet": {
        "display_name": "IFUnet",
        "description": "RIFE with IFUNet, FusionNet and RefineNet",
        "checkpoints": {
            "IFUnet.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/IFUnet.pth",
            ],
        },
    },
    "m2m": {
        "display_name": "M2M",
        "description": "Many-to-many Splatting for Efficient Video Frame Interpolation",
        "checkpoints": {
            "M2M.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/M2M.pth",
            ],
        },
    },
    "gmfss_fortuna": {
        "display_name": "GMFSS Fortuna",
        "description": "The All-In-One GMFSS for Anime Video Frame Interpolation",
        "checkpoints": {
            "GMFSS_fortuna_fusionnet.pkl": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/GMFSS_fortuna_fusionnet.pkl",
            ],
            "GMFSS_fortuna_feat.pkl": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/GMFSS_fortuna_feat.pkl",
            ],
            "GMFSS_fortuna_metric.pkl": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/GMFSS_fortuna_metric.pkl",
            ],
            "GMFSS_fortuna_flownet.pkl": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/GMFSS_fortuna_flownet.pkl",
            ],
            "GMFSS_fortuna_union_fusionnet.pkl": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/GMFSS_fortuna_union_fusionnet.pkl",
            ],
            "GMFSS_fortuna_union_feat.pkl": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/GMFSS_fortuna_union_feat.pkl",
            ],
            "GMFSS_fortuna_union_metric.pkl": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/GMFSS_fortuna_union_metric.pkl",
            ],
        },
    },
    "flavr": {
        "display_name": "FLAVR",
        "description": "Flow-Agnostic Video Representations for Fast Frame Interpolation",
        "checkpoints": {
            "FLAVR_2x.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/FLAVR_2x.pth",
            ],
            "FLAVR_4x.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/FLAVR_4x.pth",
            ],
            "FLAVR_8x.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/FLAVR_8x.pth",
            ],
        },
    },
    "stmfnet": {
        "display_name": "STMFNet",
        "description": "A Spatio-Temporal Multi-Flow Network for Frame Interpolation",
        "checkpoints": {
            "stmfnet.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/stmfnet.pth",
            ],
        },
    },
    "cain": {
        "display_name": "CAIN",
        "description": "Channel Attention Is All You Need for Video Frame Interpolation",
        "checkpoints": {
            "cain.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/cain.pth",
            ],
            "pretrained_cain.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/pretrained_cain.pth",
            ],
        },
    },
    "atm": {
        "display_name": "ATM-VFI",
        "description": "Exploiting Attention-to-Motion via Transformer for VFI",
        "checkpoints": {
            "atm-vfi-base.pt": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/atm-vfi-base.pt",
            ],
        },
    },
    "momo": {
        "display_name": "MoMo",
        "description": "Disentangled Motion Modeling for Video Frame Interpolation",
        "checkpoints": {
            "momo.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/momo.pth",
            ],
        },
    },
    "sepconv": {
        "display_name": "Sepconv",
        "description": "Revisiting Adaptive Convolutions for Video Frame Interpolation",
        "checkpoints": {
            "sepconv.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/sepconv.pth",
            ],
        },
    },
    "eisai": {
        "display_name": "EISAI",
        "description": "Efficient Interpolation with Softsplat and DTM for Video Frame Interpolation",
        "checkpoints": {
            "eisai_ssl.pt": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/eisai_ssl.pt",
            ],
            "eisai_dtm.pt": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/eisai_dtm.pt",
            ],
            "eisai_anime_interp_full.ckpt": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/eisai_anime_interp_full.ckpt",
            ],
        },
    },
    "xvfi": {
        "display_name": "XVFI",
        "description": "eXtreme Video Frame Interpolation",
        "checkpoints": {
            "xvfi.pth": [
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/xvfi.pth",
            ],
        },
    },
}


class ModelManager(QObject):
    """Unified manager for all model types.
    
    Manages:
    - TensorRT engines (.engine)
    - ONNX models (.onnx) 
    - PyTorch checkpoints (.pth, .pt, .pkl, .ckpt)
    """

    engines_updated = pyqtSignal()
    models_updated = pyqtSignal()

    # Known model patterns for parsing
    SCALE_PATTERNS = ["2x", "4x", "1x"]
    PRECISION_PATTERNS = ["fp16", "fp32", "bf16"]
    
    # Supported checkpoint extensions
    CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".pkl", ".ckpt"}

    def __init__(self, config, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._config = config
        models_dir = config.get("paths.models_dir", "models")
        # Always use absolute path
        self._models_dir: Path = Path(models_dir).resolve()
        # Checkpoints are directly in models_dir (e.g., models/rife/rife49.pth)
        self._ckpts_dir: Path = self._models_dir
        
        # Storage for different model types
        self._engines: Dict[str, EngineInfo] = {}
        self._onnx_models: Dict[str, OnnxInfo] = {}
        self._model_types: Dict[str, ModelTypeInfo] = {}
        
        # Scan all model types
        self._scan_all()

    def set_models_dir(self, path: str):
        """Set the models directory path."""
        self._models_dir = Path(path).resolve()
        self._ckpts_dir = self._models_dir
        self._scan_all()

    def _scan_all(self):
        """Scan all model types."""
        self._scan_engines()
        self._scan_onnx()
        self._scan_checkpoints()

    # ==================== TensorRT Engines ====================
    
    def _scan_engines(self):
        """Scan the models directory for TensorRT engines."""
        self._engines.clear()

        if not self._models_dir.exists():
            return

        # Find all .engine files
        for engine_path in self._models_dir.glob("**/*.engine"):
            try:
                info = self._parse_engine(engine_path)
                self._engines[str(engine_path.resolve())] = info
            except Exception:
                continue

        self.engines_updated.emit()

    def _parse_engine(self, path: Path) -> EngineInfo:
        """Parse engine file to extract metadata."""
        name = path.stem
        size_mb = path.stat().st_size / (1024 * 1024)

        # Try to extract scale from name
        scale = None
        for pattern in self.SCALE_PATTERNS:
            if pattern in name.lower():
                scale = int(pattern.replace("x", ""))
                break

        # Try to extract precision from name
        precision = None
        for pattern in self.PRECISION_PATTERNS:
            if pattern in name.lower():
                precision = pattern.upper()
                break

        return EngineInfo(
            path=str(path.resolve()),
            name=name,
            size_mb=size_mb,
            scale=scale,
            precision=precision,
        )

    def get_available_engines(self) -> List[str]:
        """Get list of available engine file paths."""
        return list(self._engines.keys())

    def get_engine_info(self, path: str) -> Optional[EngineInfo]:
        """Get information about a specific engine."""
        return self._engines.get(path)

    def get_engines_by_scale(self, scale: int) -> List[EngineInfo]:
        """Get engines filtered by scale factor."""
        return [
            info for info in self._engines.values()
            if info.scale == scale
        ]

    def get_engines_by_precision(self, precision: str) -> List[EngineInfo]:
        """Get engines filtered by precision."""
        return [
            info for info in self._engines.values()
            if info.precision and info.precision.lower() == precision.lower()
        ]

    def get_engine_display_names(self) -> Dict[str, str]:
        """Get mapping of engine paths to display names."""
        return {
            path: info.display_name
            for path, info in self._engines.items()
        }

    def validate_engine(self, path: str) -> bool:
        """Validate that an engine file exists and is readable."""
        engine_path = Path(path)
        if not engine_path.exists():
            return False
        if not engine_path.is_file():
            return False
        if engine_path.suffix.lower() != ".engine":
            return False
        return True

    def get_recommended_engines(self) -> Dict[str, str]:
        """Get recommended engines for different use cases."""
        recommendations = {}

        # Anime upscaling
        for path, info in self._engines.items():
            if "animejanai" in info.name.lower():
                recommendations["anime_2x"] = path
                break

        # General upscaling
        for path, info in self._engines.items():
            if "realesrgan" in info.name.lower() and info.scale == 4:
                recommendations["general_4x"] = path
                break

        return recommendations

    # ==================== ONNX Models ====================
    
    def _scan_onnx(self):
        """Scan the models directory for ONNX files."""
        self._onnx_models.clear()

        if not self._models_dir.exists():
            return

        # Find all .onnx files
        for onnx_path in self._models_dir.glob("**/*.onnx"):
            try:
                info = self._parse_onnx(onnx_path)
                self._onnx_models[str(onnx_path.resolve())] = info
            except Exception:
                continue

    def _parse_onnx(self, path: Path) -> OnnxInfo:
        """Parse ONNX file to extract metadata."""
        name = path.stem
        size_mb = path.stat().st_size / (1024 * 1024)
        
        # Determine model type from name
        model_type = "scene_detect"
        if "sc_" in name.lower():
            model_type = "scene_detect"

        return OnnxInfo(
            path=str(path.resolve()),
            name=name,
            size_mb=size_mb,
            model_type=model_type,
        )

    def get_available_onnx(self) -> List[str]:
        """Get list of available ONNX file paths."""
        return list(self._onnx_models.keys())

    def get_onnx_info(self, path: str) -> Optional[OnnxInfo]:
        """Get information about a specific ONNX model."""
        return self._onnx_models.get(path)

    def get_onnx_by_name(self, name: str) -> Optional[str]:
        """Get ONNX path by filename."""
        for path, info in self._onnx_models.items():
            if info.name == name or Path(path).name == name:
                return path
        return None

    # ==================== PyTorch Checkpoints ====================
    
    def _scan_checkpoints(self):
        """Scan checkpoints directory for model files."""
        self._model_types.clear()
        
        # Scan each defined model type
        for model_type, definition in MODEL_DEFINITIONS.items():
            type_info = ModelTypeInfo(
                name=model_type,
                display_name=definition["display_name"],
                description=definition.get("description", ""),
            )
            
            # Check each checkpoint
            for ckpt_name, urls in definition["checkpoints"].items():
                ckpt_info = self._scan_checkpoint(model_type, ckpt_name, urls)
                type_info.checkpoints.append(ckpt_info)
            
            self._model_types[model_type] = type_info
        
        self.models_updated.emit()

    def _scan_checkpoint(
        self,
        model_type: str,
        ckpt_name: str,
        download_urls: List[str],
    ) -> CheckpointInfo:
        """Scan a single checkpoint."""
        ckpt_path = self._ckpts_dir / model_type / ckpt_name
        
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            return CheckpointInfo(
                name=ckpt_name,
                model_type=model_type,
                path=ckpt_path.resolve(),
                size_mb=round(size_mb, 2),
                status=ModelStatus.INSTALLED,
                download_urls=download_urls,
            )
        
        return CheckpointInfo(
            name=ckpt_name,
            model_type=model_type,
            status=ModelStatus.DOWNLOADABLE,
            download_urls=download_urls,
        )

    def get_model_types(self) -> Dict[str, ModelTypeInfo]:
        """Get all model types with their info."""
        return self._model_types

    def get_installed_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of all installed checkpoints."""
        installed = []
        for type_info in self._model_types.values():
            installed.extend(cp for cp in type_info.checkpoints if cp.is_installed)
        return installed

    def get_missing_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of checkpoints not yet installed."""
        missing = []
        for type_info in self._model_types.values():
            missing.extend(cp for cp in type_info.checkpoints if not cp.is_installed)
        return missing

    def get_checkpoint_path(self, model_type: str, ckpt_name: str) -> Optional[str]:
        """Get the absolute path to a checkpoint.
        
        Args:
            model_type: Model type name (e.g., "rife", "film")
            ckpt_name: Checkpoint filename (e.g., "rife49.pth")
            
        Returns:
            Absolute path to checkpoint if installed, None otherwise.
        """
        ckpt_path = self._ckpts_dir / model_type / ckpt_name
        if ckpt_path.exists():
            return str(ckpt_path.resolve())
        return None

    def get_checkpoint_info(self, model_type: str, ckpt_name: str) -> Optional[CheckpointInfo]:
        """Get info for a specific checkpoint."""
        if model_type not in self._model_types:
            return None
        
        for ckpt in self._model_types[model_type].checkpoints:
            if ckpt.name == ckpt_name:
                return ckpt
        return None

    def is_checkpoint_installed(self, model_type: str, ckpt_name: str) -> bool:
        """Check if a specific checkpoint is installed."""
        ckpt_path = self._ckpts_dir / model_type / ckpt_name
        return ckpt_path.exists()

    def get_model_types_summary(self) -> Dict[str, Dict]:
        """Get summary of all model types."""
        summary = {}
        for model_type, type_info in self._model_types.items():
            summary[model_type] = {
                "display_name": type_info.display_name,
                "installed": type_info.installed_count,
                "total": type_info.total_count,
            }
        return summary

    def get_total_size_mb(self) -> float:
        """Get total size of installed checkpoints in MB."""
        total = 0.0
        for ckpt in self.get_installed_checkpoints():
            total += ckpt.size_mb
        return round(total, 2)

    def scan_unknown_checkpoints(self) -> List[CheckpointInfo]:
        """Scan for checkpoints not in the definitions."""
        unknown = []
        
        if not self._ckpts_dir.exists():
            return unknown
        
        # Get all known checkpoint names
        known_names: Set[str] = set()
        for definition in MODEL_DEFINITIONS.values():
            known_names.update(definition["checkpoints"].keys())
        
        # Scan for unknown files
        for ckpt_file in self._ckpts_dir.glob("**/*"):
            if ckpt_file.is_file() and ckpt_file.suffix in self.CHECKPOINT_EXTENSIONS:
                if ckpt_file.name not in known_names:
                    model_type = ckpt_file.parent.name
                    size_mb = ckpt_file.stat().st_size / (1024 * 1024)
                    unknown.append(CheckpointInfo(
                        name=ckpt_file.name,
                        model_type=model_type,
                        path=ckpt_file.resolve(),
                        size_mb=round(size_mb, 2),
                        status=ModelStatus.INSTALLED,
                    ))
        
        return unknown

    # ==================== General Methods ====================
    
    def refresh(self):
        """Refresh all model lists."""
        self._scan_all()

    def refresh_engines(self):
        """Refresh only engines."""
        self._scan_engines()

    def refresh_checkpoints(self):
        """Refresh only checkpoints."""
        self._scan_checkpoints()
