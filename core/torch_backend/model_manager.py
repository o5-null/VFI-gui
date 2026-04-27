"""Model loading, caching, and lifecycle management.

This module provides a centralized manager for VFI models, handling:
- Model loading and caching
- Automatic downloading from remote sources
- Model lifecycle management
"""

import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch

from .base import (
    VFIBaseModel,
    VFIModelInfo,
    DType,
    clear_cuda_cache,
)


# Default model download URLs
BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/",
]

# Fallback URLs for specific models (mirrors tried in order)
CKPT_FALLBACK_URLS: Dict[str, List[str]] = {
    "rife47.pth": [
        "https://huggingface.co/marduk191/rife/resolve/main/rife47.pth",
        "https://huggingface.co/wavespeed/misc/resolve/main/rife/rife47.pth",
    ],
    "rife49.pth": [
        "https://huggingface.co/marduk191/rife/resolve/main/rife49.pth",
        "https://huggingface.co/hfmaster/models-moved/resolve/main/rife/rife49.pth",
    ],
}


class ModelManager:
    """Centralized manager for VFI models.
    
    Handles model loading, caching, downloading, and lifecycle management.
    Models are cached by (model_type, checkpoint_name, dtype, compile) to avoid
    reloading weights on repeated use.
    
    Usage:
        manager = ModelManager(models_dir="path/to/models")
        
        # Get or load a model
        model = manager.get_model(
            model_type="rife",
            checkpoint_name="rife49.pth",
            dtype=DType.FLOAT16,
        )
        
        # Use the model
        result = model.interpolate(frame0, frame1, 0.5)
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        cache_dir: Optional[str] = None,
        auto_download: bool = True,
    ):
        """
        Args:
            models_dir: Directory to store downloaded models.
            cache_dir: Directory for model cache. Uses torch hub dir if None.
            auto_download: Whether to automatically download missing models.
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.auto_download = auto_download
        
        # Model cache: key = (model_type, ckpt_name, dtype, compile)
        # value = loaded model instance
        self._model_cache: Dict[Tuple, VFIBaseModel] = {}
        
        # Registry of available model types
        self._model_registry: Dict[str, Type[VFIBaseModel]] = {}
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def register_model_type(self, model_type: str, model_class: Type[VFIBaseModel]) -> None:
        """Register a model type with its implementation class.
        
        Args:
            model_type: Identifier for the model type (e.g., "rife", "amt").
            model_class: The model class implementing VFIBaseModel.
        """
        self._model_registry[model_type] = model_class
    
    def get_model(
        self,
        model_type: str,
        checkpoint_name: str,
        dtype: DType = DType.FLOAT32,
        torch_compile: bool = False,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> VFIBaseModel:
        """Get or load a model, using cache if available.
        
        Args:
            model_type: Type of model (e.g., "rife", "amt").
            checkpoint_name: Name of the checkpoint file.
            dtype: Data type for inference.
            torch_compile: Whether to compile the model.
            device: Target device. Auto-detected if None.
            **kwargs: Additional arguments passed to model constructor.
            
        Returns:
            Loaded model instance.
        """
        # Check cache
        cache_key = (model_type, checkpoint_name, dtype, torch_compile)
        if cache_key in self._model_cache:
            print(f"[ModelManager] Using cached model: {checkpoint_name} ({dtype.value})")
            return self._model_cache[cache_key]
        
        # Get model class from registry
        if model_type not in self._model_registry:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(self._model_registry.keys())}")
        
        model_class = self._model_registry[model_type]
        
        # Ensure checkpoint is available
        checkpoint_path = self._ensure_checkpoint(model_type, checkpoint_name)
        
        # Create and load model
        print(f"[ModelManager] Loading model: {checkpoint_name} ({dtype.value})"
              f"{' + torch.compile' if torch_compile else ''})")
        
        model = model_class(device=device, dtype=dtype, **kwargs)
        model.load_model(checkpoint_path)
        
        if torch_compile:
            model.compile()
        
        model.eval()
        
        # Cache the model
        self._model_cache[cache_key] = model
        print(f"[ModelManager] Model cached: {checkpoint_name}")
        
        return model
    
    def _ensure_checkpoint(self, model_type: str, checkpoint_name: str) -> str:
        """Ensure checkpoint file exists, downloading if necessary.
        
        Args:
            model_type: Type of model.
            checkpoint_name: Name of checkpoint file.
            
        Returns:
            Path to checkpoint file.
        """
        # Check in model-type-specific directory
        model_dir = self.models_dir / model_type
        checkpoint_path = model_dir / checkpoint_name
        
        if checkpoint_path.exists():
            return str(checkpoint_path)
        
        if not self.auto_download:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Set auto_download=True to download automatically."
            )
        
        # Download the checkpoint
        print(f"[ModelManager] Downloading checkpoint: {checkpoint_name}")
        return self._download_checkpoint(model_type, checkpoint_name)
    
    def _download_checkpoint(self, model_type: str, checkpoint_name: str) -> str:
        """Download checkpoint from remote sources.
        
        Tries multiple URLs in order until successful.
        
        Args:
            model_type: Type of model.
            checkpoint_name: Name of checkpoint file.
            
        Returns:
            Path to downloaded checkpoint.
        """
        model_dir = self.models_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Build list of URLs to try
        all_urls = [base + checkpoint_name for base in BASE_MODEL_DOWNLOAD_URLS]
        all_urls += CKPT_FALLBACK_URLS.get(checkpoint_name, [])
        
        error_strs = []
        for i, url in enumerate(all_urls):
            try:
                return self._load_file_from_url(url, model_dir)
            except Exception:
                traceback_str = traceback.format_exc()
                if i < len(all_urls) - 1:
                    print(f"[ModelManager] Failed to download from {url}. Trying next...")
                error_strs.append(f"Error downloading from {url}:\n{traceback_str}")
        
        error_str = "\n\n".join(error_strs)
        raise RuntimeError(
            f"Failed to download {checkpoint_name} from all sources:\n{error_str}"
        )
    
    def _load_file_from_url(
        self,
        url: str,
        model_dir: Path,
        progress: bool = True,
    ) -> str:
        """Download file from URL.
        
        Args:
            url: URL to download from.
            model_dir: Directory to save file.
            progress: Whether to show progress bar.
            
        Returns:
            Path to downloaded file.
        """
        from core.network import download_file

        model_dir.mkdir(parents=True, exist_ok=True)
        
        filename = os.path.basename(url.split("?")[0])
        cached_file = model_dir / filename
        
        if not cached_file.exists():
            print(f"[ModelManager] Downloading: {url}")
            download_file(url, cached_file)
            print(f"[ModelManager] Saved to: {cached_file}")
        
        return str(cached_file)
    
    def unload_model(
        self,
        model_type: str,
        checkpoint_name: str,
        dtype: DType = DType.FLOAT32,
        torch_compile: bool = False,
    ) -> None:
        """Unload a specific model from cache.
        
        Args:
            model_type: Type of model.
            checkpoint_name: Name of checkpoint.
            dtype: Data type.
            torch_compile: Whether model was compiled.
        """
        cache_key = (model_type, checkpoint_name, dtype, torch_compile)
        if cache_key in self._model_cache:
            model = self._model_cache.pop(cache_key)
            model.unload()
            print(f"[ModelManager] Unloaded model: {checkpoint_name}")
    
    def unload_all(self) -> None:
        """Unload all cached models."""
        for cache_key in list(self._model_cache.keys()):
            model = self._model_cache.pop(cache_key)
            model.unload()
        print("[ModelManager] All models unloaded")
    
    def get_cached_models(self) -> List[Tuple[str, str, DType, bool]]:
        """Get list of currently cached models.
        
        Returns:
            List of (model_type, checkpoint_name, dtype, torch_compile) tuples.
        """
        return [
            (key[0], key[1], key[2], key[3])
            for key in self._model_cache.keys()
        ]
    
    def get_available_checkpoints(self, model_type: str) -> List[str]:
        """Get list of available checkpoints for a model type.
        
        Args:
            model_type: Type of model.
            
        Returns:
            List of checkpoint filenames.
        """
        model_dir = self.models_dir / model_type
        if not model_dir.exists():
            return []
        
        # Find .pth and .pt files
        checkpoints = []
        for ext in [".pth", ".pt", ".ckpt"]:
            checkpoints.extend([f.name for f in model_dir.glob(f"*{ext}")])
        
        return sorted(checkpoints)
    
    def is_checkpoint_available(self, model_type: str, checkpoint_name: str) -> bool:
        """Check if a checkpoint is available locally.
        
        Args:
            model_type: Type of model.
            checkpoint_name: Name of checkpoint.
            
        Returns:
            True if checkpoint exists locally.
        """
        checkpoint_path = self.models_dir / model_type / checkpoint_name
        return checkpoint_path.exists()
    
    def clear_cache(self) -> None:
        """Clear CUDA cache and run garbage collection."""
        clear_cuda_cache()
        print("[ModelManager] Cache cleared")


# Global model manager instance
_global_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelManager()
    return _global_manager


def set_model_manager(manager: ModelManager) -> None:
    """Set the global model manager instance."""
    global _global_manager
    _global_manager = manager
