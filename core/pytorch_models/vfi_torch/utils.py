"""
Utility functions for VFI models.
"""

import gc
import torch
import numpy as np
import einops
from pathlib import Path
from typing import Optional, Dict, Any, Union
from urllib.request import urlretrieve
from tqdm import tqdm


# Model download URLs
BASE_MODEL_URLS = {
    "rife": "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "film": "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/",
    "amt": "https://huggingface.co/lalala125/AMT/resolve/main/",
}


def preprocess_frames(frames: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Preprocess frames for model input.
    
    Converts NHWC uint8 numpy array to NCHW float32 tensor.
    
    Args:
        frames: Input frames [N, H, W, C] in NHWC format, uint8
        device: Target device (default: cuda if available, else cpu)
        
    Returns:
        Preprocessed tensor [N, C, H, W] in NCHW format, float32
    """
    if device is None:
        device = get_device()
    
    # Handle different input formats
    if isinstance(frames, torch.Tensor):
        tensor = frames
    else:
        # Convert numpy array to tensor
        tensor = torch.from_numpy(frames)
    
    # Ensure float32
    if tensor.dtype != torch.float32:
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()
    
    # Convert NHWC to NCHW
    if tensor.dim() == 4 and tensor.shape[-1] in [1, 3, 4]:
        # Assume NHWC format
        tensor = tensor.permute(0, 3, 1, 2)
    
    return tensor.to(device)


def postprocess_frames(tensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess model output to frames.
    
    Converts NCHW float32 tensor to NHWC uint8 numpy array.
    
    Args:
        tensor: Model output [N, C, H, W] in NCHW format, float32
        
    Returns:
        Output frames [N, H, W, C] in NHWC format, uint8
    """
    # Move to CPU if needed
    if tensor.device != torch.device("cpu"):
        tensor = tensor.cpu()
    
    # Convert NCHW to NHWC
    if tensor.dim() == 4:
        tensor = tensor.permute(0, 2, 3, 1)
    
    # Clamp to [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to uint8
    frames = (tensor.numpy() * 255).astype(np.uint8)
    
    return frames


# ============================================================================
# Tensor-based preprocessing (for backward compatibility with torch_backend/base.py)
# ============================================================================

def preprocess_frames_tensor(frames: torch.Tensor) -> torch.Tensor:
    """Preprocess frames from NHWC to NCHW format (tensor version).
    
    Args:
        frames: Input frames in NHWC format [N, H, W, C]
        
    Returns:
        Frames in NCHW format [N, C, H, W]
    """
    # Handle RGBA by taking only RGB channels
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")


def postprocess_frames_tensor(frames: torch.Tensor) -> torch.Tensor:
    """Postprocess frames from NCHW to NHWC format (tensor version).
    
    Args:
        frames: Input frames in NCHW format [N, C, H, W]
        
    Returns:
        Frames in NHWC format [N, H, W, C]
    """
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()


def get_device(device: str = "auto", device_id: int = 0) -> torch.device:
    """
    Get the torch device.
    
    Supports CUDA (NVIDIA), ROCm (AMD), XPU (Intel), and CPU.
    ROCm devices use the "cuda" namespace in PyTorch.
    
    Args:
        device: Device string ("auto", "cuda", "cuda:0", "xpu:0", "cpu")
        device_id: GPU device ID (used when device="auto")
        
    Returns:
        torch.device instance
    """
    # Handle explicit device strings
    if device != "auto":
        return torch.device(device)
    
    # Auto-detect best available device
    # Check for CUDA/ROCm (both use torch.cuda)
    if torch.cuda.is_available():
        # Distinguish CUDA vs ROCm via torch.version.hip
        return torch.device(f"cuda:{device_id}")
    
    # Check for Intel XPU
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return torch.device(f"xpu:{device_id}")
    
    # Check for Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")


def clear_cache():
    """Clear GPU cache and run garbage collection.
    
    Supports CUDA, ROCm, and XPU devices.
    """
    gc.collect()
    
    # CUDA/ROCm both use torch.cuda namespace
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Intel XPU
    elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()


def load_model_weights(
    checkpoint_path: str,
    model_key: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Union[Dict[str, Any], torch.jit.ScriptModule]:
    """
    Load model weights from checkpoint file.
    Supports both state_dict checkpoints and TorchScript models.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_key: Key to extract from checkpoint (if nested)
        device: Target device
        
    Returns:
        State dict or TorchScript module
    """
    if device is None:
        device = get_device()
    
    # Check if this is a TorchScript model by examining the file
    # TorchScript files are zip files with specific structure
    try:
        import zipfile
        if zipfile.is_zipfile(checkpoint_path):
            with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                # TorchScript archives contain specific files
                if 'model.pkl' in zf.namelist() or 'constants.pkl' in zf.namelist():
                    # This is a TorchScript model, use torch.jit.load
                    return torch.jit.load(checkpoint_path, map_location=device)
    except Exception:
        pass
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if model_key and model_key in checkpoint:
            return checkpoint[model_key]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    
    return checkpoint


def download_model(
    model_name: str,
    filename: str,
    save_dir: str,
    url: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download a model file if it doesn't exist.
    
    Args:
        model_name: Model type name (for URL lookup)
        filename: Model filename
        save_dir: Directory to save the model
        url: Direct download URL (overrides model_name lookup)
        force: Force re-download
        
    Returns:
        Path to downloaded file
    """
    save_path = Path(save_dir) / filename
    
    if save_path.exists() and not force:
        return str(save_path)
    
    # Get download URL
    if url is None:
        if model_name in BASE_MODEL_URLS:
            url = BASE_MODEL_URLS[model_name] + filename
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    # Create directory
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar
    print(f"Downloading {filename}...")
    
    class TqdmProgress(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with TqdmProgress(unit='B', unit_scale=True, miniters=1) as pbar:
        urlretrieve(url, save_path, reporthook=pbar.update_to)
    
    return str(save_path)


class InputPadder:
    """
    Pads input to be divisible by a factor.
    Useful for models that require specific input dimensions.
    """
    
    def __init__(self, dims: tuple, divisor: int = 16):
        """
        Initialize padder.
        
        Args:
            dims: Input dimensions (H, W)
            divisor: Padding divisor
        """
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [
            pad_wd // 2, pad_wd - pad_wd // 2,
            pad_ht // 2, pad_ht - pad_ht // 2,
        ]
    
    def pad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad tensor."""
        return torch.nn.functional.pad(tensor, self._pad, mode='replicate')
    
    def unpad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Remove padding from tensor."""
        return tensor[
            ..., 
            self._pad[2]: tensor.shape[-2] - self._pad[3],
            self._pad[0]: tensor.shape[-1] - self._pad[1]
        ]


# ============================================================================
# Shared warp() — backward warping with grid_sample
# ============================================================================

_backwarp_tenGrid: dict = {}


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Warp a tensor using an optical flow field.

    Uses a cached coordinate grid keyed by (device, flow_size) for efficiency.

    Args:
        tenInput: Input tensor [B, C, H, W]
        tenFlow:  Flow tensor [B, 2, H, W]

    Returns:
        Warped tensor [B, C, H, W]
    """
    device = tenFlow.device
    k = (str(device), str(tenFlow.size()))

    if k not in _backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        _backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
    ], 1)

    g = (_backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

    if g.dtype != tenInput.dtype:
        g = g.to(tenInput.dtype)

    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# ============================================================================
# Shared ResBlock — parameterizable residual block
# ============================================================================


class ResBlock(torch.nn.Module):
    """Residual block with configurable activation.

    Each ResBlock consists of: conv1 → activation → conv2 → skip + activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        act_layer: Activation layer class (default: ``torch.nn.ReLU``)
        act_kwargs: Keyword arguments for the activation layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: type = torch.nn.ReLU,
        act_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        kwargs = act_kwargs or {}
        self.activation = act_layer(**kwargs)

        if in_channels != out_channels:
            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return self.activation(out + self.skip(x))


# ============================================================================
# Shared timestep tensor creation
# ============================================================================


def make_timestep_tensor(
    batch_size: int,
    timestep: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    ndim: int = 2,
) -> torch.Tensor:
    """Create a timestep tensor of the requested shape.

    Args:
        batch_size: Batch dimension
        timestep:  Scalar timestep value (0.0 to 1.0)
        device:    Target device
        dtype:     Target dtype
        ndim:      Number of dimensions: 2 → [B, 1], 4 → [B, 1, 1, 1]

    Returns:
        Tensor filled with `timestep` in the requested shape
    """
    if ndim == 2:
        return torch.full((batch_size, 1), timestep, device=device, dtype=dtype)
    elif ndim == 4:
        return torch.full((batch_size, 1, 1, 1), timestep, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported ndim: {ndim} (supported: 2, 4)")
