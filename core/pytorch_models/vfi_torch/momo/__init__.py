"""
MoMo — Diffusion-based Video Frame Interpolation (ECCV 2024).

Port of MoMo VFI (https://github.com/fjlian/MoMo)
Reference: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

Limitation: model always outputs a single mid-frame (t=0.5). No timestep support.
Multi-frame interpolation requires recursive binary splitting.
Higher multipliers (4x, 8x) degrade quality due to error accumulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import torch

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import get_device


CKPT_NAMES = {
    "base": "momo-base.pth",
    "lite": "momo-lite.pth",
}


class MoMoModel(PyTorchVFIModel):
    """MoMo — Diffusion-based Video Frame Interpolation (ECCV 2024).

    Uses a DDPM denoising process to generate bidirectional optical flow,
    then synthesizes the interpolated frame from warped inputs.

    Supports base and lite variants.
    Always outputs a single interpolated frame at t=0.5.
    """

    MODEL_NAME = "momo"
    SUPPORTED_VERSIONS = ["base", "lite"]
    DEFAULT_VERSION = "base"
    MIN_INPUT_FRAMES = 2

    def __init__(self, config: Optional[VFIConfig] = None, device: Optional[torch.device] = None, dtype: Any = None):
        super().__init__(config, device, dtype)
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._is_loaded = False

    def load_model(self, checkpoint_path: Optional[str] = None, **kwargs) -> None:
        """Load MoMo model from checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file. If None, infers from config.
        """
        device = get_device(self._config.device)
        self._device = device

        version = self._config.model_version or "base"
        if checkpoint_path is None:
            checkpoint_path = str(
                Path(self._config.checkpoint_path or f"models/momo/{CKPT_NAMES.get(version, CKPT_NAMES['base'])}")
            )

        from .momo import MoMo
        from .synthesis import SynthesisNet

        ckpt_name = Path(checkpoint_path).name
        if "lite" in ckpt_name:
            dims = (96, 160)
        else:
            dims = (256, 256, 512)

        synth_model = SynthesisNet()
        model = MoMo(synth_model=synth_model, dims=dims)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._model = model.to(device)
        self._is_loaded = True

    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs,
    ) -> torch.Tensor:
        """Interpolate a single mid-frame between frame0 and frame1.

        Args:
            frame0: First frame [B, C, H, W], normalized to [0, 1]
            frame1: Second frame [B, C, H, W], normalized to [0, 1]
            timestep: Ignored — model always outputs t=0.5

        Returns:
            Interpolated frame [B, C, H, W], normalized to [0, 1]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure batch dimension
        squeeze_output = False
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
            frame1 = frame1.unsqueeze(0)
            squeeze_output = True

        # MoMo forward expects [B, 3, 2, H, W]
        x = torch.stack([frame0, frame1], dim=2)

        with torch.no_grad():
            result, _ = self._model(x, num_inference_steps=kwargs.get("num_inference_steps", 8))

        if squeeze_output:
            result = result.squeeze(0)
        return result
