"""
ATM-VFI — Attention-to-Motion Video Frame Interpolation.

Port of ATM (https://github.com/HP-NTNU-VFI/ATM)
Reference: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

Limitation: model always outputs a single mid-frame (t=0.5). No timestep support.
Multi-frame interpolation requires recursive binary splitting.
Higher multipliers (4x, 8x) degrade quality due to error accumulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import get_device, InputPadder


CKPT_NAMES = {
    "base": "atm-vfi-base.pt",
    "lite": "atm-vfi-lite.pt",
    "base_pct": "atm-vfi-base-pct.pt",
}


class ATMVFIModel(PyTorchVFIModel):
    """ATM-VFI: Attention-to-Motion guided video frame interpolation.

    Supports base, lite, and base-pct variants.
    Always outputs a single interpolated frame at t=0.5.
    """

    MODEL_NAME = "atm"
    SUPPORTED_VERSIONS = ["base", "lite", "base_pct"]
    DEFAULT_VERSION = "base"
    MIN_INPUT_FRAMES = 2

    def __init__(self, config: VFIConfig):
        super().__init__(config)
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._is_loaded = False

    def load_model(self, checkpoint_path: Optional[str] = None, **kwargs) -> None:
        """Load ATM-VFI model from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file. If None, infers from config.
        """
        device = get_device(self._config.device)
        self._device = device

        version = self._config.model_version or "base"
        if checkpoint_path is None:
            checkpoint_path = str(Path(self._config.checkpoint_path or f"models/atm/{CKPT_NAMES.get(version, CKPT_NAMES['base'])}"))

        # Select network variant based on checkpoint name
        ckpt_name = Path(checkpoint_path).name
        if "lite" in ckpt_name:
            from .network_lite import Network
        else:
            from .network_base import Network

        model = Network(global_motion=True, ensemble_global_motion=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Remove incompatible keys (attn_mask, HW are runtime buffers, not weights)
        keys_to_remove = [k for k in state_dict if "attn_mask" in k or "HW" in k]
        for k in keys_to_remove:
            del state_dict[k]

        # Handle checkpoint naming variant: 'proj' vs 'proj_ref'
        for k in list(state_dict.keys()):
            if k.startswith("proj."):
                state_dict[k.replace("proj.", "proj_ref.", 1)] = state_dict.pop(k)

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

        # Pad to multiple of 64 (model's downsampling factor)
        padder = InputPadder(frame0.shape, divisor=64)
        f0_pad = padder.pad(frame0)
        f1_pad = padder.pad(frame1)

        with torch.no_grad():
            output = self._model(f0_pad, f1_pad)

        result = padder.unpad(output["I_t"])
        if squeeze_output:
            result = result.squeeze(0)
        return result

    def to(self, device: torch.device) -> "ATMVFIModel":
        if self._model is not None:
            self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self) -> "ATMVFIModel":
        if self._model is not None:
            self._model.eval()
        return self
