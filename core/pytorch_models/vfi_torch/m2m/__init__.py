"""
M2M-VFI — Many-to-Many Splatting for Efficient Video Frame Interpolation.

Pure PyTorch port of M2M (https://github.com/feinanshan/M2M_VFI)
Reference: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

Supports arbitrary interpolation timesteps (not just t=0.5).
Multi-frame interpolation uses the model's built-in multi-timestep support.

Paper: Hu et al., "Many-to-many Splatting for Efficient Video Frame Interpolation", CVPR 2022
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import get_device, InputPadder, make_timestep_tensor


CKPT_NAMES = {
    "default": "M2M.pth",
}


class M2MVFIModel(PyTorchVFIModel):
    """M2M-VFI: Many-to-Many Splatting for Video Frame Interpolation.

    Uses PWC-Net style pyramidal flow estimation + multi-branch motion refinement +
    many-to-many softmax splatting. Supports arbitrary interpolation timesteps
    via the model's native fltTimes parameter.

    Only the "default" version with 4 branches is supported.
    """

    MODEL_NAME = "m2m"
    SUPPORTED_VERSIONS = ["default"]
    DEFAULT_VERSION = "default"
    MIN_INPUT_FRAMES = 2

    def __init__(self, config: VFIConfig):
        super().__init__(config)
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._is_loaded = False

    def load_model(self, checkpoint_path: Optional[str] = None, **kwargs) -> None:
        """Load M2M model from checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file. If None, infers from config.
        """
        device = get_device(self._config.device)
        self._device = device

        if checkpoint_path is None:
            checkpoint_path = str(
                Path(self._config.checkpoint_path or f"models/m2m/{CKPT_NAMES['default']}")
            )

        from .arch import M2M_PWC

        model = M2M_PWC()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)

        # Remove incompatible keys if present
        model.load_state_dict(state_dict, strict=False)
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
        """Interpolate a single frame between frame0 and frame1.

        Unlike ATM/MoMo, M2M natively supports arbitrary timesteps.

        Args:
            frame0: First frame [B, C, H, W], normalized to [0, 1]
            frame1: Second frame [B, C, H, W], normalized to [0, 1]
            timestep: Interpolation timestep (0.0 to 1.0). Default 0.5.

        Returns:
            Interpolated frame [B, C, H, W], normalized to [0, 1]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frame0, frame1, squeeze_output = self._squeeze_batch(frame0, frame1)

        # M2M handles its own padding internally (to ratio*16 multiples)
        with torch.no_grad():
            # Create timestep tensor for the batch
            fltTime = make_timestep_tensor(frame0.shape[0], timestep, frame0.device, frame0.dtype, ndim=4)
            outputs = self._model(frame0, frame1, fltTimes=[fltTime])

        result = outputs[0]
        if squeeze_output:
            result = result.squeeze(0)
        return result

    def to(self, device: torch.device) -> "M2MVFIModel":
        if self._model is not None:
            self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self) -> "M2MVFIModel":
        if self._model is not None:
            self._model.eval()
        return self
