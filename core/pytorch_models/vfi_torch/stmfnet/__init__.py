"""STMFNet (ST-MFNet) video frame interpolation model.

Reference:
    Danier, Duolikun, et al. "ST-MFNet: A Spatio-Temporal Multi-Flow Network
    for Frame Interpolation." CVPR 2022.

Requires 4 input frames; only supports 2× interpolation (timestep=0.5).
Checkpoint: stmfnet.pth (one file, ~145 MB).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ...base import DType
from ..ops.costvol import costvol_pwc
from ..utils import InputPadder

from .stmfnet_arch import STMFNet_Model


class STMFNetModel(PyTorchVFIModel):
    """ST-MFNet model wrapper.

    Notes:
        - Requires 4 input frames (MIN_INPUT_FRAMES = 4).
        - Only supports 2× interpolation (timestep is ignored, always t=0.5).
        - No padding needed — the model handles padding internally (multiples of 128).
    """

    MODEL_NAME = "stmfnet"
    SUPPORTED_VERSIONS = ["v1"]
    DEFAULT_VERSION = "v1"
    MIN_INPUT_FRAMES = 4  # Requires I0, I1, I2, I3

    def __init__(self, config: Optional[VFIConfig] = None):
        if config is None:
            config = VFIConfig(model_type=ModelType.STMFNET)
        super().__init__(config)

    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """Load STMFNet checkpoint weights."""
        if self._model is not None:
            return

        self._model = STMFNet_Model()

        if Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._model.load_state_dict(state_dict, strict=True)

        # STMFNet uses custom ops (AdaCoF, pure-PyTorch softsplat) that only work in fp32
        self._dtype = DType.FLOAT32
        self.torch_dtype = torch.float32
        self._model = self._model.to(self.device, dtype=torch.float32)
        self._model.eval()
        self._is_loaded = True

    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs,
    ) -> torch.Tensor:
        """Interpolate a single frame.

        STMFNet needs 4 frames as a group (I0, I1, I2, I3).
        Since interpolate() receives only frame0/frame1,
        we use frame0/1 as I1/I2 and duplicate I0=I1, I3=I2.
        For batch usage, use interpolate_4frames() instead.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frame0, frame1, squeeze = self._squeeze_batch(frame0, frame1)
        frame0, frame1 = self.prepare_frames(frame0, frame1)

        # Duplicate outermost frames: use frame0 as both I0+I1, frame1 as both I2+I3
        output = self._model(frame0, frame0, frame1, frame1)

        if squeeze:
            output = output.squeeze(0)
        return output

    def interpolate_4frames(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
        I3: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate using all 4 frames for optimal quality.

        Args:
            I0: Frame at t-1  [C, H, W] or [B, C, H, W]
            I1: Frame at t    [C, H, W] or [B, C, H, W]
            I2: Frame at t+1  [C, H, W] or [B, C, H, W]
            I3: Frame at t+2  [C, H, W] or [B, C, H, W]

        Returns:
            Interpolated frame at t+0.5 [C, H, W] or [B, C, H, W]
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Handle squeezing
        squeeze = I1.dim() == 3
        if squeeze:
            I0 = I0.unsqueeze(0)
            I1 = I1.unsqueeze(0)
            I2 = I2.unsqueeze(0)
            I3 = I3.unsqueeze(0)

        I0, I1, I2, I3 = self.prepare_frames(I0, I1, I2, I3)

        with torch.no_grad():
            output = self._model(I0, I1, I2, I3)

        if squeeze:
            output = output.squeeze(0)
        return output

    def interpolate_batch(
        self,
        frames: torch.Tensor,
        multiplier: int = 2,
        callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Interpolate frames in a video sequence using 4-frame sliding window.

        For each group of 4 consecutive frames, produces 1 interpolated frame.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frames = self.prepare_frames(frames)[0]
        N = frames.shape[0]
        output_frames = []

        for i in range(N - 3):
            out = self._model(
                frames[i],     # I0
                frames[i + 1], # I1
                frames[i + 2], # I2
                frames[i + 3], # I3
            )
            if i == 0:
                output_frames.append(frames[i])      # I0
                output_frames.append(frames[i + 1])  # I1
            output_frames.append(out)                 # interpolated
            output_frames.append(frames[i + 2])       # I2
            if i == N - 4:
                output_frames.append(frames[i + 3])   # I3

            if callback:
                callback(i + 1, N - 3)

        return torch.stack(output_frames)
