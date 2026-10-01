"""FLAVR video frame interpolation model.

Reference:
    Kalluri, Tarun, et al. "FLAVR: Flow-Agnostic Video Representations for
    Fast Frame Interpolation." arXiv 2021.

Requires 4 input frames. Supports 2x, 4x, and 8x interpolation depending
on checkpoint:
    - FLAVR_2x.pth → 1 output frame (between I1-I2)
    - FLAVR_4x.pth → 3 output frames
    - FLAVR_8x.pth → 7 output frames
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import InputPadder

from .flavr_arch import UNet3D3D, InputPadder as FLAVRPadder


class FLAVRModel(PyTorchVFIModel):
    """FLAVR model wrapper.

    Notes:
        - Requires 4 input frames (MIN_INPUT_FRAMES = 4).
        - n_outputs determined from checkpoint: 1 (2x), 3 (4x), 7 (8x).
        - Uses batch mean normalisation internally.
    """

    MODEL_NAME = "flavr"
    SUPPORTED_VERSIONS = ["2x", "4x", "8x"]
    DEFAULT_VERSION = "2x"
    MIN_INPUT_FRAMES = 4

    # Map checkpoint names to (n_outputs, display_version)
    CKPT_CONFIGS = {
        "FLAVR_2x.pth": {"n_outputs": 1, "version": "2x"},
        "FLAVR_4x.pth": {"n_outputs": 3, "version": "4x"},
        "FLAVR_8x.pth": {"n_outputs": 7, "version": "8x"},
    }

    def __init__(self, config: Optional[VFIConfig] = None):
        if config is None:
            config = VFIConfig(model_type=ModelType.FLAVR)
        super().__init__(config)
        self._n_outputs: int = 1  # Determined at load time

    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """Load FLAVR checkpoint weights."""
        if self._model is not None:
            return

        # Determine n_outputs from checkpoint name
        ckpt_name = Path(checkpoint_path).name
        ckpt_config = self.CKPT_CONFIGS.get(ckpt_name, self.CKPT_CONFIGS["FLAVR_2x.pth"])
        self._n_outputs = int(ckpt_config["n_outputs"])

        # Build model
        self._model = UNet3D3D(
            "unet_18",
            n_inputs=4,
            n_outputs=self._n_outputs,
            join_type="concat",
            upmode="transpose",
        )

        # Load state dict
        if Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # Remove 'module.' prefix if present
                state_dict = {k.partition("module.")[-1]: v for k, v in state_dict.items()}
            self._model.load_state_dict(state_dict, strict=False)

        # FLAVR's 3D convolutions and batch norm require fp32
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

        FLAVR needs 4 frames as input. When only 2 are given,
        we duplicate frame0 as I0+I1 and frame1 as I2+I3.

        When n_outputs > 1, returns the frame closest to timestep.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frame0, frame1, squeeze = self._squeeze_batch(frame0, frame1)
        frame0, frame1 = self.prepare_frames(frame0, frame1)

        # Duplicate: I0=I1=frame0, I2=I3=frame1
        output = self._model([frame0, frame0, frame1, frame1])

        # Pick the frame closest to requested timestep
        if self._n_outputs > 1:
            idx = round(timestep * (self._n_outputs + 1)) - 1
            idx = max(0, min(idx, self._n_outputs - 1))
        else:
            idx = 0

        result = output[idx]

        if squeeze:
            result = result.squeeze(0)
        return result

    def interpolate_4frames(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
        I3: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Interpolate using all 4 frames.

        Args:
            I0: Frame at t-1
            I1: Frame at t
            I2: Frame at t+1
            I3: Frame at t+2

        Returns:
            Single frame (n_outputs=1) or list of frames (n_outputs>1)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        squeeze = I1.dim() == 3
        if squeeze:
            I0 = I0.unsqueeze(0)
            I1 = I1.unsqueeze(0)
            I2 = I2.unsqueeze(0)
            I3 = I3.unsqueeze(0)

        I0, I1, I2, I3 = self.prepare_frames(I0, I1, I2, I3)
        padder = FLAVRPadder(I1.shape, divisor=16)
        I0, I1, I2, I3 = [padder.pad(x) for x in (I0, I1, I2, I3)]

        with torch.no_grad():
            output = self._model([I0, I1, I2, I3])

        # Unpad
        output = [padder.unpad(o) for o in output]

        if self._n_outputs == 1:
            result = output[0]
            if squeeze:
                result = result.squeeze(0)
            return result

        if squeeze:
            output = [o.squeeze(0) for o in output]
        return output

    def interpolate_batch(
        self,
        frames: torch.Tensor,
        multiplier: int = 2,
        callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Interpolate frames in a video sequence using 4-frame sliding window."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frames = self.prepare_frames(frames)[0]
        N = frames.shape[0]
        output_frames = []

        for i in range(N - 3):
            out_list = self._model([
                frames[i],
                frames[i + 1],
                frames[i + 2],
                frames[i + 3],
            ])

            if i == 0:
                output_frames.append(frames[i])
                if self._n_outputs > 1:
                    output_frames.append(frames[i])
                output_frames.append(frames[i + 1])

            output_frames.extend(out_list)
            output_frames.append(frames[i + 2])

            if i == N - 4:
                if self._n_outputs > 1:
                    output_frames.append(frames[i + 3])
                output_frames.append(frames[i + 3])

            if callback:
                callback(i + 1, N - 3)

        return torch.stack(output_frames)
