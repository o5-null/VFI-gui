"""Shared RIFE input packing for inference backends.

Builds the single NCHW ``[1, C, H, W]`` float32 tensor that vs-mlrt RIFE
exports expect. RGB values are fed as-is in ``[0, 1]`` (no normalization).

Two channel layouts:
    - ``C == 7``  (implementation 2 / ``rife_v2``, preferred):
      ``[frame0(3ch), frame1(3ch), timestep_map(1ch)]``
    - ``C == 11`` (implementation 1 / ``rife``): the 7 above plus 4 meshgrid
      planes, in order: horizontal, vertical, multiplier_h, multiplier_w.

Frames may be torch tensors or numpy arrays with shape ``[3, H, W]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(frame: Any) -> np.ndarray:
    """Convert a torch tensor or numpy array to a float32 numpy [3, H, W]."""
    # torch tensors have .detach(); numpy arrays do not.
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame, dtype=np.float32)


def pack_rife_input(
    frame0: Any,
    frame1: Any,
    timestep: float,
    in_channels: int,
) -> np.ndarray:
    """Pack a frame pair into the RIFE NCHW ONNX/TensorRT input tensor.

    Args:
        frame0: torch.Tensor or numpy [3, H, W] float32 RGB in [0, 1].
        frame1: torch.Tensor or numpy [3, H, W] float32 RGB in [0, 1].
        timestep: interpolation position in [0, 1].
        in_channels: model input channel count (7 or 11).

    Returns:
        Contiguous numpy array ``[1, C, H, W]`` float32.
    """
    f0 = _to_numpy(frame0)
    f1 = _to_numpy(frame1)
    _, h, w = f0.shape

    # Timestep map: constant plane filled with the interpolation position.
    tmap = np.full((1, h, w), float(timestep), np.float32)

    if in_channels == 11:
        # Build the 4 meshgrid planes expected by implementation 1.
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        horizontal = np.broadcast_to(
            (2.0 * xs / max(w - 1, 1) - 1.0)[None, :], (h, w)
        )
        vertical = np.broadcast_to(
            (2.0 * ys / max(h - 1, 1) - 1.0)[:, None], (h, w)
        )
        multiplier_h = np.full((h, w), 2.0 / max(w - 1, 1), np.float32)
        multiplier_w = np.full((h, w), 2.0 / max(h - 1, 1), np.float32)
        stacked = np.concatenate(
            [
                f0,
                f1,
                tmap,
                horizontal[None, :, :],
                vertical[None, :, :],
                multiplier_h[None, :, :],
                multiplier_w[None, :, :],
            ],
            axis=0,
        )
    else:
        # C == 7 layout (preferred): [f0(3), f1(3), tmap(1)]
        stacked = np.concatenate([f0, f1, tmap], axis=0)

    return np.ascontiguousarray(stacked[None, ...])
