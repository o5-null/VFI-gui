"""
Backward warping utilities for MoMo VFI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class BackWarp(nn.Module):
    """Backward warping with grid_sample."""

    def __init__(self, clip: bool = False, interpolation: str = "bilinear", align_corners: bool = False):
        super().__init__()
        self.clip = clip
        self.interpolation = interpolation
        self.align_corners = align_corners

    def forward(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img.shape
        u = flow[:, 0]
        v = flow[:, 1]

        gridY, gridX = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
        x = repeat(gridX.float(), "h w -> b h w", b=b) + u
        y = repeat(gridY.float(), "h w -> b h w", b=b) + v

        # Normalize to [-1, 1]
        x = (x / w) * 2 - 1
        y = (y / h) * 2 - 1

        grid = torch.stack((x, y), dim=-1)
        padding_mode = "border" if self.clip else "zeros"
        return F.grid_sample(img, grid, mode=self.interpolation, align_corners=self.align_corners, padding_mode=padding_mode)
