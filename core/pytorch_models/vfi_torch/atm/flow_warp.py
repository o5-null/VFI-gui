"""
flow_warp — grid-sample based backward warping.

Copied from GMFlow: https://github.com/haofeixu/gmflow
"""

import torch
import torch.nn.functional as F


def coords_grid(b: int, h: int, w: int, homogeneous: bool = False, device=None) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    stacks = [x, y]
    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float()
    grid = grid[None].repeat(b, 1, 1, 1)
    if device is not None:
        grid = grid.to(device)
    return grid


def bilinear_sample(
    img: torch.Tensor, sample_coords: torch.Tensor, mode: str = "bilinear", padding_mode: str = "zeros", return_mask: bool = False
):
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)
    b, _, h, w = sample_coords.shape
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        return img, mask
    return img


def flow_warp(feature: torch.Tensor, flow: torch.Tensor, mask: bool = False, padding_mode: str = "zeros") -> torch.Tensor:
    """Backward warp feature with optical flow.

    Args:
        feature: [B, C, H, W]
        flow: [B, 2, H, W]
    Returns:
        Warped feature [B, C, H, W]
    """
    b, c, h, w = feature.size()
    assert flow.size(1) == 2
    grid = coords_grid(b, h, w).to(flow.device) + flow
    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)
