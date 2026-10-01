"""
Pure PyTorch Correlation Cost Volume.

Computes the correlation (dot product) between two feature maps in a
local neighborhood (9×9 = 81 positions), using efficient unfold + bmm.

Reference:
    PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
    (Sun et al., CVPR 2018)
"""

import torch
import torch.nn.functional as F


def costvol(tenOne: torch.Tensor, tenTwo: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """Compute correlation cost volume between two feature maps.

    For each pixel in tenOne, computes the dot product with features in a
    (2*ksize+1) x (2*ksize+1) neighborhood in tenTwo.

    Args:
        tenOne: Reference features [B, C, H, W]
        tenTwo: Target features [B, C, H, W]
        ksize: Neighborhood radius (default 3 → 7×7 window = 49 positions)

    Returns:
        Cost volume [B, (2*ksize+1)^2, H, W]
    """
    B, C, H, W = tenOne.shape
    pad = ksize

    # Unfold tenTwo into patches: [B, C*win, L] where win = (2*ksize+1)^2
    tenTwo_unfold = F.unfold(tenTwo, kernel_size=2 * ksize + 1, padding=pad)  # B, C*win, H*W
    _, C_win, L = tenTwo_unfold.shape
    win = C_win // C  # (2*ksize+1)^2

    # Reshape: [B, C, win, L] where L = H*W
    tenTwo_unfold = tenTwo_unfold.view(B, C, win, L)

    # Reshape tenOne: [B, C, L]
    tenOne_flat = tenOne.view(B, C, L)

    # Compute correlation: [B, win, L]
    # For each spatial position, dot product each C-channel vector
    corr = torch.einsum("b c l, b c w l -> b w l", tenOne_flat, tenTwo_unfold)

    # Normalize by feature dimension
    corr = corr / (float(C) ** 0.5)

    # Reshape to [B, win, H, W]
    corr = corr.view(B, win, H, W)

    return corr


def costvol_pwc(tenOne: torch.Tensor, tenTwo: torch.Tensor) -> torch.Tensor:
    """PWC-Net style cost volume with 9×9 search window (81 positions).

    Args:
        tenOne: Reference features [B, C, H, W]
        tenTwo: Target features [B, C, H, W]

    Returns:
        Cost volume [B, 81, H, W]
    """
    return costvol(tenOne, tenTwo, ksize=4)


class FunctionCostVol(torch.autograd.Function):
    """Correlation cost volume as an autograd function."""

    @staticmethod
    def forward(ctx, tenOne, tenTwo):
        result = costvol_pwc(tenOne, tenTwo)
        ctx.save_for_backward(tenOne, tenTwo, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        tenOne, tenTwo, _ = ctx.saved_tensors
        B, _, H, W = tenOne.shape
        ksize = 4
        pad = ksize
        win = (2 * ksize + 1) ** 2

        # Gradient w.r.t tenOne
        tenTwo_unfold = F.unfold(tenTwo, kernel_size=2 * ksize + 1, padding=pad)
        tenTwo_unfold = tenTwo_unfold.view(B, -1, win, H * W)
        tenTwo_unfold = tenTwo_unfold.permute(0, 2, 1, 3)  # B, win, C, H*W

        grad_out_flat = grad_output.view(B, win, H * W)  # B, win, H*W

        # grad_tenOne: [B, C, H*W]
        grad_tenOne = torch.einsum("b w c l, b w l -> b c l", tenTwo_unfold, grad_out_flat)
        grad_tenOne = grad_tenOne.view(B, -1, H, W) / (win**0.5)

        # Gradient w.r.t tenTwo
        tenOne_flat = tenOne.view(B, -1, H * W)  # B, C, H*W
        grad_out_reshaped = grad_output.view(B, win, 1, H * W)  # B, win, 1, H*W
        tenOne_expanded = tenOne_flat.unsqueeze(1)  # B, 1, C, H*W
        grad_tenTwo_patches = grad_out_reshaped * tenOne_expanded  # B, win, C, H*W
        grad_tenTwo_patches = grad_tenTwo_patches.permute(0, 2, 1, 3)  # B, C, win, H*W
        grad_tenTwo_patches = grad_tenTwo_patches.contiguous().view(B, -1, H * W)  # B, C*win, H*W

        grad_tenTwo = F.fold(
            grad_tenTwo_patches,
            output_size=(H, W),
            kernel_size=2 * ksize + 1,
            padding=pad,
        )
        grad_tenTwo = grad_tenTwo / (win**0.5)

        return grad_tenOne, grad_tenTwo


costvol_func = FunctionCostVol.apply
