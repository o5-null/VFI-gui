"""
Pure-PyTorch softsplat — Forward splatting with scatter-add.

Implements the forward splatting operation used by GMFSS Fortuna for
feature warping. Unlike backward warping (grid_sample), forward splatting
propagates each source pixel's value to its flow-destination position.

Supports sum/avg/linear/soft modes matching the original taichi/CUDA ops.

Reference:
    https://github.com/98mxr/GMFSS_Fortuna
    https://github.com/viser-Weber/SoftSplat
"""

from __future__ import annotations

import torch


def softsplat(
    tenIn: torch.Tensor,
    tenFlow: torch.Tensor,
    tenMetric: torch.Tensor | None = None,
    strMode: str = "sum",
) -> torch.Tensor:
    """Forward splatting: splat each source pixel to flow-destination.

    Pure PyTorch implementation using scatter-add with bilinear
    distribution weights. Supports inference only (no gradient).

    Args:
        tenIn: Input tensor [B, C, H, W]
        tenFlow: Optical flow [B, 2, H, W]
        tenMetric: Metric/confidence [B, 1, H, W] (required for linear/soft)
        strMode: "sum" | "avg" | "linear" | "soft" (optional "-addeps"/"-zeroeps"/"-clipeps" suffix)

    Returns:
        Splatted tensor [B, C, H, W]
    """
    assert strMode.split("-")[0] in ("sum", "avg", "linear", "soft")

    B, C, H, W = tenIn.shape
    device = tenIn.device

    # --- Apply metric weighting before splatting ---
    if strMode == "sum":
        data = tenIn.contiguous()
    elif strMode == "avg":
        ones = tenIn.new_ones(B, 1, H, W)
        data = torch.cat([tenIn, ones], dim=1).contiguous()
    elif strMode.split("-")[0] == "linear":
        assert tenMetric is not None
        data = torch.cat([tenIn * tenMetric, tenMetric], dim=1).contiguous()
    elif strMode.split("-")[0] == "soft":
        assert tenMetric is not None
        w = tenMetric.exp()
        data = torch.cat([tenIn * w, w], dim=1).contiguous()

    C_all = data.shape[1]

    # --- Scatter-based forward splatting ---
    # For each input pixel (y, x) with flow (fy, fx), we distribute
    # its value to the 4 surrounding output pixels using bilinear weights.

    # Source grid
    src_y = (
        torch.arange(H, device=device, dtype=torch.float32)
        .view(1, 1, H, 1)
        .expand(B, 1, H, W)
    )
    src_x = (
        torch.arange(W, device=device, dtype=torch.float32)
        .view(1, 1, 1, W)
        .expand(B, 1, H, W)
    )

    # Target positions
    tgt_x = src_x + tenFlow[:, 0:1, :, :]  # B,1,H,W
    tgt_y = src_y + tenFlow[:, 1:2, :, :]  # B,1,H,W

    # Integer corners (unclamped for bounds check)
    ix0 = torch.floor(tgt_x).long()
    iy0 = torch.floor(tgt_y).long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Clamped for actual indexing
    ix0_c = ix0.clamp(0, W - 1)
    iy0_c = iy0.clamp(0, H - 1)
    ix1_c = ix1.clamp(0, W - 1)
    iy1_c = iy1.clamp(0, H - 1)

    # Bilinear weights (use float for precision)
    w_tl = ((ix1.float() - tgt_x) * (iy1.float() - tgt_y)).clamp(0, 1)
    w_tr = ((tgt_x - ix0.float()) * (iy1.float() - tgt_y)).clamp(0, 1)
    w_bl = ((ix1.float() - tgt_x) * (tgt_y - iy0.float())).clamp(0, 1)
    w_br = ((tgt_x - ix0.float()) * (tgt_y - iy0.float())).clamp(0, 1)

    # Mask out-of-bounds contributions
    valid_tl = (ix0 >= 0) & (ix0 < W) & (iy0 >= 0) & (iy0 < H)
    valid_tr = (ix1 >= 0) & (ix1 < W) & (iy1 >= 0) & (iy1 < H)
    valid_bl = (ix0 >= 0) & (ix0 < W) & (iy1 >= 0) & (iy1 < H)
    valid_br = (ix1 >= 0) & (ix1 < W) & (iy0 >= 0) & (iy0 < H)

    # Flatten indices: idx = iy * W + ix
    HW = H * W
    data_flat = data.view(B, C_all, HW)  # B, C_all, HW

    def _scatter(
        b: int,
        iy_idx: torch.Tensor,
        ix_idx: torch.Tensor,
        weight: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter contributions for one corner into output.

        Returns contributions [C_all, HW] for batch b.
        """
        # Flatten target index: [1, HW]
        idx = (iy_idx[b, 0] * W + ix_idx[b, 0])  # HW
        w = weight[b, 0] * valid[b, 0].float()  # HW
        # Add weighted source values to output
        contrib = data_flat[b] * w.unsqueeze(0)  # C_all, HW
        return idx, contrib

    out_flat = torch.zeros(B, C_all, HW, device=device, dtype=data.dtype)

    for b in range(B):
        for iy_c, ix_c, w, v in [
            (iy0_c, ix0_c, w_tl, valid_tl),
            (iy1_c, ix1_c, w_tr, valid_tr),
            (iy1_c, ix0_c, w_bl, valid_bl),
            (iy0_c, ix1_c, w_br, valid_br),
        ]:
            idx = (iy_c[b, 0] * W + ix_c[b, 0])  # HW
            wgt = w[b, 0] * v[b, 0].float()  # HW
            out_flat[b].index_add_(1, idx, data_flat[b] * wgt.unsqueeze(0))

    out = out_flat.view(B, C_all, H, W)

    # --- Normalize for avg/linear/soft modes ---
    if strMode.split("-")[0] in ("avg", "linear", "soft"):
        denom = out[:, -1:, :, :]

        eps_suffix = strMode.split("-")[1] if "-" in strMode else ""
        if eps_suffix == "zeroeps":
            denom[denom == 0.0] = 1.0
        elif eps_suffix == "clipeps":
            denom = denom.clip(1e-7, None)
        else:
            denom = denom + 1e-7

        out = out[:, :-1, :, :] / denom

    return out
