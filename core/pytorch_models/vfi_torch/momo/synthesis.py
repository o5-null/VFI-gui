"""
SynthesisNet — frame synthesis from optical flow.

Multi-scale recurrent synthesis that warps and blends frames guided by optical flow.
Reference: MoMo (ECCV 2024)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .flow import BackWarp


class SynthesisNet(nn.Module):
    """Multi-scale recurrent frame synthesis from optical flow."""

    def __init__(
        self,
        latent_dim: int = 32,
        recurrent_min_res: int = 64,
        normalize_inputs: bool = True,
        align_corners: bool = False,
        padding: str = "replicate",
        interpolation: str = "bicubic",
        act: type = nn.GELU,
        antialias: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_min_res = recurrent_min_res
        self.normalize_inputs = normalize_inputs
        self.align_corners = align_corners
        self.padding = padding
        self.interpolation = interpolation
        self.antialias = antialias

        self.bwarp = BackWarp(interpolation=interpolation, align_corners=align_corners)

        dim = latent_dim * 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, latent_dim, 3, 1, 1, padding_mode=padding),
            act(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1, padding_mode=padding),
            act(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1, padding_mode=padding),
        )

        self.decoder = nn.Sequential(
            act(),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode=padding),
            act(),
            nn.Conv2d(dim, 4, 3, 1, 1, padding_mode=padding),
        )

        self.blender = UNet(in_channels=4 + 3 + latent_dim * 2, out_channels=dim, n_lvls=2, dim=dim, act=act)

    def preprocess(self, x, eps=1e-8, stats=None):
        if self.normalize_inputs:
            if stats is None:
                x_flat = x.view(x.shape[0], -1)
                _mean, _std = torch.mean(x_flat, dim=-1), torch.std(x_flat, dim=-1) + eps
                while len(_mean.shape) < len(x.shape):
                    _mean, _std = _mean.unsqueeze(-1), _std.unsqueeze(-1)
                return (x - _mean) / _std, (_mean, _std)
            else:
                _mean, _std = stats
                return (x - _mean) / _std, None
        return x * 2 - 1, None

    def postprocess(self, x, stats=None):
        if self.normalize_inputs and stats is not None:
            _mean, _std = stats
            return torch.clamp((x * _std) + _mean, 0, 1)
        return torch.clamp((x + 1) / 2, 0, 1)

    def get_n_lvls(self, size):
        return int(np.ceil(np.log2(min(size) / self.recurrent_min_res))) + 1

    def decode2rgb(self, xt, warped_xt_rgb):
        output = self.decoder(xt)
        res_rgb, blend_w = output.split([3, 1], dim=1)
        blend_w = torch.sigmoid(blend_w)
        blend_w = torch.stack([blend_w, 1 - blend_w], dim=2)
        synth_out = torch.sum(warped_xt_rgb * blend_w, dim=2) + res_rgb
        return synth_out

    def forward(self, x, flows):
        """Synthesize interpolated frame from two input frames and estimated flows.

        Args:
            x: Input frames stacked as [B, 3, 2, H, W]
            flows: Estimated flows [B, 4, H, W] (2 flows × 2 channels)
        Returns:
            Synthesized frame [B, 3, H, W], normalized to [0, 1]
        """
        x, x_norm_stats = self.preprocess(rearrange(x, "b c f h w -> b (f c) h w"))
        x = rearrange(x, "b (f c) h w -> (f b) c h w", f=2)
        flows = rearrange(flows, "b (f c) h w -> (f b) c h w", f=2)
        n_lvls = self.get_n_lvls(flows.shape[-2:])

        for i in range(n_lvls - 1, -1, -1):
            scale_factor = 1 / (2**i)
            x_lvl = F.interpolate(x, scale_factor=scale_factor, mode=self.interpolation, align_corners=self.align_corners, antialias=self.antialias)
            flows_lvl = F.interpolate(flows, scale_factor=scale_factor, mode=self.interpolation, align_corners=self.align_corners, antialias=self.antialias) * scale_factor

            warped_xt0_rgb, warped_xt1_rgb = self.bwarp(x_lvl, flows_lvl).chunk(2, dim=0)
            warped_xt_rgb = torch.stack([warped_xt0_rgb, warped_xt1_rgb], dim=2)

            enc_x_lvl = self.encoder(x_lvl)
            if i == n_lvls - 1:
                xt = (warped_xt0_rgb + warped_xt1_rgb) / 2
            else:
                xt = F.interpolate(xt, size=flows_lvl.shape[-2:], mode=self.interpolation, align_corners=self.align_corners, antialias=self.antialias)

            warped_xs_lvl = self.bwarp(enc_x_lvl, flows_lvl)
            warped_xs_lvl = rearrange(warped_xs_lvl, "(f b) c h w -> b (f c) h w", f=2)

            xt = self.blender(torch.cat([xt, warped_xs_lvl, rearrange(flows_lvl, "(f b) c h w -> b (f c) h w", f=2)], dim=1))
            xt = self.decode2rgb(xt, warped_xt_rgb)

        return self.postprocess(xt, stats=x_norm_stats)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act=nn.ReLU, padding_mode="replicate"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, padding_mode=padding_mode),
            act(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", padding_mode=padding_mode),
            act(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act=nn.ReLU, padding_mode="replicate", interpolation="bicubic"):
        super().__init__()
        self.interpolation = interpolation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding="same", padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size, 1, padding="same", padding_mode=padding_mode)
        self.act = act()

    def forward(self, x, skip):
        _, _, h, w = skip.shape
        x = F.interpolate(x, size=(h, w), mode=self.interpolation)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(torch.cat((x, skip), 1)))
        return x


class UNet(nn.Module):
    """Lightweight UNet used as the blending network in SynthesisNet."""

    def __init__(self, in_channels, out_channels, n_lvls=4, dim=32, max_dim=None, act=nn.ReLU, padding_mode="replicate", interpolation="bicubic"):
        super().__init__()
        self.n_lvls = n_lvls
        self.in_feats = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, 1, 1, padding_mode=padding_mode),
            act(),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode=padding_mode),
            act(),
        )

        dim_tracker = [dim]
        down_blocks = []
        for i in range(n_lvls):
            prev_dim = dim_tracker[i]
            next_dim = prev_dim * 2
            if max_dim is not None:
                next_dim = min(max_dim, next_dim)
            dim_tracker.append(next_dim)
            down_blocks.append(DownBlock(prev_dim, next_dim, kernel_size=3, act=act, padding_mode=padding_mode))
        self.down_blocks = nn.ModuleList(down_blocks)

        dim_tracker_rev = dim_tracker[::-1]
        up_blocks = []
        for i in range(n_lvls):
            up_blocks.append(UpBlock(dim_tracker_rev[i], dim_tracker_rev[i + 1], kernel_size=3, act=act, padding_mode=padding_mode, interpolation=interpolation))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.to_out = nn.Conv2d(dim_tracker_rev[-1], out_channels, 3, 1, 1, padding_mode=padding_mode)

    def forward(self, x):
        mid_results = [self.in_feats(x)]
        for down_block in self.down_blocks:
            mid_results.append(down_block(mid_results[-1]))
        h = mid_results.pop()
        for up_block in self.up_blocks:
            h = up_block(h, mid_results.pop())
        return self.to_out(h)
