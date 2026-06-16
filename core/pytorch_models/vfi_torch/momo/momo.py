"""
MoMo — Diffusion-based Video Frame Interpolation.

Diffusion model that generates bidirectional flow maps via iterative denoising,
then synthesizes the interpolated frame via the synthesis network.

Reference: MoMo (ECCV 2024), https://momo-vfi.github.io/
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange

from .unet import ConvexUpUNet2DModel


class MoMo(nn.Module):
    """MoMo diffusion model for video frame interpolation.

    Generates bidirectional flow maps via DDPM denoising, then
    synthesizes the interpolated frame via the synthesis network.
    """

    def __init__(
        self,
        synth_model: nn.Module = None,
        dims=(256, 256, 512),
        T=1000,
        flow_scaler=128,
        prediction_type="sample",
        beta_schedule="linear",
        use_attn=False,
        norm_in=True,
        padding="replicate",
    ):
        super().__init__()

        # Synthesis model (frozen)
        self.synth_model = synth_model
        if self.synth_model is not None:
            for p in self.synth_model.parameters():
                p.requires_grad = False

        # UNet for denoising
        self.dims = dims
        down_blocks = ["DownBlock2D"] * len(dims)
        up_blocks = ["UpBlock2D"] * len(dims)
        self.model = ConvexUpUNet2DModel(
            sample_size=None,
            in_channels=3,
            out_channels=4,
            down_block_types=tuple(down_blocks),
            up_block_types=tuple(up_blocks),
            block_out_channels=dims,
            add_attention=use_attn,
        )

        # DDPM scheduler
        self.prediction_type = prediction_type
        self.scheduler = DDPMScheduler(
            num_train_timesteps=T,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=True,
            clip_sample_range=1,
            timestep_spacing="trailing",
        )

        self.flow_scaler = flow_scaler
        self.norm_in = norm_in
        self.padding = padding
        self.min_ds = 2 + len(dims)

    def prepare_latents(self, shape, **kwargs):
        noise = torch.randn(*shape, **kwargs) * self.scheduler.init_noise_sigma
        return noise

    def preprocess(self, x, eps=1e-8):
        if self.norm_in:
            b = x.shape[0]
            x_flat = x.view(b, -1)
            _mean, _std = torch.mean(x_flat, dim=-1), torch.std(x_flat, dim=-1) + eps
            while len(_mean.shape) < len(x.shape):
                _mean, _std = _mean.unsqueeze(-1), _std.unsqueeze(-1)
            return (x - _mean) / _std, (_mean, _std)
        return x * 2 - 1, None

    def postprocess(self, x, stats=None):
        if self.norm_in:
            _mean, _std = stats
            return (x * _std) + _mean
        return torch.clamp((x + 1) / 2, 0, 1)

    def normalize_flows(self, x):
        return x / self.flow_scaler

    def denormalize_flows(self, latent):
        return latent * self.flow_scaler

    def ensure_resolution_fit(self, x, pad_to_fit_unet=False):
        h, w = x.shape[-2:]
        ds = 2**self.min_ds
        pad_size = None
        if pad_to_fit_unet:
            pad_h = int(np.ceil(x.shape[-2] / ds) * ds) - x.shape[-2]
            pad_w = int(np.ceil(x.shape[-1] / ds) * ds) - x.shape[-1]
            if pad_h > 0 or pad_w > 0:
                pad_size = [pad_w // 2, pad_w - pad_w // 2, 0, pad_h]
                x = F.pad(x, pad_size, mode=self.padding)
        else:
            new_h = int(np.round(h / ds) * ds)
            new_w = int(np.round(w / ds) * ds)
            x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
        return x, pad_size

    def restore_orig_resolution(self, x, orig_hw, pad_size=None):
        if pad_size is not None:
            cur_h, cur_w = x.shape[-2:]
            x = x[..., pad_size[2] : cur_h - pad_size[3], pad_size[0] : cur_w - pad_size[1]]
        out_h, out_w = x.shape[-2:]
        orig_h, orig_w = orig_hw
        scale = torch.tensor([orig_w / out_w, orig_h / out_h], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1)
        scale = torch.cat([scale, scale], dim=1)
        x = F.interpolate(x, size=(orig_h, orig_w), mode="bicubic", align_corners=False) * scale
        return x

    def forward(self, x, num_inference_steps=8):
        """Generate interpolated frame via diffusion.

        Args:
            x: Input frames [B, 3, 2, H, W], normalized to [0, 1]
            num_inference_steps: Number of DDPM denoising steps

        Returns:
            Tuple of (interpolated_frame [B, 3, H, W], flows [B, 4, H, W])
        """
        orig_x = x
        x = rearrange(x, "b c f h w -> b (f c) h w")
        x, x_norm_stats = self.preprocess(x)
        b, _, h, w = x.shape
        orig_hw = (h, w)

        # Resolution alignment
        x, pad_size = self.ensure_resolution_fit(x)

        # DDPM denoising
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=x.device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(shape=(b, 4, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)

        for t in timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            _input = torch.cat([latent_model_input, x], dim=1)
            pred = self.model(_input, t).sample
            latents = self.scheduler.step(pred, t, latents).prev_sample

        # Denormalize flows
        flows = self.denormalize_flows(latents)
        flows = self.restore_orig_resolution(flows, orig_hw=orig_hw, pad_size=pad_size)

        # Synthesize final frame
        xt = self.synth_model(orig_x, flows)
        return xt, flows
