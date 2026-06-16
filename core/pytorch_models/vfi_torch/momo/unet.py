"""
ConvexUpUNet — modified UNet2DModel from diffusers with RAFT-style convex upsampling.

Reference: diffusers UNet2DModel + MoMo modification.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Upsample2D, ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import get_down_block
from diffusers.utils import BaseOutput


@dataclass
class UNet2DOutput(BaseOutput):
    sample: torch.FloatTensor


class ConvexUpUNet2DModel(ModelMixin, ConfigMixin):
    """UNet2D with 8x downsampling inputs and RAFT-style convex upsampling output.

    Modified from diffusers UNet2DModel for MoMo VFI.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 4,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types: Tuple[str] = ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (256, 256, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. "
                f"down_block_types: {down_block_types}, up_block_types: {up_block_types}."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. "
                f"block_out_channels: {block_out_channels}, down_block_types: {down_block_types}."
            )

        # Input projection — 8x downsampling
        self.down_patch = nn.Sequential(nn.Conv2d(in_channels, int(block_out_channels[0] / 2), 8, 8), nn.SiLU())
        self.down_latent = nn.Sequential(nn.Conv2d(out_channels, block_out_channels[0], 8, 8), nn.SiLU())
        self.proj_inputs = nn.Conv2d(block_out_channels[0] * 2, block_out_channels[0], kernel_size=1)

        # Time embedding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16, log=False)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # Class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # First residual block (no downsampling)
        self.first_block = get_down_block(
            down_block_types[0],
            num_layers=layers_per_block,
            in_channels=block_out_channels[0],
            out_channels=block_out_channels[0],
            temb_channels=time_embed_dim,
            add_downsample=False,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            attention_head_dim=attention_head_dim,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

        # Main middle model
        self.mid_model = UNet2DModel(
            sample_size=None,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            down_block_types=down_block_types[1:],
            up_block_types=up_block_types[:-1],
            block_out_channels=block_out_channels[1:],
            add_attention=add_attention,
            class_embed_type=None,
        )

        # Convex upsampling output
        mask_w = 8 * 8 * 9 * 2
        self.out_up = UpMaskBlock2D(
            num_layers=layers_per_block + 1,
            in_channels=block_out_channels[0],
            out_channels=mask_w,
            prev_output_channel=out_channels,
            temb_channels=time_embed_dim,
            add_upsample=False,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    def convex_upsampling(self, flow, mask):
        """RAFT-style convex upsampling: 8x up."""
        b, _, h, w = flow.shape
        mask = mask.view(b, 2, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=3)
        up_flow = F.unfold(flow, kernel_size=3, padding=1).view(b, 2, 2, 9, 1, 1, h, w)
        up_flow = torch.sum(mask * up_flow, dim=3)
        up_flow = up_flow.permute(0, 1, 2, 5, 3, 6, 4).reshape(b, self.config.out_channels, h * 8, w * 8)
        up_flow = up_flow * 8
        return up_flow

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        # 1. Time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            emb = emb + self.class_embedding(class_labels).to(dtype=self.dtype)

        # 2. Pre-process: split input into latents, x0, x1
        latents, x0, x1 = sample.split([self.config.out_channels, self.config.in_channels, self.config.in_channels], dim=1)

        # 3. 8x downsampling
        dx0, dx1 = self.down_patch(torch.cat([x0, x1], dim=0)).chunk(2, dim=0)
        dl = self.down_latent(latents)
        sample = self.proj_inputs(torch.cat([dx0, dx1, dl], dim=1))
        down_block_res_samples = (sample,)

        # 4. First residual block
        sample, res_samples = self.first_block(hidden_states=sample, temb=emb)
        down_block_res_samples += res_samples

        # 5. Mid model
        sample = self.mid_model(sample, timesteps).sample

        # 6. 8x convex upsampling
        res_samples = down_block_res_samples[-len(self.out_up.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(self.out_up.resnets)]
        up_mask = self.out_up(sample, res_samples, emb)
        sample = self.convex_upsampling(sample, up_mask)

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)
        return UNet2DOutput(sample=sample)


class UpMaskBlock2D(nn.Module):
    """Up block that produces convex upsampling mask weights."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = False,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        resnets = []
        self.proj_in = nn.Identity()
        hidden_dim = int(math.ceil((prev_output_channel + in_channels) / resnet_groups) * resnet_groups)
        if hidden_dim != prev_output_channel + in_channels:
            self.proj_in = nn.Conv2d(prev_output_channel + in_channels, hidden_dim, 3, 1, 1)
        for i in range(num_layers):
            resnet_in_channels = hidden_dim if i == 0 else in_channels * 2
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(in_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
            self.proj_out = nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for i, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if i == 0:
                hidden_states = self.proj_in(hidden_states)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        else:
            hidden_states = self.proj_out(hidden_states)
        return hidden_states
