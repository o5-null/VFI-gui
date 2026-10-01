"""FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation.

Pure PyTorch implementation adapted from:
    https://github.com/tarun005/FLAVR
    https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

Reference:
    Kalluri, Tarun, et al. "FLAVR: Flow-Agnostic Video Representations for
    Fast Frame Interpolation." arXiv 2021.

Architecture:
    - UNet with 3D ResNet-18 encoder + 3D decoder
    - SEGating (Squeeze-and-Excitation) throughout
    - Batch mean normalisation
    - Supports arbitrary multipliers (2x, 4x, 8x via checkpoint selection)

Model takes 4 frames as input [I0, I1, I2, I3] and outputs n_outputs frames.
- 2x mode: outputs 1 intermediate frame between I1-I2
- 4x mode: outputs 3 intermediate frames
- 8x mode: outputs 7 intermediate frames
"""

from __future__ import annotations

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEGating(nn.Module):
    """Squeeze-and-Excitation gating for 3D."""
    def __init__(self, inplanes, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn_layer(self.pool(x))


def join_tensors(x1, x2, join_type="concat"):
    if join_type == "concat":
        return torch.cat([x1, x2], dim=1)
    elif join_type == "add":
        return x1 + x2
    return x1


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose", batchnorm=False):
        super().__init__()
        if upmode == "transpose":
            modules = [
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch),
            ]
        else:
            modules = [
                nn.Upsample(mode="trilinear", scale_factor=(1, 2, 2), align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1),
                SEGating(out_ch),
            ]
        if batchnorm:
            modules.append(nn.BatchNorm3d(out_ch))
        self.upconv = nn.Sequential(*modules)

    def forward(self, x):
        return self.upconv(x)


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):
        super().__init__()
        modules = [
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            SEGating(out_ch),
        ]
        if batchnorm:
            modules.append(nn.BatchNorm3d(out_ch))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class UpConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose", batchnorm=False):
        super().__init__()
        if upmode == "transpose":
            modules = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]
        else:
            modules = [
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            ]
        if batchnorm:
            modules.append(nn.BatchNorm2d(out_ch))
        self.upconv = nn.Sequential(*modules)

    def forward(self, x):
        return self.upconv(x)


class UNet3D3D(nn.Module):
    """FLAVR main UNet with 3D ResNet encoder + 3D decoder.

    Takes 4 input frames, produces n_outputs intermediate frames.
    """

    def __init__(self, block, n_inputs, n_outputs, batchnorm=False, join_type="concat", upmode="transpose"):
        super().__init__()
        nf = [512, 256, 128, 64]
        out_channels = 3 * n_outputs
        self.join_type = join_type
        self.n_outputs = n_outputs
        growth = 2 if join_type == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)

        # 3D ResNet encoder
        resnet_mod = importlib.import_module(".resnet_3d", "core.pytorch_models.vfi_torch.flavr")
        if n_outputs > 1:
            resnet_mod.useBias = True
        self.encoder = getattr(resnet_mod, block)(pretrained=False)

        # 3D decoder
        self.decoder = nn.Sequential(
            Conv3dBlock(nf[0], nf[1], kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            UpConv3D(nf[1] * growth, nf[2], kernel_size=(3, 4, 4), stride=(1, 2, 2),
                     padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            UpConv3D(nf[2] * growth, nf[3], kernel_size=(3, 4, 4), stride=(1, 2, 2),
                     padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            Conv3dBlock(nf[3] * growth, nf[3], kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            UpConv3D(nf[3] * growth, nf[3], kernel_size=(3, 4, 4), stride=(1, 2, 2),
                     padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
        )

        # 2D feature fusion
        self.feature_fuse = Conv2dBlock(
            nf[3] * n_inputs, nf[3], kernel_size=1, stride=1, batchnorm=batchnorm,
        )

        # Output convolution
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels, kernel_size=7, stride=1, padding=0),
        )

    def forward(self, images):
        """Forward pass.

        Args:
            images: List of 4 tensors [B, C, H, W] — [I0, I1, I2, I3]

        Returns:
            List of n_outputs tensors [B, C, H, W] — intermediate frames
        """
        # Stack along time dimension: [B, C, T=4, H, W]
        x = torch.stack(images, dim=2)

        # Batch mean normalisation
        mean_ = x.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        x = x - mean_

        # Encoder: 5-level feature pyramid
        x_0, x_1, x_2, x_3, x_4 = self.encoder(x)

        # Decoder with skip connections
        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = join_tensors(dx_3, x_3, join_type=self.join_type)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = join_tensors(dx_2, x_2, join_type=self.join_type)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = join_tensors(dx_1, x_1, join_type=self.join_type)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = join_tensors(dx_0, x_0, join_type=self.join_type)

        dx_out = self.lrelu(self.decoder[4](dx_0))

        # Unbind time dimension → concat channels
        dx_out = torch.cat(torch.unbind(dx_out, 2), 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)

        # Split back into individual frames
        out_frames = torch.split(out, dim=1, split_size_or_sections=3)
        mean_ = mean_.squeeze(2)  # [B, C, 1, 1]
        return [o + mean_ for o in out_frames]


class InputPadder:
    """Pads images such that dimensions are divisible by divisor."""
    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, input_tensor):
        return F.pad(input_tensor, self._pad, mode="replicate")

    def unpad(self, input_tensor):
        ht, wd = input_tensor.shape[-2:]
        return input_tensor[..., self._pad[2]:ht - self._pad[3], self._pad[0]:wd - self._pad[1]]
