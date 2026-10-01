"""
ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation.

Pure PyTorch implementation adapted from:
    https://github.com/danielism97/ST-MFNet
    https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

Reference:
    Danier, Duolikun, et al. "ST-MFNet: A Spatio-Temporal Multi-Flow Network
    for Frame Interpolation." CVPR 2022.

Architecture overview:
    - Takes 4 input frames (I0, I1, I2, I3), outputs middle frame between I1-I2
    - UMultiScaleResNext for spatio-temporal feature extraction
    - AdaCoF kernel estimation at 3 scales (1/2×, 1×, 2×)
    - PWCNet optical flow estimation for softmax splatting refinement
    - MIMOGridNet multi-scale synthesis
    - UNet3d_18 for dynamic texture generation
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.costvol import costvol_pwc
from ..ops.softsplat import softsplat_func


# ============================================================================
# Pure PyTorch AdaCoF forward warping
# ============================================================================

def adacof_warp(
    tenInput: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Adaptive Collaboration of Flows warping (pure PyTorch).

    For each output pixel, samples from K×K positions in the source image
    using estimated offsets and weights, then combines them.

    Args:
        tenInput: Input image [B, C, H, W]
        weight: Kernel weights [B, K*K, H, W]
        alpha: X offsets (normalised) [B, K*K, H, W]
        beta: Y offsets (normalised) [B, K*K, H, W]
        dilation: Kernel dilation factor

    Returns:
        Warped image [B, C, H, W]
    """
    B, C, H, W = tenInput.shape
    K = int(round(math.sqrt(weight.shape[1])))
    device = tenInput.device

    # Normalised pixel grid in [-1, 1]
    y = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1)
    x = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W)

    output = torch.zeros_like(tenInput)

    for ki in range(K):
        for kj in range(K):
            idx = ki * K + kj
            w = weight[:, idx:idx + 1, :, :]       # [B, 1, H, W]
            a = alpha[:, idx:idx + 1, :, :]        # [B, 1, H, W]
            b = beta[:, idx:idx + 1, :, :]         # [B, 1, H, W]

            # Convert pixel-space offsets to normalised [-1, 1] coordinates
            sx = x + a * (2.0 / W) * dilation
            sy = y + b * (2.0 / H) * dilation
            grid = torch.stack([sx, sy], dim=-1).squeeze(1)  # [B, H, W, 2]

            sampled = F.grid_sample(
                tenInput, grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )
            output = output + sampled * w

    return output


# ============================================================================
# PWC-Net optical flow estimator
# ============================================================================

class PWCNet(nn.Module):
    """PWC-Net for optical flow estimation (using pure PyTorch correlation)."""

    class Extractor(nn.Module):
        """Feature pyramid extractor (6 levels)."""
        def __init__(self):
            super().__init__()
            self.netOne = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(16, 16, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(16, 16, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netTwo = nn.Sequential(
                nn.Conv2d(16, 32, 3, 2, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netThr = nn.Sequential(
                nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netFou = nn.Sequential(
                nn.Conv2d(64, 96, 3, 2, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(96, 96, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(96, 96, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netFiv = nn.Sequential(
                nn.Conv2d(96, 128, 3, 2, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netSix = nn.Sequential(
                nn.Conv2d(128, 196, 3, 2, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(196, 196, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(196, 196, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )

        def forward(self, x):
            f1 = self.netOne(x)
            f2 = self.netTwo(f1)
            f3 = self.netThr(f2)
            f4 = self.netFou(f3)
            f5 = self.netFiv(f4)
            f6 = self.netSix(f5)
            return [f1, f2, f3, f4, f5, f6]

    class Decoder(nn.Module):
        """Single-scale flow decoder with correlation volume."""
        def __init__(self, intLevel: int):
            super().__init__()
            intPrev = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                       81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
            intCurr = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                       81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

            if intLevel < 6:
                self.netUpflow = nn.ConvTranspose2d(2, 2, 4, 2, 1)
            if intLevel < 6:
                self.netUpfeat = nn.ConvTranspose2d(
                    intPrev + 128 + 128 + 96 + 64 + 32, 2, 4, 2, 1,
                )
            if intLevel < 6:
                self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

            self.netOne = nn.Sequential(
                nn.Conv2d(intCurr, 128, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netTwo = nn.Sequential(
                nn.Conv2d(intCurr + 128, 128, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netThr = nn.Sequential(
                nn.Conv2d(intCurr + 128 + 128, 96, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netFou = nn.Sequential(
                nn.Conv2d(intCurr + 128 + 128 + 96, 64, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netFiv = nn.Sequential(
                nn.Conv2d(intCurr + 128 + 128 + 96 + 64, 32, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
            )
            self.netSix = nn.Sequential(
                nn.Conv2d(intCurr + 128 + 128 + 96 + 64 + 32, 2, 3, 1, 1),
            )

        def forward(self, tenFirst, tenSecond, objPrevious):
            if objPrevious is None:
                # Pure correlation volume at coarsest level
                vol = F.leaky_relu(costvol_pwc(tenFirst, tenSecond), 0.1, inplace=False)
                tenFeat = vol
            else:
                tenFlow = self.netUpflow(objPrevious["tenFlow"])
                tenFeat = self.netUpfeat(objPrevious["tenFeat"])
                # Warp second feature with upsampled flow
                warped = warp_flow(tenSecond, tenFlow * self.fltBackwarp)
                vol = F.leaky_relu(costvol_pwc(tenFirst, warped), 0.1, inplace=False)
                tenFeat = torch.cat([vol, tenFirst, tenFlow, tenFeat], 1)

            tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
            tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
            tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
            tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
            tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)
            tenFlow = self.netSix(tenFeat)
            return {"tenFlow": tenFlow, "tenFeat": tenFeat}

    class Refiner(nn.Module):
        """Contextual flow refinement with dilated convolutions."""
        def __init__(self):
            super().__init__()
            self.netMain = nn.Sequential(
                nn.Conv2d(81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 128, 3, 1, 1, dilation=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(128, 128, 3, 1, 2, dilation=2),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(128, 128, 3, 1, 4, dilation=4),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(128, 96, 3, 1, 8, dilation=8),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(96, 64, 3, 1, 16, dilation=16),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(64, 32, 3, 1, 1, dilation=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(32, 2, 3, 1, 1, dilation=1),
            )

        def forward(self, x):
            return self.netMain(x)

    def __init__(self):
        super().__init__()
        self.netExtractor = self.Extractor()
        self.netTwo = self.Decoder(2)
        self.netThr = self.Decoder(3)
        self.netFou = self.Decoder(4)
        self.netFiv = self.Decoder(5)
        self.netSix = self.Decoder(6)
        self.netRefiner = self.Refiner()

    def forward(self, tenFirst, tenSecond, *args):
        # Optionally pass pre-extracted feature pyramid
        if len(args) == 0:
            tenFirst = self.netExtractor(tenFirst)
            tenSecond = self.netExtractor(tenSecond)
        else:
            tenFirst, tenSecond = args

        obj = self.netSix(tenFirst[-1], tenSecond[-1], None)
        obj = self.netFiv(tenFirst[-2], tenSecond[-2], obj)
        obj = self.netFou(tenFirst[-3], tenSecond[-3], obj)
        obj = self.netThr(tenFirst[-4], tenSecond[-4], obj)
        obj = self.netTwo(tenFirst[-5], tenSecond[-5], obj)

        return obj["tenFlow"] + self.netRefiner(obj["tenFeat"])

    def extract_pyramid(self, tenFirst, tenSecond):
        return self.netExtractor(tenFirst), self.netExtractor(tenSecond)

    def extract_pyramid_single(self, tenFirst):
        return self.netExtractor(tenFirst)


# ============================================================================
# Flow warping helper
# ============================================================================

_backwarp_grid: dict = {}

def warp_flow(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Backward warp using flow field with cached coordinate grid."""
    device = tenFlow.device
    key = (str(device), str(tenFlow.shape))

    if key not in _backwarp_grid:
        hor = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
        hor = hor.view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        ver = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
        ver = ver.view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        _backwarp_grid[key] = torch.cat([hor, ver], 1)

    flow = torch.cat([
        tenFlow[:, 0:1] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2] / ((tenInput.shape[2] - 1.0) / 2.0),
    ], 1)

    # Also pass alpha channel for occlusion handling
    ones = tenFlow.new_ones([tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])
    inp = torch.cat([tenInput, ones], 1)
    g = (_backwarp_grid[key] + flow).permute(0, 2, 3, 1)

    if g.dtype != inp.dtype:
        g = g.to(inp.dtype)

    out = F.grid_sample(inp, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = out[:, -1:, :, :]
    mask = (mask > 0.999).float()
    return out[:, :-1, :, :] * mask


# ============================================================================
# 8-tap Upsampler
# ============================================================================

class Upsampler8Tap(nn.Module):
    """8-tap separable upsampling (2×)."""
    def __init__(self):
        super().__init__()
        filt = torch.tensor([[-1, 4, -11, 40, 40, -11, 4, -1]]).div(64)
        self.filter = nn.Parameter(filt.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, im):
        B, C, H, W = im.shape
        im_up = torch.zeros(B, C, H * 2, W * 2, device=im.device, dtype=im.dtype)
        im_up[:, :, ::2, ::2] = im

        p = (8 - 1) // 2
        row = F.conv2d(F.pad(im, (p, p + 1, 0, 0), mode="reflect"), self.filter, groups=3)
        im_up[:, :, 0::2, 1::2] = row
        col = torch.transpose(
            F.conv2d(F.pad(torch.transpose(im, 2, 3), (p, p + 1, 0, 0), mode="reflect"),
                     self.filter, groups=3),
            2, 3,
        )
        im_up[:, :, 1::2, 0::2] = col
        cross = F.conv2d(F.pad(im_up[:, :, 1::2, ::2], (p, p + 1, 0, 0), mode="reflect"),
                         self.filter, groups=3)
        im_up[:, :, 1::2, 1::2] = cross
        return im_up


# ============================================================================
# Gaussian kernel
# ============================================================================

def gaussian_kernel(sz: int, sigma: float) -> torch.Tensor:
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2)
    k = torch.exp(-1.0 / (2 * sigma ** 2) * k ** 2)
    k = k.reshape(-1, 1) * k.reshape(1, -1)
    return k / torch.sum(k)


def module_normalize(frame: torch.Tensor) -> torch.Tensor:
    """Channel-wise mean normalisation (ImageNet-ish stats)."""
    mean = torch.tensor([0.4631, 0.4352, 0.3990], device=frame.device)
    return frame - mean.view(1, -1, 1, 1)


# ============================================================================
# SEGating (Squeeze-and-Excitation for 3D)
# ============================================================================

class SEGating(nn.Module):
    """Squeeze-and-Excitation gating for 3D convolutions."""
    def __init__(self, inplanes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn_layer(self.pool(x))


class Identity(nn.Module):
    def forward(self, x):
        return x


# ============================================================================
# 3D convolution helpers
# ============================================================================

class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super().__init__(in_planes, out_planes, kernel_size=(3, 3, 3),
                         stride=stride, padding=padding, bias=False)

    @staticmethod
    def get_downsample_stride(stride, temporal_stride):
        return (temporal_stride, stride, stride) if temporal_stride else (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        super().__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding), bias=False),
            nn.BatchNorm3d(midplanes), nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super().__init__(in_planes, out_planes, kernel_size=(1, 3, 3),
                         stride=(1, stride, stride), padding=(0, padding, padding), bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


# ============================================================================
# 3D ResNet building blocks
# ============================================================================

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes),
        )
        self.fg = SEGating(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class BasicStem(nn.Sequential):
    def __init__(self, outplanes=32):
        super().__init__(
            nn.Conv3d(3, outplanes, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(outplanes),
            nn.ReLU(inplace=True),
        )


class VideoResNet(nn.Module):
    """Generic 3D ResNet for video."""
    def __init__(self, block, conv_makers, layers, stem,
                 zero_init_residual=False, channels=(32, 64, 96, 128)):
        super().__init__()
        self.inplanes = channels[0]
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], channels[1], layers[1], stride=2, temporal_stride=1)
        self.layer3 = self._make_layer(block, conv_makers[2], channels[2], layers[2], stride=2, temporal_stride=1)
        self.layer4 = self._make_layer(block, conv_makers[3], channels[3], layers[3], stride=1, temporal_stride=1)
        self._initialize_weights()

    def forward(self, x):
        c0 = self.stem(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c0, c1, c2, c3, c4

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, temporal_stride=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride, temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
            stride = ds_stride

        layers_ = [block(self.inplanes, planes, conv_builder, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_.append(block(self.inplanes, planes, conv_builder))
        return nn.Sequential(*layers_)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def r3d_18(bn: bool = True, channels=(32, 64, 96, 128)) -> VideoResNet:
    """Construct R3D-18 encoder for STMFNet's UNet3d."""
    global _batchnorm_3d
    _batchnorm_3d = nn.BatchNorm3d if bn else Identity

    return VideoResNet(
        block=BasicBlock3D,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        channels=channels,
    )


_batchnorm_3d = nn.BatchNorm3d


# ============================================================================
# 3D UNet for dynamic texture
# ============================================================================

class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
            SEGating(out_ch),
            _batchnorm_3d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose"):
        super().__init__()
        if upmode == "transpose":
            modules = [
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch),
                _batchnorm_3d(out_ch),
            ]
        else:
            modules = [
                nn.Upsample(mode="trilinear", scale_factor=(1, 2, 2), align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1),
                SEGating(out_ch),
                _batchnorm_3d(out_ch),
            ]
        self.upconv = nn.Sequential(*modules)

    def forward(self, x):
        return self.upconv(x)


class UNet3d18(nn.Module):
    """3D UNet with R3D-18 encoder for dynamic texture generation.

    Takes 5 frames stacked along time dim: I0, I1, output_tilde, I2, I3.
    """
    def __init__(self, channels=(32, 64, 96, 128), bn: bool = True):
        super().__init__()
        growth = 2
        upmode = "transpose"
        rev = channels[::-1]  # [128, 96, 64, 32]

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.encoder = r3d_18(bn=bn, channels=channels)

        self.decoder = nn.Sequential(
            Conv3dBlock(rev[0], rev[1], kernel_size=3, padding=1, bias=True),
            UpConv3D(rev[1] * growth, rev[2], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), upmode=upmode),
            UpConv3D(rev[2] * growth, rev[3], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), upmode=upmode),
            Conv3dBlock(rev[3] * growth, rev[3], kernel_size=3, padding=1, bias=True),
            UpConv3D(rev[3] * growth, rev[3], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), upmode=upmode),
        )

        self.feature_fuse = nn.Sequential(
            nn.Conv2d(rev[3] * 5, rev[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(rev[3]) if bn else Identity(),
        )

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(rev[3], 3, kernel_size=7, stride=1, padding=0),
        )

    def forward(self, im1, im3, im5, im7, im4_tilde):
        """Args: I1, I3, I5, I7, output_tilde — each [B, C, H, W]."""
        images = torch.stack((im1, im3, im4_tilde, im5, im7), dim=2)

        x_0, x_1, x_2, x_3, x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = torch.cat([dx_3, x_3], dim=1)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = torch.cat([dx_2, x_2], dim=1)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = torch.cat([dx_1, x_1], dim=1)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = torch.cat([dx_0, x_0], dim=1)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out, 2), 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)
        return out


# ============================================================================
# SE Block (2D)
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super().__init__()
        mid = input_dim // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


# ============================================================================
# ResNeXt-style blocks (encoder / decoder)
# ============================================================================

class ResNextBlock(nn.Module):
    """ResNeXt-style block for 2D, supports down/up-sampling via stride/transpose."""
    def __init__(self, down: bool, cin, cout, ks, stride=1, groups=32, base_width=4, norm_layer=None):
        super().__init__()
        if norm_layer is None or norm_layer == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_layer == "identity":
            norm_layer = Identity

        width = int(cout * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(cin, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)

        if down:
            self.conv2 = nn.Conv2d(width, width, kernel_size=ks, stride=stride,
                                   padding=(ks - 1) // 2, groups=groups, bias=False)
        else:
            self.conv2 = nn.ConvTranspose2d(width, width, kernel_size=ks, stride=stride,
                                            padding=(ks - stride) // 2, groups=groups, bias=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2d(width, cout, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(cout)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or cin != cout:
            if down:
                self.downsample = nn.Sequential(
                    nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                    norm_layer(cout),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=stride, bias=False),
                    norm_layer(cout),
                )
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class MultiScaleResNextBlock(nn.Module):
    """Multi-scale ResNeXt block with SE attention."""
    def __init__(self, down, cin, cout, ks_s, ks_l, stride, norm_layer):
        super().__init__()
        self.resnext_small = ResNextBlock(down, cin, cout // 2, ks_s, stride, norm_layer=norm_layer)
        self.resnext_large = ResNextBlock(down, cin, cout // 2, ks_l, stride, norm_layer=norm_layer)
        self.attention = SEBlock(cout)

    def forward(self, x):
        out = torch.cat([self.resnext_small(x), self.resnext_large(x)], 1)
        return self.attention(out)


class UMultiScaleResNext(nn.Module):
    """U-shaped multi-scale ResNeXt encoder for spatio-temporal feature extraction.

    Takes two frames (I1, I2) concatenated channel-wise.
    """
    def __init__(self, channels=(64, 128, 256, 512), norm_layer="batch", inplanes=6):
        super().__init__()
        self.conv1 = MultiScaleResNextBlock(True, inplanes, channels[0], ks_s=3, ks_l=7, stride=2, norm_layer=norm_layer)
        self.conv2 = MultiScaleResNextBlock(True, channels[0], channels[1], ks_s=3, ks_l=7, stride=2, norm_layer=norm_layer)
        self.conv3 = MultiScaleResNextBlock(True, channels[1], channels[2], ks_s=3, ks_l=5, stride=2, norm_layer=norm_layer)
        self.conv4 = MultiScaleResNextBlock(True, channels[2], channels[3], ks_s=3, ks_l=5, stride=2, norm_layer=norm_layer)

        self.deconv4 = MultiScaleResNextBlock(True, channels[3], channels[3], ks_s=3, ks_l=5, stride=1, norm_layer=norm_layer)
        self.deconv3 = MultiScaleResNextBlock(False, channels[3], channels[2], ks_s=4, ks_l=6, stride=2, norm_layer=norm_layer)
        self.deconv2 = MultiScaleResNextBlock(False, channels[2], channels[1], ks_s=4, ks_l=8, stride=2, norm_layer=norm_layer)
        self.deconv1 = MultiScaleResNextBlock(False, channels[1], channels[0], ks_s=4, ks_l=8, stride=2, norm_layer=norm_layer)

    def forward(self, im0, im2):
        joint = torch.cat([im0, im2], 1)  # [B, 6, H, W]

        c1 = self.conv1(joint)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        d4 = self.deconv4(c4)
        d3 = self.deconv3(d4 + c4)
        d2 = self.deconv2(d3 + c3)
        d1 = self.deconv1(d2 + c2)
        return d1


# ============================================================================
# GridNet variants for multi-scale synthesis
# ============================================================================

class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(), nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.PReLU(), nn.Conv2d(ch_out, ch_out, 3, 1, 1),
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x


class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(), nn.Conv2d(ch_in, ch_out, 3, 2, 1),
            nn.PReLU(), nn.Conv2d(ch_out, ch_out, 3, 1, 1),
        )

    def forward(self, x):
        return self.f(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.PReLU(), nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.PReLU(), nn.Conv2d(ch_out, ch_out, 3, 1, 1),
        )

    def forward(self, x):
        return self.f(x)


class MIMOGridNet(nn.Module):
    """Multi-Input Multi-Output GridNet for multi-scale synthesis.

    Supports multiple output rows.
    """
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96), n_row=3, n_col=6, outrow=(0, 1, 2)):
        super().__init__()
        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = grid_chs
        self.outrow = outrow

        for r, n_ch in enumerate(self.n_chs):
            setattr(self, f"lateral_{r}_0", LateralBlock(in_chs[r], n_ch))
            for c in range(1, self.n_col):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(n_ch, n_ch))

        for r in range(self.n_row - 1):
            for c in range(self.n_col // 2):
                setattr(self, f"down_{r}_{c}", DownSamplingBlock(self.n_chs[r], self.n_chs[r + 1]))

        for r in range(self.n_row - 1):
            for c in range(self.n_col // 2):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(self.n_chs[r + 1], self.n_chs[r]))

        for i, r in enumerate(outrow):
            setattr(self, f"lateral_final_{r}", LateralBlock(self.n_chs[r], out_chs[i]))

    def forward(self, *args):
        cur_col = list(args)
        # Down-sampling phase
        for c in range(self.n_col // 2):
            for r in range(self.n_row):
                cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                if r != 0:
                    cur_col[r] += getattr(self, f"down_{r-1}_{c}")(cur_col[r - 1])
        # Up-sampling phase
        for c in range(self.n_col // 2, self.n_col):
            for r in range(self.n_row - 1, -1, -1):
                cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                if r != self.n_row - 1:
                    cur_col[r] += getattr(self, f"up_{r}_{c - self.n_col // 2}")(cur_col[r + 1])

        return [getattr(self, f"lateral_final_{r}")(cur_col[r]) for r in self.outrow]


# ============================================================================
# Kernel estimation (AdaCoF kernel prediction)
# ============================================================================

class KernelEstimation(nn.Module):
    """Estimates AdaCoF kernels (weights + offsets) at 3 scales from features."""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        ks2 = kernel_size ** 2

        def subnet_weight():
            return nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, ks2, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(ks2, ks2, 3, 1, 1), nn.Softmax(dim=1),
            )

        def subnet_weight_ds():
            return nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, ks2, 3, 1, 1), nn.Softmax(dim=1),
            )

        def subnet_weight_us():
            return nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, ks2, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                nn.Conv2d(ks2, ks2, 3, 1, 1), nn.Softmax(dim=1),
            )

        def subnet_offset(ks):
            return nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, ks, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(ks, ks, 3, 1, 1),
            )

        def subnet_offset_ds(ks):
            return nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, ks, 3, 1, 1),
            )

        def subnet_offset_us(ks):
            return nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Conv2d(64, ks, 3, 1, 1), nn.ReLU(inplace=False),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                nn.Conv2d(ks, ks, 3, 1, 1),
            )

        self.moduleWeight1_ds = subnet_weight_ds()
        self.moduleAlpha1_ds = subnet_offset_ds(ks2)
        self.moduleBeta1_ds = subnet_offset_ds(ks2)
        self.moduleWeight2_ds = subnet_weight_ds()
        self.moduleAlpha2_ds = subnet_offset_ds(ks2)
        self.moduleBeta2_ds = subnet_offset_ds(ks2)

        self.moduleWeight1 = subnet_weight()
        self.moduleAlpha1 = subnet_offset(ks2)
        self.moduleBeta1 = subnet_offset(ks2)
        self.moduleWeight2 = subnet_weight()
        self.moduleAlpha2 = subnet_offset(ks2)
        self.moduleBeta2 = subnet_offset(ks2)

        self.moduleWeight1_us = subnet_weight_us()
        self.moduleAlpha1_us = subnet_offset_us(ks2)
        self.moduleBeta1_us = subnet_offset_us(ks2)
        self.moduleWeight2_us = subnet_weight_us()
        self.moduleAlpha2_us = subnet_offset_us(ks2)
        self.moduleBeta2_us = subnet_offset_us(ks2)

    def forward(self, feats):
        return (
            self.moduleWeight1_ds(feats), self.moduleAlpha1_ds(feats), self.moduleBeta1_ds(feats),
            self.moduleWeight2_ds(feats), self.moduleAlpha2_ds(feats), self.moduleBeta2_ds(feats),
            self.moduleWeight1(feats), self.moduleAlpha1(feats), self.moduleBeta1(feats),
            self.moduleWeight2(feats), self.moduleAlpha2(feats), self.moduleBeta2(feats),
            self.moduleWeight1_us(feats), self.moduleAlpha1_us(feats), self.moduleBeta1_us(feats),
            self.moduleWeight2_us(feats), self.moduleAlpha2_us(feats), self.moduleBeta2_us(feats),
        )


# ============================================================================
# Main STMFNet Model
# ============================================================================

class STMFNet_Model(nn.Module):
    """ST-MFNet video frame interpolation model.

    Takes 4 frames (I0, I1, I2, I3) and outputs the interpolated frame
    between I1 and I2 (timestep = 0.5).
    """

    def __init__(self):
        super().__init__()

        class Metric(nn.Module):
            """Learned per-pixel metric for softsplat confidence."""
            def __init__(self):
                super().__init__()
                self.paramScale = nn.Parameter(-torch.ones(1, 1, 1, 1))

            def forward(self, tenFirst, tenSecond, tenFlow):
                return self.paramScale * F.l1_loss(
                    input=tenFirst,
                    target=warp_flow(tenSecond, tenFlow),
                    reduction="none",
                ).mean(1, True)

        self.kernel_size = 5
        self.dilation = 1
        self.featc = (64, 128, 256, 512)
        self.featnorm = "batch"
        self.finetune_pwc = False

        self.kernel_pad = ((self.kernel_size - 1) * self.dilation) // 2

        # Spatio-temporal feature extraction (from I1, I2)
        self.feature_extractor = UMultiScaleResNext(self.featc, norm_layer=self.featnorm)

        # AdaCoF kernel estimation (3 scales from features)
        self.get_kernel = KernelEstimation(self.kernel_size)

        # Padding for AdaCoF warping
        self.modulePad = nn.ReplicationPad2d(
            [self.kernel_pad] * 4
        )

        # Gaussian blur kernel for down-sampling
        self.gauss_kernel = nn.Parameter(
            gaussian_kernel(5, 0.5).repeat(3, 1, 1, 1), requires_grad=False
        )

        # 2× upsampler
        self.upsampler = Upsampler8Tap()

        # Multi-scale synthesis (GridNet with 3 rows, 4 cols)
        self.scale_synthesis = MIMOGridNet(
            (6, 6 + 6, 6), (3,), grid_chs=(32, 64, 96), n_row=3, n_col=4, outrow=(1,),
        )

        # PWCNet optical flow
        self.flow_estimator = PWCNet()

        # Softmax splatting (from shared ops)
        # softsplat_func is used directly in forward()

        self.metric = Metric()

        # Dynamic texture 3D UNet
        self.dyntex_generator = UNet3d18(bn=(self.featnorm == "batch"))

        # Freeze PWCNet unless finetuning
        if not self.finetune_pwc:
            for p in self.flow_estimator.parameters():
                p.requires_grad = False

    def forward(self, I0, I1, I2, I3):
        """Forward pass: I0..I3 are [B, C, H, W] tensors, returns [B, C, H, W]."""
        B, _, H, W = I1.shape

        # Check frame size consistency
        assert I2.shape[2:] == (H, W), "Frame sizes do not match"

        # Pad to multiples of 128
        h_pad, w_pad = False, False
        if H % 128 != 0:
            pad_h = 128 - H % 128
            I0 = F.pad(I0, (0, 0, 0, pad_h), mode="reflect")
            I1 = F.pad(I1, (0, 0, 0, pad_h), mode="reflect")
            I2 = F.pad(I2, (0, 0, 0, pad_h), mode="reflect")
            I3 = F.pad(I3, (0, 0, 0, pad_h), mode="reflect")
            h_pad = True
        if W % 128 != 0:
            pad_w = 128 - W % 128
            I0 = F.pad(I0, (0, pad_w, 0, 0), mode="reflect")
            I1 = F.pad(I1, (0, pad_w, 0, 0), mode="reflect")
            I2 = F.pad(I2, (0, pad_w, 0, 0), mode="reflect")
            I3 = F.pad(I3, (0, pad_w, 0, 0), mode="reflect")
            w_pad = True

        _, _, Hp, Wp = I1.shape

        # ---- Step 1: Feature extraction ----
        feats = self.feature_extractor(module_normalize(I1), module_normalize(I2))

        # ---- Step 2: AdaCoF kernel estimation ----
        kernel_out = self.get_kernel(feats)
        (moduleWeight1_ds, moduleAlpha1_ds, moduleBeta1_ds, moduleWeight2_ds, moduleAlpha2_ds, moduleBeta2_ds,
         moduleWeight1, moduleAlpha1, moduleBeta1, moduleWeight2, moduleAlpha2, moduleBeta2,
         moduleWeight1_us, moduleAlpha1_us, moduleBeta1_us, moduleWeight2_us, moduleAlpha2_us, moduleBeta2_us) = kernel_out

        # ---- Step 3: Multi-scale AdaCoF warping ----
        # Note: No explicit padding needed — grid_sample(padding_mode="border") handles boundaries

        # Original scale
        warp1 = adacof_warp(I1, moduleWeight1, moduleAlpha1, moduleBeta1, self.dilation)
        warp2 = adacof_warp(I2, moduleWeight2, moduleAlpha2, moduleBeta2, self.dilation)

        # 1/2 down-sampled
        p_g = (self.gauss_kernel.shape[-1] - 1) // 2
        I1_blur = F.conv2d(F.pad(I1, [p_g] * 4, mode="reflect"), self.gauss_kernel, groups=3)
        I2_blur = F.conv2d(F.pad(I2, [p_g] * 4, mode="reflect"), self.gauss_kernel, groups=3)
        I1_ds = F.interpolate(I1_blur, size=(Hp // 2, Wp // 2), mode="bilinear", align_corners=False)
        I2_ds = F.interpolate(I2_blur, size=(Hp // 2, Wp // 2), mode="bilinear", align_corners=False)
        warp1_ds = adacof_warp(I1_ds, moduleWeight1_ds, moduleAlpha1_ds, moduleBeta1_ds, self.dilation)
        warp2_ds = adacof_warp(I2_ds, moduleWeight2_ds, moduleAlpha2_ds, moduleBeta2_ds, self.dilation)

        # 2× up-sampled
        I1_us = self.upsampler(I1)
        I2_us = self.upsampler(I2)
        warp1_us = adacof_warp(I1_us, moduleWeight1_us, moduleAlpha1_us, moduleBeta1_us, self.dilation)
        warp2_us = adacof_warp(I2_us, moduleWeight2_us, moduleAlpha2_us, moduleBeta2_us, self.dilation)

        # ---- Step 4: Softsplat refinement via optical flow ----
        pyramid0, pyramid2 = self.flow_estimator.extract_pyramid(I1, I2)
        flow_0_2 = 20.0 * self.flow_estimator(I1, I2, pyramid0, pyramid2)
        flow_0_2 = F.interpolate(flow_0_2, size=(Hp, Wp), mode="bilinear", align_corners=False)
        flow_2_0 = 20.0 * self.flow_estimator(I2, I1, pyramid2, pyramid0)
        flow_2_0 = F.interpolate(flow_2_0, size=(Hp, Wp), mode="bilinear", align_corners=False)

        met_0_2 = self.metric(I1, I2, flow_0_2)
        met_2_0 = self.metric(I2, I1, flow_2_0)

        soft0 = softsplat_func(I1, 0.5 * flow_0_2, met_0_2)
        soft2 = softsplat_func(I2, 0.5 * flow_2_0, met_2_0)

        # ---- Step 5: Multi-scale synthesis ----
        comb_us = torch.cat([warp1_us, warp2_us], dim=1)
        comb = torch.cat([warp1, warp2, soft0[..., :Hp, :Wp], soft2[..., :Hp, :Wp]], dim=1)
        comb_ds = torch.cat([warp1_ds, warp2_ds], dim=1)

        # Pad if softsplat changed dimensions slightly
        _, _, Hc, Wc = comb.shape
        if Hc != Hp or Wc != Wp:
            comb = F.interpolate(comb, size=(Hp, Wp), mode="bilinear", align_corners=False)

        output_tilde = self.scale_synthesis(comb_us, comb, comb_ds)[0]

        # ---- Step 6: Dynamic texture ----
        dyntex = self.dyntex_generator(I0, I1, I2, I3, output_tilde)
        output = output_tilde + dyntex

        # Remove padding
        if h_pad:
            output = output[:, :, :H, :]
        if w_pad:
            output = output[:, :, :, :W]

        return output
