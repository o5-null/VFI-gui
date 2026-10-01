"""
M2M — Many-to-Many Splatting for Efficient Video Frame Interpolation.

Pure PyTorch port of the original M2M VFI architecture.

Reference:
    "Many-to-many Splatting for Efficient Video Frame Interpolation"
    Hu et al., CVPR 2022
    https://github.com/feinanshan/M2M_VFI

This implementation uses pure-PyTorch substitutes for the CuPy/Taichi ops:
  - softsplat_func: Scatter-add based forward warping
  - costvol_func: Unfold-based correlation cost volume
"""

from __future__ import annotations

import collections
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import softsplat_func, costvol_func


# ============================================================================
# Utility: coordinate grid cache for backward warping
# ============================================================================

_obj_backwarp_cache: dict = {}


def backwarp(tenIn: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Backward warp a feature map using grid_sample with cached grid.

    Args:
        tenIn: Input features [B, C, H, W]
        tenFlow: Optical flow [B, 2, H, W]

    Returns:
        Warped features [B, C, H, W]
    """
    dtype = str(tenFlow.dtype)
    device = str(tenFlow.device)
    H, W = tenFlow.shape[2], tenFlow.shape[3]
    cache_key = f"grid_{dtype}_{device}_{H}_{W}"

    if cache_key not in _obj_backwarp_cache:
        tenHor = (
            torch.linspace(start=-1.0, end=1.0, steps=W, dtype=tenFlow.dtype, device=tenFlow.device)
            .view(1, 1, 1, -1)
            .repeat(1, 1, H, 1)
        )
        tenVer = (
            torch.linspace(start=-1.0, end=1.0, steps=H, dtype=tenFlow.dtype, device=tenFlow.device)
            .view(1, 1, -1, 1)
            .repeat(1, 1, 1, W)
        )
        _obj_backwarp_cache[cache_key] = torch.cat([tenHor, tenVer], dim=1)

    # Normalize flow to [-1, 1] grid
    if W != H:
        flow_norm = tenFlow * torch.tensor(
            data=[2.0 / (W - 1.0), 2.0 / (H - 1.0)],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        ).view(1, 2, 1, 1)
    else:
        flow_norm = tenFlow * (2.0 / (W - 1.0))

    return F.grid_sample(
        input=tenIn,
        grid=(_obj_backwarp_cache[cache_key] + flow_norm).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


# ============================================================================
# Basic building block — configurable conv/activation from string description
# ============================================================================


class Evenize(nn.Module):
    """Pad to even dimensions."""

    def __init__(self, strPad: str = "zeros"):
        super().__init__()
        self.strPad = strPad

    def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
        intPad = [0, 0, 0, 0]
        if tenIn.shape[3] % 2 != 0:
            intPad[1] = 1
        if tenIn.shape[2] % 2 != 0:
            intPad[3] = 1
        if min(intPad) != 0 or max(intPad) != 0:
            mode = "constant" if self.strPad == "zeros" else self.strPad
            tenIn = F.pad(input=tenIn, pad=intPad, mode=mode, value=0.0)
        return tenIn


class BilinearUp(nn.Module):
    """2x bilinear upsampling."""

    def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
        return F.interpolate(input=tenIn, scale_factor=2.0, mode="bilinear", align_corners=False)


class PixelShuffleUp(nn.Module):
    """2x pixel shuffle upsampling."""

    def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(tenIn, upscale_factor=2)


class DownSample(nn.Module):
    """Bilinear downsampling by a factor."""

    def __init__(self, fltScale: float):
        super().__init__()
        self.fltScale = fltScale

    def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
        return F.interpolate(input=tenIn, scale_factor=self.fltScale, mode="bilinear", align_corners=False)


def _parse_basic_block(strType: str, intChans: list) -> tuple:
    """Parse the compact string notation into a sequential network."""
    net_main: list = []
    net_shortcut = None
    flt_stride = 1.0
    int_in = intChans[0]
    int_out = intChans[-1]

    for str_part in strType.split("+")[0].split("-"):
        if str_part.startswith("evenize"):
            pad_mode = "zeros"
            if "(" in str_part:
                opts = str_part.split("(")[1].split(")")[0].split(",")
                if "replpad" in opts:
                    pad_mode = "replicate"
                if "reflpad" in opts:
                    pad_mode = "reflect"
            net_main.append(Evenize(pad_mode))

        elif str_part.startswith("conv"):
            ksize = 3
            pad = 1
            pad_mode = "zeros"
            if "(" in str_part:
                ksize = int(str_part.split("(")[1].split(")")[0].split(",")[0])
                pad = int(math.floor(0.5 * (ksize - 1)))
                opts = str_part.split("(")[1].split(")")[0].split(",")
                if "replpad" in opts:
                    pad_mode = "replicate"
                if "reflpad" in opts:
                    pad_mode = "reflect"
            if "nopad" in strType.split("+"):
                pad = 0
            net_main.append(
                nn.Conv2d(
                    in_channels=intChans[0],
                    out_channels=intChans[1],
                    kernel_size=ksize,
                    stride=1,
                    padding=pad,
                    padding_mode=pad_mode,
                    bias=True,
                )
            )
            intChans = intChans[1:]
            flt_stride *= 1.0

        elif str_part.startswith("sconv"):
            ksize = 3
            pad = 1
            pad_mode = "zeros"
            if "(" in str_part:
                ksize = int(str_part.split("(")[1].split(")")[0].split(",")[0])
                pad = int(math.floor(0.5 * (ksize - 1)))
                opts = str_part.split("(")[1].split(")")[0].split(",")
                if "replpad" in opts:
                    pad_mode = "replicate"
                if "reflpad" in opts:
                    pad_mode = "reflect"
            if "nopad" in strType.split("+"):
                pad = 0
            net_main.append(
                nn.Conv2d(
                    in_channels=intChans[0],
                    out_channels=intChans[1],
                    kernel_size=ksize,
                    stride=2,
                    padding=pad,
                    padding_mode=pad_mode,
                    bias=True,
                )
            )
            intChans = intChans[1:]
            flt_stride *= 2.0

        elif str_part.startswith("up"):
            if "(" in str_part:
                uptype = str_part.split("(")[1].split(")")[0].split(",")[0]
                if uptype == "shuffle":
                    net_main.append(PixelShuffleUp())
                    flt_stride *= 0.5
                    continue
            net_main.append(BilinearUp())
            flt_stride *= 0.5

        elif str_part.startswith("prelu"):
            init_val = float(str_part.split("(")[1].split(")")[0].split(",")[0])
            net_main.append(nn.PReLU(num_parameters=1, init=init_val))

        else:
            raise AssertionError(f"Unknown block type: {str_part}")

    # Parse shortcut
    for str_part in strType.split("+")[1:]:
        if str_part.startswith("skip"):
            if int_in == int_out and flt_stride == 1.0:
                net_shortcut = nn.Identity()
            elif int_in != int_out and flt_stride == 1.0:
                net_shortcut = nn.Conv2d(
                    in_channels=int_in,
                    out_channels=int_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif int_in == int_out and flt_stride != 1.0:
                net_shortcut = DownSample(1.0 / flt_stride)
            elif int_in != int_out and flt_stride != 1.0:
                net_shortcut = nn.Sequential(
                    DownSample(1.0 / flt_stride),
                    nn.Conv2d(
                        in_channels=int_in,
                        out_channels=int_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                )

    return nn.Sequential(*net_main), net_shortcut


class Basic(nn.Module):
    """Configurable conv/activation block built from a compact string notation."""

    def __init__(self, strType: str, intChans: list, objScratch=None):
        super().__init__()
        self.netEvenize = None
        self.netMain = None
        self.netShortcut = None

        # Check for evenize prefix
        parts = strType.split("+")[0].split("-")
        if parts[0].startswith("evenize"):
            pad_mode = "zeros"
            if "(" in parts[0]:
                opts = parts[0].split("(")[1].split(")")[0].split(",")
                if "replpad" in opts:
                    pad_mode = "replicate"
                if "reflpad" in opts:
                    pad_mode = "reflect"
            self.netEvenize = Evenize(pad_mode)
            # Drop the evenize token from the main sequence (it is handled by
            # netEvenize); re-inserting it would add a phantom param-less module
            # and shift every netMain index, breaking checkpoint loading.
            strType = "-".join(parts[1:])

        main, shortcut = _parse_basic_block(strType, intChans)
        self.netMain = main
        self.netShortcut = shortcut

    def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
        if self.netEvenize is not None:
            tenIn = self.netEvenize(tenIn)
        assert self.netMain is not None
        tenOut = self.netMain(tenIn)
        if self.netShortcut is not None:
            tenOut = tenOut + self.netShortcut(tenIn)
        return tenOut


# ============================================================================
# PWC-Net style pyramidal optical flow estimation
# ============================================================================


class Extractor(nn.Module):
    """Pyramid feature extractor."""

    def __init__(self):
        super().__init__()
        self.netOne = Basic(
            "evenize(replpad)-sconv(2)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)",
            [3, 32, 32, 32],
        )
        self.netTwo = Basic(
            "evenize(replpad)-sconv(2)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)",
            [32, 32, 32, 32],
        )
        self.netThr = Basic(
            "evenize(replpad)-sconv(2)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)",
            [32, 32, 32, 32],
        )

    def forward(self, tenIn: torch.Tensor) -> list:
        tenOne = self.netOne(tenIn)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = F.avg_pool2d(input=tenThr, kernel_size=2, stride=2, count_include_pad=False)
        tenFiv = F.avg_pool2d(input=tenFou, kernel_size=2, stride=2, count_include_pad=False)
        return [tenOne, tenTwo, tenThr, tenFou, tenFiv]


class Decoder(nn.Module):
    """Flow decoder with cost volume at each pyramid level."""

    def __init__(self, intChannels: int):
        super().__init__()
        self.netCostacti = nn.PReLU(num_parameters=1, init=0.25)
        self.netMain = Basic(
            "conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)"
            "-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)"
            "-conv(3,replpad)-prelu(0.25)-conv(3,replpad)",
            [intChannels, 128, 128, 96, 64, 32, 2],
        )

    def forward(self, tenOne: torch.Tensor, tenTwo: torch.Tensor, tenFlow: Optional[torch.Tensor]) -> torch.Tensor:
        if tenFlow is not None:
            tenFlowUp = 2.0 * F.interpolate(input=tenFlow, scale_factor=2.0, mode="bilinear", align_corners=False)
        else:
            tenFlowUp = None

        tenMain = [tenOne]

        if tenFlow is None:
            tenMain.append(self.netCostacti(costvol_func(tenOne, tenTwo)))
            residual = self.netMain(torch.cat(tenMain, dim=1))
            return residual
        else:
            # tenFlow is not None → tenFlowUp is also not None
            assert tenFlowUp is not None
            tenMain.append(self.netCostacti(costvol_func(tenOne, backwarp(tenTwo, tenFlowUp.detach()))))
            tenMain.append(tenFlowUp)
            residual = self.netMain(torch.cat(tenMain, dim=1))
            return tenFlowUp + residual


class Network(nn.Module):
    """PWC-Net style bidirectional optical flow network."""

    def __init__(self):
        super().__init__()
        self.netExtractor = Extractor()
        self.netFiv = Decoder(32 + 81)    # 32 feat + 81 cost
        self.netFou = Decoder(32 + 81 + 2)  # +2 flow
        self.netThr = Decoder(32 + 81 + 2)
        self.netTwo = Decoder(32 + 81 + 2)
        self.netOne = Decoder(32 + 81 + 2)

    def bidir(self, tenOne: torch.Tensor, tenTwo: torch.Tensor) -> tuple:
        """Estimate bidirectional optical flow.

        Args:
            tenOne: Frame 0 [B, 3, H, W], normalized to [0, 1]
            tenTwo: Frame 1 [B, 3, H, W], normalized to [0, 1]

        Returns:
            Tuple of (forward flow, backward flow) each [B, 2, H, W]
        """
        # Shared feature extraction, then split
        feats = self.netExtractor(torch.cat([tenOne, tenTwo], dim=0))
        tenOne_feats, tenTwo_feats = zip(*[torch.split(f, [tenOne.shape[0], tenTwo.shape[0]], dim=0) for f in feats])

        # Forward flow (frame0 -> frame1)
        tenFwd = None
        tenFwd = self.netFiv(tenOne_feats[-1], tenTwo_feats[-1], tenFwd)
        tenFwd = self.netFou(tenOne_feats[-2], tenTwo_feats[-2], tenFwd)
        tenFwd = self.netThr(tenOne_feats[-3], tenTwo_feats[-3], tenFwd)
        tenFwd = self.netTwo(tenOne_feats[-4], tenTwo_feats[-4], tenFwd)
        tenFwd = self.netOne(tenOne_feats[-5], tenTwo_feats[-5], tenFwd)

        # Backward flow (frame1 -> frame0)
        tenBwd = None
        tenBwd = self.netFiv(tenTwo_feats[-1], tenOne_feats[-1], tenBwd)
        tenBwd = self.netFou(tenTwo_feats[-2], tenOne_feats[-2], tenBwd)
        tenBwd = self.netThr(tenTwo_feats[-3], tenOne_feats[-3], tenBwd)
        tenBwd = self.netTwo(tenTwo_feats[-4], tenOne_feats[-4], tenBwd)
        tenBwd = self.netOne(tenTwo_feats[-5], tenOne_feats[-5], tenBwd)

        return tenFwd, tenBwd


# ============================================================================
# Image pyramid encoder-decoder for motion refinement
# ============================================================================

c: int = 16  # base channel count


def _conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True),
        nn.PReLU(out_planes),
    )


def _deconv(in_planes: int, out_planes: int) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes),
    )


class Conv2(nn.Module):
    """2-layer convolution block with stride."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 2):
        super().__init__()
        self.conv1 = _conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = _conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ImgPyramid(nn.Module):
    """Multi-scale image feature pyramid."""

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x: torch.Tensor) -> list:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return [x1, x2, x3, x4]


class EncDec(nn.Module):
    """Encoder-decoder with cross-attention (C/H/W) for flow refinement.

    Implements the tri-plane attention (Channel, Height, Width) for
    spatial feature recalibration.
    """

    def __init__(self, branch: int):
        super().__init__()
        self.branch = branch

        self.down0 = Conv2(8, 2 * c)
        self.down1 = Conv2(6 * c, 4 * c)
        self.down2 = Conv2(12 * c, 8 * c)
        self.down3 = Conv2(24 * c, 16 * c)

        self.up0 = _deconv(48 * c, 8 * c)
        self.up1 = _deconv(16 * c, 4 * c)
        self.up2 = _deconv(8 * c, 2 * c)
        self.up3 = _deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 2 * self.branch, 3, 1, 1)
        self.conv_m = nn.Conv2d(c, 1, 3, 1, 1)

        # Tri-plane attention
        self.conv_C = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16 * c, 16 * 16 * c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.conv_H = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(16 * c, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.conv_W = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(16 * c, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, flow0: torch.Tensor, flow1: torch.Tensor, im0: torch.Tensor, im1: torch.Tensor, c0: list, c1: list
    ) -> tuple:
        N, C_, H_, W_ = im0.shape

        # ---- Level 0 ----
        wim1 = backwarp(im1, flow0)
        wim0 = backwarp(im0, flow1)
        s0_0 = self.down0(torch.cat((flow0, im0, wim1), 1))
        s1_0 = self.down0(torch.cat((flow1, im1, wim0), 1))

        # ---- Level 1 ----
        flow0_half = F.interpolate(flow0, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1_half = F.interpolate(flow1, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp(torch.cat((s0_0, c0[0]), 1), flow1_half)
        wf1 = backwarp(torch.cat((s1_0, c1[0]), 1), flow0_half)
        s0_1 = self.down1(torch.cat((s0_0, c0[0], wf1), 1))
        s1_1 = self.down1(torch.cat((s1_0, c1[0], wf0), 1))

        # ---- Level 2 ----
        flow0_quart = F.interpolate(flow0_half, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1_quart = F.interpolate(flow1_half, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp(torch.cat((s0_1, c0[1]), 1), flow1_quart)
        wf1 = backwarp(torch.cat((s1_1, c1[1]), 1), flow0_quart)
        s0_2 = self.down2(torch.cat((s0_1, c0[1], wf1), 1))
        s1_2 = self.down2(torch.cat((s1_1, c1[1], wf0), 1))

        # ---- Level 3 ----
        flow0_eighth = F.interpolate(flow0_quart, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1_eighth = F.interpolate(flow1_quart, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp(torch.cat((s0_2, c0[2]), 1), flow1_eighth)
        wf1 = backwarp(torch.cat((s1_2, c1[2]), 1), flow0_eighth)
        s0_3 = self.down3(torch.cat((s0_2, c0[2], wf1), 1))
        s1_3 = self.down3(torch.cat((s1_2, c1[2], wf0), 1))

        # ---- Tri-plane attention ----
        s0_3_c = self.conv_C(s0_3).view(N, 16, -1, 1, 1)
        s0_3_h = self.conv_H(s0_3).view(N, 16, 1, -1, 1)
        s0_3_w = self.conv_W(s0_3).view(N, 16, 1, 1, -1)
        cube0 = (s0_3_c * s0_3_h * s0_3_w).mean(1)
        s0_3 = s0_3 * cube0

        s1_3_c = self.conv_C(s1_3).view(N, 16, -1, 1, 1)
        s1_3_h = self.conv_H(s1_3).view(N, 16, 1, -1, 1)
        s1_3_w = self.conv_W(s1_3).view(N, 16, 1, 1, -1)
        cube1 = (s1_3_c * s1_3_h * s1_3_w).mean(1)
        s1_3 = s1_3 * cube1

        # ---- Decoder ----
        flow0_sixteenth = F.interpolate(flow0_eighth, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1_sixteenth = F.interpolate(flow1_eighth, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp(torch.cat((s0_3, c0[3]), 1), flow1_sixteenth)
        wf1 = backwarp(torch.cat((s1_3, c1[3]), 1), flow0_sixteenth)

        x0 = self.up0(torch.cat((s0_3, c0[3], wf1), 1))
        x1 = self.up0(torch.cat((s1_3, c1[3], wf0), 1))

        x0 = self.up1(torch.cat((s0_2, x0), 1))
        x1 = self.up1(torch.cat((s1_2, x1), 1))

        x0 = self.up2(torch.cat((s0_1, x0), 1))
        x1 = self.up2(torch.cat((s1_1, x1), 1))

        x0 = self.up3(torch.cat((s0_0, x0), 1))
        x1 = self.up3(torch.cat((s1_0, x1), 1))

        m0 = self.sigmoid(self.conv_m(x0)) * 0.8 + 0.1
        m1 = self.sigmoid(self.conv_m(x1)) * 0.8 + 0.1

        x0 = self.conv(x0)
        x1 = self.conv(x1)

        return x0, x1, m0.repeat(1, self.branch, 1, 1), m1.repeat(1, self.branch, 1, 1)


# ============================================================================
# Many-to-many forward warping (softmax splatting)
# ============================================================================


def forwarp_mframe_mask(
    tenIn1: torch.Tensor,
    tenFlow1: torch.Tensor,
    t1: torch.Tensor,
    tenIn2: torch.Tensor,
    tenFlow2: torch.Tensor,
    t2: torch.Tensor,
    tenMetric1: Optional[torch.Tensor] = None,
    tenMetric2: Optional[torch.Tensor] = None,
) -> tuple:
    """Many-to-many forward warping with softmax splatting.

    Warps both input frames using multiple flow candidates and
    blends them with softmax-weighted normalization.

    Args:
        tenIn1: First frame [B*N, 3, H, W] (N = branches)
        tenFlow1: Forward flow candidates [B*N, 2, H, W]
        t1: Time weight for frame 0 [B*N, 1, 1, 1]
        tenIn2: Second frame [B*N, 3, H, W]
        tenFlow2: Backward flow candidates [B*N, 2, H, W]
        t2: Time weight for frame 1
        tenMetric1: Optical/photo consistency weight for frame 0
        tenMetric2: Optical/photo consistency weight for frame 1

    Returns:
        tuple: (splatted_output [B, 3, H, W], mask [B, 1, H, W])
    """

    def one_fdir(tenIn: torch.Tensor, tenFlow: torch.Tensor, td: torch.Tensor, tenMetric: torch.Tensor) -> tuple:
        # Weight input by softmax metric
        metricW = (tenMetric).clip(-20.0, 20.0).exp()
        tenCat = torch.cat(
            [
                tenIn * td * metricW,
                td * metricW,
            ],
            dim=1,
        )
        tenOut = softsplat_func(tenCat, tenFlow)
        return tenOut[:, :-1, :, :], tenOut[:, -1:, :, :] + 0.0000001

    flow_num = tenFlow1.shape[0]
    tenOut = 0.0
    tenNormalize = 0.0
    assert tenMetric1 is not None
    assert tenMetric2 is not None
    for idx in range(flow_num):
        tenOutF, tenNormalizeF = one_fdir(
            tenIn1[idx : idx + 1], tenFlow1[idx : idx + 1], t1[idx : idx + 1], tenMetric1[idx : idx + 1]
        )
        tenOutB, tenNormalizeB = one_fdir(
            tenIn2[idx : idx + 1], tenFlow2[idx : idx + 1], t2[idx : idx + 1], tenMetric2[idx : idx + 1]
        )
        tenOut = tenOut + tenOutF + tenOutB
        tenNormalize = tenNormalize + tenNormalizeF + tenNormalizeB

    return tenOut / tenNormalize, tenNormalize < 0.00001


# ============================================================================
# Main M2M PWC model
# ============================================================================


class M2M_PWC(nn.Module):
    """Many-to-Many Splatting for Efficient Video Frame Interpolation.

    Pipeline:
        1. Estimate bidirectional optical flow (PWC-Net style)
        2. Refine flow with MotionRefineNet (multi-branch + tri-plane attention)
        3. Many-to-many softmax splatting to synthesize interpolated frame(s)

    Supports arbitrary interpolation timesteps and multiple output frames
    in a single forward pass.
    """

    def __init__(self, ratio: int = 4):
        super().__init__()
        self.branch = 4
        self.ratio = ratio

        self.netFlow = Network()
        self.paramAlpha = nn.Parameter(10.0 * torch.ones(1, 1, 1, 1))

        class MotionRefineNet(nn.Module):
            """Motion refinement with image guidance and tri-plane attention."""

            def __init__(self, branch: int):
                super().__init__()
                self.branch = branch
                self.img_pyramid = ImgPyramid()
                self.motion_encdec = EncDec(branch)

            def forward(self, flow0: torch.Tensor, flow1: torch.Tensor, im0: torch.Tensor, im1: torch.Tensor, ratio: int) -> tuple:
                flow0_up = ratio * F.interpolate(input=flow0, scale_factor=ratio, mode="bilinear", align_corners=False)
                flow1_up = ratio * F.interpolate(input=flow1, scale_factor=ratio, mode="bilinear", align_corners=False)

                c0 = self.img_pyramid(im0)
                c1 = self.img_pyramid(im1)

                flow_res = self.motion_encdec(flow0_up, flow1_up, im0, im1, c0, c1)

                flow0_refined = flow0_up.repeat(1, self.branch, 1, 1) + flow_res[0]
                flow1_refined = flow1_up.repeat(1, self.branch, 1, 1) + flow_res[1]

                return flow0_refined, flow1_refined, flow_res[2], flow_res[3]

        self.MRN = MotionRefineNet(self.branch)

    def forward(
        self, im0: torch.Tensor, im1: torch.Tensor, fltTimes: Optional[list] = None, ratio: Optional[int] = None
    ) -> list:
        """Interpolate frame(s) between im0 and im1.

        Args:
            im0: First frame [B, 3, H, W], normalized to [0, 1]
            im1: Second frame [B, 3, H, W], normalized to [0, 1]
            fltTimes: Interpolation timesteps (e.g., [0.5] for single mid-frame).
                      Each value must be in (0, 1).
            ratio: Upsampling ratio for flow (default: self.ratio = 4)

        Returns:
            List of interpolated frames [B, 3, H, W] (one per timestep)
        """
        device = im0.device
        if fltTimes is None:
            fltTimes = [torch.full((im0.shape[0], 1, 1, 1), 0.5, device=device)]
        if ratio is None:
            ratio = self.ratio

        B, _, H, W = im0.shape

        # Pad to multiple of (ratio * 16)
        intPadr = ((ratio * 16) - (W % (ratio * 16))) % (ratio * 16)
        intPadb = ((ratio * 16) - (H % (ratio * 16))) % (ratio * 16)

        im0_pad = F.pad(input=im0, pad=[0, intPadr, 0, intPadb], mode="replicate")
        im1_pad = F.pad(input=im1, pad=[0, intPadr, 0, intPadb], mode="replicate")

        _, _, H_, W_ = im0_pad.shape

        # Normalize
        with torch.set_grad_enabled(False):
            tenStats = [im0_pad, im1_pad]
            tenMean_ = torch.stack([tenIn.mean([1, 2, 3], keepdim=True) for tenIn in tenStats]).mean(dim=0)
            tenStd_ = (
                torch.stack(
                    [
                        tenIn.std([1, 2, 3], keepdim=True).square()
                        + (tenMean_ - tenIn.mean([1, 2, 3], keepdim=True)).square()
                        for tenIn in tenStats
                    ]
                ).mean(dim=0)
            ).sqrt()

            im0_norm = (im0_pad - tenMean_) / (tenStd_ + 0.0000001)
            im1_norm = (im1_pad - tenMean_) / (tenStd_ + 0.0000001)

        im0_o = im0_norm
        im1_o = im1_norm

        # Downsample for flow estimation
        im0_ = F.interpolate(input=im0_norm, scale_factor=2.0 / ratio, mode="bilinear", align_corners=False)
        im1_ = F.interpolate(input=im1_norm, scale_factor=2.0 / ratio, mode="bilinear", align_corners=False)

        # Bidirectional flow
        tenFwd, tenBwd = self.netFlow.bidir(im0_, im1_)

        # Motion refinement (multi-branch)
        tenFwd, tenBwd, WeiMF, WeiMB = self.MRN(tenFwd, tenBwd, im0_norm, im1_norm, ratio)

        # Prepare fltTimes tensors
        fltTimes_tensors = []
        for t in fltTimes:
            if isinstance(t, torch.Tensor):
                fltTimes_tensors.append(t.to(device))
            else:
                fltTimes_tensors.append(torch.full((B, 1, 1, 1), t, device=device))

        outputs = []
        for fltTime_ in fltTimes_tensors:
            # Reshape for branch processing
            im0_repeat = im0_o.repeat(1, self.branch, 1, 1)  # B, C*N, H, W
            im1_repeat = im1_o.repeat(1, self.branch, 1, 1)

            tenStd_repeat = tenStd_.repeat(1, self.branch, 1, 1)
            tenMean_repeat = tenMean_.repeat(1, self.branch, 1, 1)

            tenFwd_b = tenFwd.reshape(B * self.branch, 2, H_, W_)
            tenBwd_b = tenBwd.reshape(B * self.branch, 2, H_, W_)
            WeiMF_b = WeiMF.reshape(B * self.branch, 1, H_, W_)
            WeiMB_b = WeiMB.reshape(B * self.branch, 1, H_, W_)
            im0_b = im0_repeat.reshape(B * self.branch, 3, H_, W_)
            im1_b = im1_repeat.reshape(B * self.branch, 3, H_, W_)
            tenStd_b = tenStd_repeat.reshape(B * self.branch, 1, 1, 1)
            tenMean_b = tenMean_repeat.reshape(B * self.branch, 1, 1, 1)
            fltTime_b = fltTime_.repeat(1, self.branch, 1, 1).reshape(B * self.branch, 1, 1, 1)

            # Photo-consistency weights
            tenPhotoone = (
                (1.0 - (WeiMF_b * (im0_b - backwarp(im1_b, tenFwd_b).detach()).abs().mean(dim=1, keepdim=True)))
                .clip(0.001)
                .square()
            )
            tenPhototwo = (
                (1.0 - (WeiMB_b * (im1_b - backwarp(im0_b, tenBwd_b).detach()).abs().mean(dim=1, keepdim=True)))
                .clip(0.001)
                .square()
            )

            # Time-scaled flows
            t0_b = fltTime_b
            flow0_s = tenFwd_b * t0_b
            metric0 = self.paramAlpha * tenPhotoone

            t1_b = 1.0 - fltTime_b
            flow1_s = tenBwd_b * t1_b
            metric1 = self.paramAlpha * tenPhototwo

            # Many-to-many splatting (all 4D: [B*N, C, H, W])
            tenOutput, mask = forwarp_mframe_mask(im0_b, flow0_s, t1_b, im1_b, flow1_s, t0_b, metric0, metric1)

            # Handle uncovered pixels with fallback (weighted average of inputs)
            tenOutput = tenOutput + mask * (t1_b.mean(0, keepdim=True) * im0_o + t0_b.mean(0, keepdim=True) * im1_o)

            # Denormalize
            outputs.append(tenOutput * (tenStd_ + 0.0000001) + tenMean_)

        # Crop to original size
        return [out[:, :, :H, :W] for out in outputs]
