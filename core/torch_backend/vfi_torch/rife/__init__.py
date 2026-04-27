"""
RIFE 模型实现。
基于 https://github.com/hzwer/Practical-RIFE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

from ..base import BaseVFIModel, VFIConfig, VFIResult, ModelType
from ..utils import get_device, clear_cache


class ResConv(nn.Module):
    """Residual convolution block."""
    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


# Cache for backward warping grid
_backwarp_tenGrid: Dict = {}


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Warp input tensor using flow field."""
    device = tenInput.device
    k = (str(device), str(tenFlow.size()))
    
    if k not in _backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        _backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
    ], 1)

    g = (_backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    
    if g.dtype != tenInput.dtype:
        g = g.to(tenInput.dtype)

    return F.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, 
         padding: int = 1, dilation: int = 1, arch_ver: str = "4.0") -> nn.Sequential:
    """Create convolution block based on architecture version."""
    if arch_ver == "4.0":
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True),
            nn.PReLU(out_planes),
        )
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True),
        nn.LeakyReLU(0.2, True),
    )


def deconv(in_planes: int, out_planes: int, kernel_size: int = 4, stride: int = 2, 
           padding: int = 1, arch_ver: str = "4.0") -> nn.Sequential:
    """Create transposed convolution block."""
    if arch_ver == "4.0":
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True),
            nn.PReLU(out_planes),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True),
        nn.LeakyReLU(0.2, True),
    )


class Conv2(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 2, arch_ver: str = "4.0"):
        super().__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1, arch_ver=arch_ver)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1, arch_ver=arch_ver)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class IFBlock(nn.Module):
    """Intermediate Flow estimation block."""
    def __init__(self, in_planes: int, c: int = 64, arch_ver: str = "4.0"):
        super().__init__()
        self.arch_ver = arch_ver
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1, arch_ver=arch_ver),
            conv(c // 2, c, 3, 2, 1, arch_ver=arch_ver),
        )

        if arch_ver in ["4.0", "4.2", "4.3"]:
            self.convblock = nn.Sequential(*[conv(c, c, arch_ver=arch_ver) for _ in range(8)])
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        elif arch_ver in ["4.5", "4.6", "4.7", "4.10", "4.17"]:
            self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
                nn.PixelShuffle(2)
            )
        elif arch_ver == "4.26":
            self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
                nn.PixelShuffle(2)
            )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        
        feat = self.conv0(x)
        if self.arch_ver in ["4.2", "4.3", "4.5", "4.6", "4.7", "4.10", "4.17", "4.26"]:
            feat = self.convblock(feat)
        else:
            feat = self.convblock(feat) + feat

        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        
        if self.arch_ver == "4.26":
            mask = tmp[:, 4:5]
            feat_out = tmp[:, 5:]
            return flow, mask, feat_out
        
        mask = tmp[:, 4:5]
        return flow, mask


class Head(nn.Module):
    """Feature extraction head for RIFE 4.26."""
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class IFNet(nn.Module):
    """Intermediate Flow Network."""
    def __init__(self, arch_ver: str = "4.22"):
        super().__init__()
        self.arch_ver = arch_ver
        
        # Architecture-specific configuration
        if arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
            self.block0 = IFBlock(7, c=192, arch_ver=arch_ver)
            self.block1 = IFBlock(8 + 4, c=128, arch_ver=arch_ver)
            self.block2 = IFBlock(8 + 4, c=96, arch_ver=arch_ver)
            self.block3 = IFBlock(8 + 4, c=64, arch_ver=arch_ver)
            self.num_blocks = 4
        elif arch_ver in ["4.7", "4.10", "4.17"]:
            self.block0 = IFBlock(7 + 8, c=192, arch_ver=arch_ver)
            self.block1 = IFBlock(8 + 4 + 8, c=128, arch_ver=arch_ver)
            self.block2 = IFBlock(8 + 4 + 8, c=96, arch_ver=arch_ver)
            self.block3 = IFBlock(8 + 4 + 8, c=64, arch_ver=arch_ver)
            self.encode = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1),
                nn.ConvTranspose2d(16, 4, 4, 2, 1)
            )
            self.num_blocks = 4
        elif arch_ver == "4.26":
            self.block0 = IFBlock(7 + 8, c=192, arch_ver=arch_ver)
            self.block1 = IFBlock(8 + 4 + 8 + 8, c=128, arch_ver=arch_ver)
            self.block2 = IFBlock(8 + 4 + 8 + 8, c=96, arch_ver=arch_ver)
            self.block3 = IFBlock(8 + 4 + 8 + 8, c=64, arch_ver=arch_ver)
            self.block4 = IFBlock(8 + 4 + 8 + 8, c=32, arch_ver=arch_ver)
            self.encode = Head()
            self.num_blocks = 5
        else:
            # Default to 4.22 (same as 4.7)
            self.block0 = IFBlock(7 + 8, c=192, arch_ver="4.7")
            self.block1 = IFBlock(8 + 4 + 8, c=128, arch_ver="4.7")
            self.block2 = IFBlock(8 + 4 + 8, c=96, arch_ver="4.7")
            self.block3 = IFBlock(8 + 4 + 8, c=64, arch_ver="4.7")
            self.encode = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1),
                nn.ConvTranspose2d(16, 4, 4, 2, 1)
            )
            self.num_blocks = 4

    def forward(self, img0, img1, timestep=0.5, scale_list=None, fastmode=True, ensemble=False):
        if scale_list is None:
            scale_list = [8, 4, 2, 1]
        
        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if not torch.is_tensor(timestep):
            timestep = torch.ones(img0.shape[0], 1, img0.shape[2], img0.shape[3], device=img0.device, dtype=img0.dtype) * timestep

        # Encode features for newer architectures
        f0: Optional[torch.Tensor] = None
        f1: Optional[torch.Tensor] = None
        if hasattr(self, 'encode'):
            f0 = self.encode(img0[:, :3])
            f1 = self.encode(img1[:, :3])

        flow: Optional[torch.Tensor] = None
        mask: Optional[torch.Tensor] = None
        feat: Optional[torch.Tensor] = None
        blocks = [self.block0, self.block1, self.block2, self.block3]
        if hasattr(self, 'block4'):
            blocks.append(self.block4)

        for i, block in enumerate(blocks[:self.num_blocks]):
            if i >= len(scale_list):
                break
                
            scale = scale_list[i]
            
            if flow is None:
                if self.arch_ver in ["4.7", "4.10", "4.17", "4.26"]:
                    assert f0 is not None and f1 is not None
                    if self.arch_ver == "4.26":
                        flow, mask, feat = block(
                            torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                            None, scale=scale
                        )
                    else:
                        flow, mask = block(
                            torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                            None, scale=scale
                        )
                else:
                    flow, mask = block(
                        torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                        None, scale=scale
                    )
            else:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                
                if self.arch_ver in ["4.7", "4.10", "4.17", "4.26"]:
                    assert f0 is not None and f1 is not None and mask is not None
                    if self.arch_ver == "4.26":
                        assert feat is not None
                        fd, mask, feat = block(
                            torch.cat((
                                warped_img0[:, :3], warped_img1[:, :3],
                                warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4]),
                                timestep, mask, feat
                            ), 1),
                            flow, scale=scale
                        )
                    else:
                        fd, mask = block(
                            torch.cat((
                                warped_img0[:, :3], warped_img1[:, :3],
                                warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4]),
                                timestep, mask
                            ), 1),
                            flow, scale=scale
                        )
                    flow = flow + fd
                else:
                    assert mask is not None
                    f0_delta, m0 = block(
                        torch.cat((warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1),
                        flow, scale=scale
                    )
                    flow = flow + f0_delta
                    mask = mask + m0

        assert flow is not None and mask is not None
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        merged = warped_img0 * mask + warped_img1 * (1 - mask)

        return merged[:, :, :h, :w]


class RIFEModel(BaseVFIModel):
    """
    RIFE (Real-Time Intermediate Flow Estimation) Model.
    
    Supports versions 4.0 - 4.26.
    """
    
    MODEL_NAME = "rife"
    SUPPORTED_VERSIONS = ["4.0", "4.2", "4.3", "4.5", "4.6", "4.7", "4.10", "4.17", "4.22", "4.26"]
    DEFAULT_VERSION = "4.22"
    MIN_INPUT_FRAMES = 2
    
    # Version to checkpoint mapping
    CKPT_VERSION_MAP = {
        "4.0": "sudo_rife4_269.662_testV1_scale1.pth",
        "4.7": "rife47.pth",
        "4.9": "rife49.pth",
        "4.17": "rife417.pth",
        "4.26": "rife426.pth",
    }
    
    def __init__(self, config: VFIConfig):
        super().__init__(config)
        self.arch_ver = self._get_arch_version(config.model_version)
        
    def _get_arch_version(self, version: str) -> str:
        """Map model version to architecture version."""
        version_map = {
            "4.0": "4.0", "4.1": "4.0",
            "4.2": "4.2",
            "4.3": "4.3", "4.4": "4.3",
            "4.5": "4.5",
            "4.6": "4.6",
            "4.7": "4.7", "4.8": "4.7", "4.9": "4.7",
            "4.10": "4.10", "4.11": "4.10", "4.12": "4.10",
            "4.17": "4.17",
            "4.22": "4.7",  # 4.22 uses 4.7 architecture
            "4.26": "4.26",
        }
        return version_map.get(version, "4.7")
    
    def load_model(self, checkpoint_path: str = "") -> None:
        """Load RIFE model weights."""
        if self._model is not None:
            return
            
        # Create model
        self._model = IFNet(arch_ver=self.arch_ver)
        
        # Use provided path or config path
        ckpt_path = checkpoint_path or self._config.checkpoint_path
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self._model.load_state_dict(state_dict, strict=False)
        
        self._model = self._model.to(self.device)
        self._model.eval()
        
        if self._config.fp16 and self.device.type == "cuda":
            self._model = self._model.half()
        
        self._is_loaded = True
    
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs
    ) -> VFIResult:
        """
        Interpolate a single frame between two input frames.
        
        Args:
            frame0: First input frame [C, H, W] or [B, C, H, W]
            frame1: Second input frame [C, H, W] or [B, C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)
            
        Returns:
            VFIResult with interpolated frame
        """
        if not self._is_loaded:
            self.load_model()
        
        # Ensure batch dimension
        squeeze_output = False
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
            frame1 = frame1.unsqueeze(0)
            squeeze_output = True
        
        # Move to device
        frame0 = frame0.to(self.device)
        frame1 = frame1.to(self.device)
        
        # Handle precision
        if self._config.fp16 and self.device.type == "cuda":
            frame0 = frame0.half()
            frame1 = frame1.half()
        
        # Calculate scale list based on scale factor
        scale = self._config.scale
        if self.arch_ver == "4.26":
            scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        else:
            scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        
        assert self._model is not None
        with torch.no_grad():
            output = self._model(
                frame0, frame1,
                timestep=timestep,
                scale_list=scale_list,
                fastmode=self._config.fast_mode,
                ensemble=self._config.ensemble,
            )
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return VFIResult(frame=output, model_type=ModelType.RIFE)
