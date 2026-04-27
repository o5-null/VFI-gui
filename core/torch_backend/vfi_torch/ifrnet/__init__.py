"""IFRNet (Intermediate Feature Refine Network) 模型实现。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path

from ..base import BaseVFIModel, VFIConfig, VFIResult, ModelType
from ..utils import load_model_weights, download_model


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp feature map using optical flow."""
    B, C, H, W = x.size()
    
    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=x.device),
        torch.arange(0, W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
    
    # Normalize grid to [-1, 1]
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]
    
    output = F.grid_sample(
        x, vgrid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return output


class ResBlock(nn.Module):
    """Residual block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + self.skip(x))


class Encoder(nn.Module):
    """IFRNet encoder."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 7, 2, 3)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        
        self.res1 = ResBlock(32, 32)
        self.res2 = ResBlock(64, 64)
        self.res3 = ResBlock(128, 128)
        self.res4 = ResBlock(256, 256)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        f1 = F.relu(self.conv1(x))
        f1 = self.res1(f1)
        
        f2 = F.relu(self.conv2(f1))
        f2 = self.res2(f2)
        
        f3 = F.relu(self.conv3(f2))
        f3 = self.res3(f3)
        
        f4 = F.relu(self.conv4(f3))
        f4 = self.res4(f4)
        
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    """Decoder for level 4 features."""
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 2, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        
    def forward(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, flow], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.deconv(x)


class Decoder3(nn.Module):
    """Decoder for level 3 features."""
    
    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2 + 2, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        
    def forward(self, x: torch.Tensor, f: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, f, flow], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.deconv(x)


class Decoder2(nn.Module):
    """Decoder for level 2 features."""
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2 + 2, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        
    def forward(self, x: torch.Tensor, f: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, f, flow], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.deconv(x)


class Decoder1(nn.Module):
    """Decoder for level 1 features (final output)."""
    
    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2 + 2, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 3, 3, 1, 1)
        
    def forward(self, x: torch.Tensor, f: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, f, flow], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


class IFRNet_S(nn.Module):
    """IFRNet Small model."""
    
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        
        # Flow prediction heads
        self.flow4 = nn.Conv2d(256, 2, 3, 1, 1)
        self.flow3 = nn.Conv2d(128, 2, 3, 1, 1)
        self.flow2 = nn.Conv2d(64, 2, 3, 1, 1)
        
    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img0: First frame [B, 3, H, W]
            img1: Second frame [B, 3, H, W]
            timestep: Time step [B, 1] or [B, 1, 1, 1]
        Returns:
            Interpolated frame [B, 3, H, W]
        """
        # Ensure timestep has correct shape
        if timestep.dim() == 2:
            timestep = timestep.view(-1, 1, 1, 1)
        
        # Encode both frames
        f1_0, f2_0, f3_0, f4_0 = self.encoder(img0)
        f1_1, f2_1, f3_1, f4_1 = self.encoder(img1)
        
        # Warp features from img1 to img0 position
        # Flow from img1 to middle position
        flow4 = self.flow4(f4_0 - f4_1) * timestep
        f4_1_warp = warp(f4_1, flow4)
        
        # Decoder 4
        d4 = self.decoder4(f4_0, flow4)
        
        # Flow 3
        flow3 = self.flow3(d4) + F.interpolate(flow4, scale_factor=2, mode='bilinear') * 0.5
        f3_1_warp = warp(f3_1, flow3)
        
        # Decoder 3
        d3 = self.decoder3(d4, f3_0 + f3_1_warp, flow3)
        
        # Flow 2
        flow2 = self.flow2(d3) + F.interpolate(flow3, scale_factor=2, mode='bilinear') * 0.5
        f2_1_warp = warp(f2_1, flow2)
        
        # Decoder 2
        d2 = self.decoder2(d3, f2_0 + f2_1_warp, flow2)
        
        # Flow 1
        flow1 = F.interpolate(flow2, scale_factor=2, mode='bilinear') * 0.5
        f1_1_warp = warp(f1_1, flow1)
        
        # Decoder 1 (final output)
        output = self.decoder1(d2, f1_0 + f1_1_warp, flow1)
        
        return output


class IFRNet_L(nn.Module):
    """IFRNet Large model (more channels)."""
    
    def __init__(self):
        super().__init__()
        # Larger channel counts
        self.encoder = Encoder()
        # Override with larger channels
        self.encoder.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.encoder.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.encoder.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.encoder.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        
        self.decoder4 = Decoder4(512)
        self.decoder3 = Decoder3(256)
        self.decoder2 = Decoder2(128)
        self.decoder1 = Decoder1(64)
        
        self.flow4 = nn.Conv2d(512, 2, 3, 1, 1)
        self.flow3 = nn.Conv2d(256, 2, 3, 1, 1)
        self.flow2 = nn.Conv2d(128, 2, 3, 1, 1)
        
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Similar to IFRNet_S but with larger feature maps
        if timestep.dim() == 2:
            timestep = timestep.view(-1, 1, 1, 1)
        
        f1_0, f2_0, f3_0, f4_0 = self.encoder(img0)
        f1_1, f2_1, f3_1, f4_1 = self.encoder(img1)
        
        flow4 = self.flow4(f4_0 - f4_1) * timestep
        f4_1_warp = warp(f4_1, flow4)
        
        d4 = self.decoder4(f4_0, flow4)
        
        flow3 = self.flow3(d4) + F.interpolate(flow4, scale_factor=2, mode='bilinear') * 0.5
        f3_1_warp = warp(f3_1, flow3)
        
        d3 = self.decoder3(d4, f3_0 + f3_1_warp, flow3)
        
        flow2 = self.flow2(d3) + F.interpolate(flow3, scale_factor=2, mode='bilinear') * 0.5
        f2_1_warp = warp(f2_1, flow2)
        
        d2 = self.decoder2(d3, f2_0 + f2_1_warp, flow2)
        
        flow1 = F.interpolate(flow2, scale_factor=2, mode='bilinear') * 0.5
        f1_1_warp = warp(f1_1, flow1)
        
        output = self.decoder1(d2, f1_0 + f1_1_warp, flow1)
        
        return output


class IFRNetModel(BaseVFIModel):
    """IFRNet model wrapper."""
    
    MODEL_FILES = {
        "IFRNet_S_Vimeo90K.pth": "IFRNet_S_Vimeo90K.pth",
        "IFRNet_L_Vimeo90K.pth": "IFRNet_L_Vimeo90K.pth",
        "IFRNet_S_GoPro.pth": "IFRNet_S_GoPro.pth",
        "IFRNet_L_GoPro.pth": "IFRNet_L_GoPro.pth",
    }
    
    ARCH_MAP = {
        "IFRNet_S_Vimeo90K.pth": "small",
        "IFRNet_S_GoPro.pth": "small",
        "IFRNet_L_Vimeo90K.pth": "large",
        "IFRNet_L_GoPro.pth": "large",
    }
    
    def __init__(self, config: Optional[VFIConfig] = None):
        if config is None:
            config = VFIConfig(model_type=ModelType.IFRNET)
        super().__init__(config)
        self._arch = "small"
        
    def _create_model(self) -> nn.Module:
        """Create IFRNet model instance based on architecture."""
        if self._arch == "large":
            return IFRNet_L()
        return IFRNet_S()
    
    def _get_model_url(self, ckpt_name: str) -> str:
        """Get download URL for model checkpoint."""
        from utils import BASE_MODEL_URLS
        base_url = BASE_MODEL_URLS.get("vsgan", "")
        return f"{base_url}{ckpt_name}"
    
    def load_model(self, checkpoint_path: str) -> None:
        """Load IFRNet model weights."""
        ckpt_name = Path(checkpoint_path).name
        
        if ckpt_name in self.ARCH_MAP:
            self._arch = self.ARCH_MAP[ckpt_name]
        
        if not Path(checkpoint_path).exists():
            url = self._get_model_url(ckpt_name)
            download_model("ifrnet", ckpt_name, str(Path(checkpoint_path).parent), url=url)
        
        state_dict = load_model_weights(checkpoint_path)
        
        if self._model is None:
            self._model = self._create_model()
            
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()
        
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
        **kwargs
    ) -> VFIResult:
        """
        Interpolate a single frame pair.
        
        Args:
            frame0: First frame [B, C, H, W] or [C, H, W]
            frame1: Second frame [B, C, H, W] or [C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)
            scale: Processing scale factor
            
        Returns:
            VFIResult with interpolated frame
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        squeeze_output = False
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
            frame1 = frame1.unsqueeze(0)
            squeeze_output = True
        
        # Apply scale if needed
        h, w = frame0.shape[2], frame0.shape[3]
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            frame0 = F.interpolate(frame0, (new_h, new_w), mode='bilinear')
            frame1 = F.interpolate(frame1, (new_h, new_w), mode='bilinear')
        
        timestep_tensor = torch.full(
            (frame0.shape[0], 1),
            timestep,
            device=frame0.device,
            dtype=frame0.dtype
        )
        
        with torch.no_grad():
            output = self._model(frame0, frame1, timestep_tensor)
        
        # Scale back if needed
        if scale != 1.0:
            output = F.interpolate(output, (h, w), mode='bilinear')
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return VFIResult(
            frame=output,
            model_type=ModelType.IFRNET,
            metadata={"timestep": timestep, "scale": scale}
        )
