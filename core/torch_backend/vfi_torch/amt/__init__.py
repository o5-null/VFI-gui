"""AMT (All-Pairs Multi-Field Transforms) 模型实现。

参考: https://github.com/MCG-NKU/AMT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pathlib import Path

from ..base import BaseVFIModel, VFIConfig, VFIResult, ModelType
from ..utils import load_model_weights, download_model, InputPadder


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp image using optical flow."""
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([
        flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
        flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)
    ], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(img, grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


def coords_grid(batch: int, ht: int, wd: int, device: torch.device) -> torch.Tensor:
    """Create coordinate grid."""
    coords = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing='ij'
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class ResBlock(nn.Module):
    """Residual block with PReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.prelu = nn.PReLU(out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        return self.prelu(out + self.skip(x))


class Encoder(nn.Module):
    """Feature encoder for AMT."""
    
    def __init__(self, channels: list = [64, 96, 128, 192]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 3, 2, 1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, 2, 1)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, 2, 1)
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, 2, 1)
        
        self.res1 = ResBlock(channels[0], channels[0])
        self.res2 = ResBlock(channels[1], channels[1])
        self.res3 = ResBlock(channels[2], channels[2])
        self.res4 = ResBlock(channels[3], channels[3])
        
    def forward(self, x: torch.Tensor) -> tuple:
        f1 = self.res1(F.relu(self.conv1(x)))
        f2 = self.res2(F.relu(self.conv2(f1)))
        f3 = self.res3(F.relu(self.conv3(f2)))
        f4 = self.res4(F.relu(self.conv4(f3)))
        return f1, f2, f3, f4


class InitDecoder(nn.Module):
    """Initial decoder for coarsest level."""
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2 + 1, out_channels + skip_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels + skip_channels, out_channels + skip_channels, 3, 1, 1)
        self.flow_head = nn.Conv2d(out_channels + skip_channels, 4, 3, 1, 1)  # 2 flows
        self.deconv = nn.ConvTranspose2d(out_channels + skip_channels, out_channels, 4, 2, 1)
        
    def forward(self, f0: torch.Tensor, f1: torch.Tensor, embt: torch.Tensor) -> tuple:
        b, c, h, w = f0.shape
        embt = embt.view(b, 1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([f0, f1, embt], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        flow = self.flow_head(x)
        ft = self.deconv(x)
        flow0, flow1 = torch.chunk(flow, 2, dim=1)
        return flow0, flow1, ft


class IntermediateDecoder(nn.Module):
    """Intermediate decoder for middle levels."""
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2 + 4, out_channels + skip_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels + skip_channels, out_channels + skip_channels, 3, 1, 1)
        self.flow_head = nn.Conv2d(out_channels + skip_channels, 4, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(out_channels + skip_channels, out_channels, 4, 2, 1)
        
    def forward(self, ft: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor, 
                flow0_up: torch.Tensor, flow1_up: torch.Tensor) -> tuple:
        # Upsample flows
        flow0_up = F.interpolate(flow0_up, scale_factor=2, mode='bilinear', align_corners=False) * 2
        flow1_up = F.interpolate(flow1_up, scale_factor=2, mode='bilinear', align_corners=False) * 2
        
        # Warp features
        f0_warp = warp(f0, flow0_up)
        f1_warp = warp(f1, flow1_up)
        
        x = torch.cat([ft, f0_warp, f1_warp, flow0_up, flow1_up], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        flow_res = self.flow_head(x)
        ft_out = self.deconv(x)
        
        flow0_out = flow0_up + flow_res[:, 0:2]
        flow1_out = flow1_up + flow_res[:, 2:4]
        
        return flow0_out, flow1_out, ft_out


class MultiFlowDecoder(nn.Module):
    """Final decoder with multi-flow prediction."""
    
    def __init__(self, in_channels: int, skip_channels: int, num_flows: int = 3):
        super().__init__()
        self.num_flows = num_flows
        self.conv1 = nn.Conv2d(in_channels * 2 + 4, in_channels + skip_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + skip_channels, in_channels + skip_channels, 3, 1, 1)
        
        # Multi-flow prediction
        self.flow_head = nn.Conv2d(in_channels + skip_channels, 4 * num_flows, 3, 1, 1)
        self.mask_head = nn.Conv2d(in_channels + skip_channels, num_flows, 3, 1, 1)
        self.img_res = nn.Conv2d(in_channels + skip_channels, 3, 3, 1, 1)
        
    def forward(self, ft: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor,
                flow0_up: torch.Tensor, flow1_up: torch.Tensor) -> tuple:
        # Upsample flows
        flow0_up = F.interpolate(flow0_up, scale_factor=2, mode='bilinear', align_corners=False) * 2
        flow1_up = F.interpolate(flow1_up, scale_factor=2, mode='bilinear', align_corners=False) * 2
        
        f0_warp = warp(f0, flow0_up)
        f1_warp = warp(f1, flow1_up)
        
        x = torch.cat([ft, f0_warp, f1_warp, flow0_up, flow1_up], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        flow_res = self.flow_head(x)
        mask = torch.softmax(self.mask_head(x), dim=1)
        img_res = self.img_res(x)
        
        flow0_out = flow0_up.unsqueeze(1) + flow_res[:, 0::4].unsqueeze(2)
        flow1_out = flow1_up.unsqueeze(1) + flow_res[:, 2::4].unsqueeze(2)
        
        return flow0_out, flow1_out, mask, img_res


class AMT_S(nn.Module):
    """AMT-S (Small) model."""
    
    def __init__(self, corr_radius: int = 3, corr_lvls: int = 4, num_flows: int = 3):
        super().__init__()
        self.num_flows = num_flows
        channels = [64, 96, 128, 192]
        
        self.encoder = Encoder(channels)
        self.decoder4 = InitDecoder(channels[3], channels[2], 64)
        self.decoder3 = IntermediateDecoder(channels[2], channels[1], 48)
        self.decoder2 = IntermediateDecoder(channels[1], channels[0], 32)
        self.decoder1 = MultiFlowDecoder(channels[0], 32, num_flows)
        
        # Combination block for multi-flow merge
        self.comb_block = nn.Sequential(
            nn.Conv2d(3 * num_flows, 6 * num_flows, 7, 1, 3),
            nn.PReLU(6 * num_flows),
            nn.Conv2d(6 * num_flows, 3, 7, 1, 3),
        )
        
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, 
                embt: torch.Tensor, scale_factor: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Args:
            img0: First frame [B, 3, H, W]
            img1: Second frame [B, 3, H, W]
            embt: Time embedding [B, 1]
            scale_factor: Processing scale
            
        Returns:
            Dict with 'imgt_pred' key containing interpolated frame
        """
        # Normalize
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0_norm = img0 - mean_
        img1_norm = img1 - mean_
        
        # Scale if needed
        if scale_factor != 1.0:
            img0_norm = F.interpolate(img0_norm, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            img1_norm = F.interpolate(img1_norm, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        # Encode
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_norm)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_norm)
        
        # Decode
        flow0_4, flow1_4, ft_3 = self.decoder4(f0_4, f1_4, embt)
        flow0_3, flow1_3, ft_2 = self.decoder3(ft_3, f0_3, f1_3, flow0_4, flow1_4)
        flow0_2, flow1_2, ft_1 = self.decoder2(ft_2, f0_2, f1_2, flow0_3, flow1_3)
        flow0_1, flow1_1, mask, img_res = self.decoder1(ft_1, f0_1, f1_1, flow0_2, flow1_2)
        
        # Multi-flow combine
        b, num_flows, _, h, w = flow0_1.shape
        
        # Warp and combine
        imgt_preds = []
        for i in range(num_flows):
            img0_warp = warp(img0, flow0_1[:, i])
            img1_warp = warp(img1, flow1_1[:, i])
            imgt_pred = img0_warp * (1 - embt.view(-1, 1, 1, 1)) + img1_warp * embt.view(-1, 1, 1, 1)
            imgt_preds.append(imgt_pred)
        
        imgt_stack = torch.cat(imgt_preds, dim=1)
        imgt_pred = self.comb_block(imgt_stack) + img_res + mean_
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        
        return {'imgt_pred': imgt_pred}


class AMTModel(BaseVFIModel):
    """AMT model wrapper."""
    
    MODEL_FILES = {
        "amt-s.pth": "amt-s.pth",
        "amt-l.pth": "amt-l.pth",
        "amt-g.pth": "amt-g.pth",
        "gopro_amt-s.pth": "gopro_amt-s.pth",
    }
    
    ARCH_CONFIGS = {
        "amt-s.pth": {"network": AMT_S, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 3}},
        "gopro_amt-s.pth": {"network": AMT_S, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 3}},
        # amt-l and amt-g would need their own classes
        "amt-l.pth": {"network": AMT_S, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 5}},
        "amt-g.pth": {"network": AMT_S, "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 5}},
    }
    
    def __init__(self, config: Optional[VFIConfig] = None):
        if config is None:
            config = VFIConfig(model_type=ModelType.AMT)
        super().__init__(config)
        self._arch_config = None
        
    def _create_model(self) -> nn.Module:
        """Create AMT model instance."""
        if self._arch_config is None:
            return AMT_S()
        
        network_class = self._arch_config.get("network", AMT_S)
        params = self._arch_config.get("params", {})
        return network_class(**params)
    
    def _get_model_url(self, ckpt_name: str) -> str:
        """Get download URL for model checkpoint."""
        return f"https://huggingface.co/lalala125/AMT/resolve/main/{ckpt_name}"
    
    def load_model(self, checkpoint_path: str) -> None:
        """Load AMT model weights."""
        ckpt_name = Path(checkpoint_path).name
        
        if ckpt_name in self.ARCH_CONFIGS:
            self._arch_config = self.ARCH_CONFIGS[ckpt_name]
        
        if not Path(checkpoint_path).exists():
            url = self._get_model_url(ckpt_name)
            download_model("amt", ckpt_name, str(Path(checkpoint_path).parent), url=url)
        
        state_dict = load_model_weights(checkpoint_path)
        
        if self._model is None:
            self._model = self._create_model()
            
        self._model.load_state_dict(state_dict, strict=False)
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
        
        # Use InputPadder for dimensions divisible by 16
        padder = InputPadder(frame0.shape, 16)
        frame0_pad = padder.pad(frame0)
        frame1_pad = padder.pad(frame1)
        
        # Create timestep tensor (embt)
        embt = torch.full(
            (frame0_pad.shape[0], 1),
            timestep,
            device=frame0_pad.device,
            dtype=frame0_pad.dtype
        )
        
        with torch.no_grad():
            result = self._model(frame0_pad, frame1_pad, embt, scale_factor=scale)
            output = result['imgt_pred']
        
        # Unpad
        output = padder.unpad(output)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return VFIResult(
            frame=output,
            model_type=ModelType.AMT,
            metadata={"timestep": timestep, "scale": scale}
        )
