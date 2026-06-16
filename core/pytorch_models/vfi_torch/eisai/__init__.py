"""EISAI (Efficient Interpolation with Softsplat and DTM) 模型实现。

参考: https://github.com/ShuhongChen/eisai-anime-interpolator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from pathlib import Path
from argparse import Namespace

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import get_device, load_model_weights, download_model


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp image using optical flow."""
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1).to(img)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W).to(img)
    grid = torch.cat([xx, yy], 1)
    flow_ = torch.cat([
        flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
        flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)
    ], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(img, grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


class ResBlock(nn.Module):
    """Residual block with PReLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.net(x)


class Synthesizer(nn.Module):
    """Synthesizer network for final frame generation."""
    def __init__(self, channels_image: int = 12, channels_flow: int = 4, 
                 channels_mask: int = 2, channels_feature: int = 32):
        super().__init__()
        ch = channels_image + channels_flow // 2 + channels_mask + channels_feature
        self.net = nn.Sequential(
            nn.Conv2d(ch + 3, 64, 1),
            ResBlock(64),
            nn.PReLU(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            ResBlock(32),
            nn.PReLU(32),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            ResBlock(16),
            nn.PReLU(16),
            nn.Conv2d(16, 3, 3, 1, 1),
        )
    
    def forward(self, images: List[torch.Tensor], flows: List[torch.Tensor],
                masks: List[torch.Tensor], features: List[torch.Tensor]) -> torch.Tensor:
        # Average of input images
        avg_img = (images[0] + images[1]) / 2
        
        # Concatenate all inputs
        x = torch.cat([
            avg_img, images[0], images[1],
            flows[0].norm(dim=1, keepdim=True),
            flows[1].norm(dim=1, keepdim=True),
            masks[0], masks[1],
            features[0]
        ], dim=1)
        
        residual = self.net(x)
        return torch.sigmoid(avg_img[:, :3] + 0.5 * residual)


class HalfWarper(nn.Module):
    """Half-way warping module."""
    def __init__(self):
        super().__init__()
    
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, 
                flow0: torch.Tensor, flow1: torch.Tensor, t: float) -> Dict[str, torch.Tensor]:
        """Warp images to intermediate time step."""
        # Forward warp flows
        flow0_t = (1 - t) * flow0
        flow1_t = t * flow1
        
        # Simple backward warp for intermediate frame
        warped0 = warp(img0, -flow0_t)
        warped1 = warp(img1, flow1_t)
        
        # Blend
        mask0 = torch.ones_like(warped0[:, :1])
        mask1 = torch.ones_like(warped1[:, :1])
        
        return {
            'images': [warped0, warped1],
            'masks': [mask0, mask1],
        }


class SoftsplatLite(nn.Module):
    """Lightweight softsplat module."""
    def __init__(self):
        super().__init__()
        self.half_warper = HalfWarper()
        self.synthesizer = Synthesizer()
    
    def forward(self, img0: torch.Tensor, img1: torch.Tensor,
                flow0: torch.Tensor, flow1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """Forward pass."""
        # Warp to intermediate
        warped = self.half_warper(img0, img1, flow0, flow1, t)
        
        # Synthesize
        output = self.synthesizer(
            [img0, img1],
            [flow0, flow1],
            warped['masks'],
            [torch.zeros(img0.shape[0], 32, img0.shape[2], img0.shape[3], device=img0.device)]
        )
        
        return output


class DTM(nn.Module):
    """Distance Transform Module."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            nn.Conv2d(16, 3, 3, 1, 1),
        )
    
    def forward(self, base_output: torch.Tensor) -> torch.Tensor:
        """Refine output."""
        return torch.sigmoid(base_output[:, :3] + self.net(base_output[:, :6]))


class BasicEncoder(nn.Module):
    """Basic feature encoder from RAFT."""
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, output_dim, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class FlowHead(nn.Module):
    """Flow prediction head."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    """Convolutional GRU cell."""
    def __init__(self, hidden_dim: int = 128, input_dim: int = 256):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class RAFT(nn.Module):
    """RAFT optical flow network (simplified)."""
    def __init__(self):
        super().__init__()
        self.fnet = BasicEncoder(256)
        self.cnet = BasicEncoder(128)
        self.update_block = ConvGRU()
        self.flow_head = FlowHead(128, 256)
        self.hidden_dim = 128
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor, 
                iters: int = 12) -> torch.Tensor:
        """Estimate optical flow."""
        # Feature extraction
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        
        # Context features
        cmap = self.cnet(image1)
        h, _ = cmap[:, :self.hidden_dim], cmap[:, self.hidden_dim:]
        
        # Initialize flow
        B, _, H, W = fmap1.shape
        coords = torch.meshgrid(
            torch.arange(H, device=image1.device),
            torch.arange(W, device=image1.device),
            indexing='ij'
        )
        coords = torch.stack(coords[::-1], dim=0).float()
        coords = coords[None].expand(B, -1, -1, -1)
        
        flow = torch.zeros(B, 2, H, W, device=image1.device)
        
        for _ in range(iters):
            # Correlation (simplified)
            # In real RAFT, this uses correlation pyramid
            warped = warp(fmap2, flow)
            corr = torch.cat([fmap1, warped], dim=1)
            
            # Update
            h = self.update_block(h, corr)
            delta_flow = self.flow_head(h)
            flow = flow + delta_flow
        
        return flow


class EISAI(nn.Module):
    """EISAI model combining RAFT, SoftsplatLite, and DTM."""
    def __init__(self):
        super().__init__()
        self.raft = RAFT()
        self.ssl = SoftsplatLite()
        self.dtm = DTM()
    
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """Forward pass."""
        with torch.no_grad():
            flow0 = self.raft(img0, img1)
            flow1 = self.raft(img1, img0)
        
        out_ssl = self.ssl(img0, img1, flow0, flow1, t)
        out_dtm = self.dtm(out_ssl)
        
        return out_dtm


class EISAIModel(PyTorchVFIModel):
    """EISAI model wrapper for VFI-gui."""
    
    MODEL_NAME = "eisai"
    SUPPORTED_VERSIONS = ["default"]
    DEFAULT_VERSION = "default"
    MIN_INPUT_FRAMES = 2
    
    # Model files required
    MODEL_FILES = {
        "raft": "eisai_anime_interp_full.ckpt",
        "ssl": "eisai_ssl.pt",
        "dtm": "eisai_dtm.pt",
    }
    
    def __init__(self, config: Optional[VFIConfig] = None):
        if config is None:
            config = VFIConfig(model_type=ModelType.EISAI)
        super().__init__(config)
        self._model_files_loaded = False
    
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """Load EISAI model weights.
        
        Note: EISAI requires 3 model files. checkpoint_path should be the directory
        containing all three files, or we use the models/eisai/ directory.
        """
        if self._model is not None:
            return
        
        # Create model
        self._model = EISAI()
        
        # Load weights if checkpoint files exist
        checkpoint_dir = Path(checkpoint_path).parent if Path(checkpoint_path).exists() else Path(checkpoint_path)
        
        raft_path = checkpoint_dir / self.MODEL_FILES["raft"]
        ssl_path = checkpoint_dir / self.MODEL_FILES["ssl"]
        dtm_path = checkpoint_dir / self.MODEL_FILES["dtm"]
        
        if raft_path.exists():
            try:
                state_dict = torch.load(raft_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                # Filter for raft-related keys
                raft_state = {k.replace('module.flownet.', ''): v 
                             for k, v in state_dict.items() 
                             if 'flownet' in k or 'raft' in k.lower()}
                if raft_state:
                    self._model.raft.load_state_dict(raft_state, strict=False)
            except Exception:
                pass
        
        if ssl_path.exists():
            try:
                state_dict = torch.load(ssl_path, map_location=self.device, weights_only=False)
                self._model.ssl.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
        
        if dtm_path.exists():
            try:
                state_dict = torch.load(dtm_path, map_location=self.device, weights_only=False)
                self._model.dtm.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
        
        self._model = self._model.to(self.device)
        self._model.eval()
        
        self.to_device()
    
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs
    ) -> torch.Tensor:
        """Interpolate a single frame between two input frames."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        squeeze_output = False
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
            frame1 = frame1.unsqueeze(0)
            squeeze_output = True
        
        frame0, frame1 = self.prepare_frames(frame0, frame1)
        
        assert self._model is not None
        with torch.no_grad():
            output = self._model(frame0, frame1, t=timestep)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
