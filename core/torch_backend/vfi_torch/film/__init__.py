"""FILM (Frame Interpolation for Large Motion) 模型实现。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path

from ..base import BaseVFIModel, VFIConfig, VFIResult, ModelType
from ..utils import load_model_weights, download_model


class FeatureExtractor(nn.Module):
    """Feature extraction network for FILM."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, 3, 2, 1)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, 3, 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class PyramidFlowEstimator(nn.Module):
    """Pyramid flow estimation network."""
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels // 2, 2, 3, 1, 1)  # flow output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Fusion(nn.Module):
    """Feature fusion network."""
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([x1, x2], dim=1))


class Interpolator(nn.Module):
    """Final interpolation network."""
    
    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1)
        
    def forward(self, features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # Warp features using flow
        b, c, h, w = features.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=features.device),
            torch.arange(w, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        
        # Apply flow
        warped_grid = grid + flow.permute(0, 2, 3, 1)
        warped_grid = warped_grid * 2 / torch.tensor([w - 1, h - 1], device=features.device) - 1
        
        warped_features = F.grid_sample(
            features, warped_grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        x = torch.cat([features, warped_features], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class FILMNet(nn.Module):
    """FILM network for frame interpolation."""
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.flow_estimator = PyramidFlowEstimator()
        self.fusion = Fusion()
        self.interpolator = Interpolator()
        
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
        # Extract features
        feat0 = self.feature_extractor(img0)
        feat1 = self.feature_extractor(img1)
        
        # Fuse features
        fused = self.fusion(feat0, feat1)
        
        # Estimate flow
        flow = self.flow_estimator(fused)
        
        # Scale flow by timestep
        if timestep.dim() == 2:
            timestep = timestep.view(-1, 1, 1, 1)
        flow = flow * timestep
        
        # Interpolate
        output = self.interpolator(feat0, flow)
        
        return output


class FILMModel(BaseVFIModel):
    """FILM model wrapper."""
    
    MODEL_FILES = {
        "film_net_fp32.pt": "film_net_fp32.pt",
    }
    
    VERSION_MAP = {
        "film_net_fp32.pt": "1.0",
    }
    
    def __init__(self, config: Optional[VFIConfig] = None):
        if config is None:
            config = VFIConfig(model_type=ModelType.FILM)
        super().__init__(config)
        
    def _create_model(self) -> nn.Module:
        """Create FILM model instance."""
        return FILMNet()
    
    def _get_model_url(self, ckpt_name: str) -> str:
        """Get download URL for model checkpoint."""
        from utils import BASE_MODEL_URLS
        base_url = BASE_MODEL_URLS.get("film", "")
        return f"{base_url}{ckpt_name}"
    
    def load_model(self, checkpoint_path: str) -> None:
        """Load FILM model weights."""
        if not Path(checkpoint_path).exists():
            # Try to download
            ckpt_name = Path(checkpoint_path).name
            url = self._get_model_url(ckpt_name)
            download_model("film", ckpt_name, str(Path(checkpoint_path).parent), url=url)
        
        loaded = load_model_weights(checkpoint_path)
        
        # Check if loaded object is a TorchScript module
        if isinstance(loaded, torch.jit.ScriptModule):
            # Use the TorchScript model directly
            self._model = loaded
            self._model.to(self.device)
            self._model.eval()
        else:
            # It's a state dict, load into model
            if self._model is None:
                self._model = self._create_model()
                
            self._model.load_state_dict(loaded)
            self._model.to(self.device)
            self._model.eval()
        
    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs
    ) -> VFIResult:
        """
        Interpolate a single frame pair.
        
        Args:
            frame0: First frame [B, C, H, W] or [C, H, W]
            frame1: Second frame [B, C, H, W] or [C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)
            
        Returns:
            VFIResult with interpolated frame
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure batch dimension
        squeeze_output = False
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
            frame1 = frame1.unsqueeze(0)
            squeeze_output = True
        
        # Create timestep tensor
        timestep_tensor = torch.full(
            (frame0.shape[0], 1), 
            timestep, 
            device=frame0.device, 
            dtype=frame0.dtype
        )
        
        with torch.no_grad():
            output = self._model(frame0, frame1, timestep_tensor)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return VFIResult(
            frame=output,
            model_type=ModelType.FILM,
            metadata={"timestep": timestep}
        )
    
    def interpolate_multi(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        num_frames: int = 1,
        **kwargs
    ) -> VFIResult:
        """
        Interpolate multiple frames between two frames.
        
        Args:
            frame0: First frame
            frame1: Second frame
            num_frames: Number of intermediate frames to generate
            
        Returns:
            VFIResult with all interpolated frames
        """
        frames = []
        timesteps = torch.linspace(1, num_frames, num_frames) / (num_frames + 1)
        
        for t in timesteps:
            result = self.interpolate(frame0, frame1, t.item())
            frames.append(result.frame)
        
        return VFIResult(
            frame=torch.stack(frames),
            model_type=ModelType.FILM,
            metadata={"num_frames": num_frames, "timesteps": timesteps.tolist()}
        )
