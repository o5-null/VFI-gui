"""XVFI (eXtreme Video Frame Interpolation) 模型实现。

参考: https://github.com/JihyongOh/XVFI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import get_device, load_model_weights, download_model


class ResBlock2D_3D(nn.Module):
    """Residual block for 2D/3D convolutions."""
    def __init__(self, nf: int):
        super().__init__()
        self.conv3x3_1 = nn.Conv3d(nf, nf, [1, 3, 3], 1, [0, 1, 1])
        self.conv3x3_2 = nn.Conv3d(nf, nf, [1, 3, 3], 1, [0, 1, 1])
        self.lrelu = nn.ReLU()

    def forward(self, x):
        out = self.conv3x3_2(self.lrelu(self.conv3x3_1(x)))
        return x + out


class RResBlock2D_3D(nn.Module):
    """Recursive residual block."""
    def __init__(self, nf: int, T_reduce_flag: bool = False):
        super().__init__()
        self.T_reduce_flag = T_reduce_flag
        self.resblock1 = ResBlock2D_3D(nf)
        self.resblock2 = ResBlock2D_3D(nf)
        if T_reduce_flag:
            self.reduceT_conv = nn.Conv3d(nf, nf, [3, 1, 1], 1, [0, 0, 0])

    def forward(self, x):
        out = self.resblock1(x)
        out = self.resblock2(out)
        if self.T_reduce_flag:
            return self.reduceT_conv(out + x)
        return out + x


class RefineUNet(nn.Module):
    """Refinement U-Net."""
    def __init__(self, nf: int, scale: int, img_ch: int = 3):
        super().__init__()
        self.scale = scale
        self.nf = nf
        self.conv1 = nn.Conv2d(nf, nf, [3, 3], 1, [1, 1])
        self.conv2 = nn.Conv2d(nf, nf, [3, 3], 1, [1, 1])
        self.lrelu = nn.ReLU()
        self.NN = nn.UpsamplingNearest2d(scale_factor=2)

        self.enc1 = nn.Conv2d((4 * nf) // scale // scale + 4 * img_ch + 4, nf, [4, 4], 2, [1, 1])
        self.enc2 = nn.Conv2d(nf, 2 * nf, [4, 4], 2, [1, 1])
        self.enc3 = nn.Conv2d(2 * nf, 4 * nf, [4, 4], 2, [1, 1])
        self.dec0 = nn.Conv2d(4 * nf, 4 * nf, [3, 3], 1, [1, 1])
        self.dec1 = nn.Conv2d(4 * nf + 2 * nf, 2 * nf, [3, 3], 1, [1, 1])
        self.dec2 = nn.Conv2d(2 * nf + nf, nf, [3, 3], 1, [1, 1])
        self.dec3 = nn.Conv2d(nf, 1 + img_ch, [3, 3], 1, [1, 1])

    def forward(self, concat):
        enc1 = self.lrelu(self.enc1(concat))
        enc2 = self.lrelu(self.enc2(enc1))
        out = self.lrelu(self.enc3(enc2))
        out = self.lrelu(self.dec0(out))
        out = self.NN(out)
        out = torch.cat((out, enc2), dim=1)
        out = self.lrelu(self.dec1(out))
        out = self.NN(out)
        out = torch.cat((out, enc1), dim=1)
        out = self.lrelu(self.dec2(out))
        out = self.NN(out)
        out = self.dec3(out)
        return out


class VFInet(nn.Module):
    """VFI network module."""
    def __init__(self, nf: int, scale: int, S_trn: int, S_tst: int, img_ch: int = 3):
        super().__init__()
        self.nf = nf
        self.scale = scale
        self.S_trn = S_trn
        self.S_tst = S_tst
        self.img_ch = img_ch
        
        self.lrelu = nn.ReLU()
        self.channel_converter = nn.Sequential(
            nn.Conv3d(img_ch, nf, [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            nn.ReLU()
        )
        
        self.rec_ext_ds_module = [self.channel_converter]
        self.rec_ext_ds = nn.Conv3d(nf, nf, [1, 3, 3], [1, 2, 2], [0, 1, 1])
        import numpy as np
        for _ in range(int(np.log2(scale))):
            self.rec_ext_ds_module.append(self.rec_ext_ds)
            self.rec_ext_ds_module.append(nn.ReLU())
        self.rec_ext_ds_module.append(nn.Conv3d(nf, nf, [1, 3, 3], 1, [0, 1, 1]))
        self.rec_ext_ds_module.append(RResBlock2D_3D(nf, T_reduce_flag=False))
        self.rec_ext_ds_module = nn.Sequential(*self.rec_ext_ds_module)
        
        self.rec_ctx_ds = nn.Conv3d(nf, nf, [1, 3, 3], [1, 2, 2], [0, 1, 1])
        
        # Flow estimation modules
        self.conv_flow_bottom = nn.Sequential(
            nn.Conv2d(2 * nf, 2 * nf, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(2 * nf, 4 * nf, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(4 * nf, 2 * nf, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(2 * nf, nf, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(nf, 6, [3, 3], 1, [1, 1]),
        )
        
        self.conv_flow1 = nn.Conv2d(2 * nf, nf, [3, 3], 1, [1, 1])
        
        self.conv_flow2 = nn.Sequential(
            nn.Conv2d(2 * nf + 4, 2 * nf, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(2 * nf, 4 * nf, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(4 * nf, 2 * nf, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(2 * nf, nf, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(nf, 6, [3, 3], 1, [1, 1]),
        )
        
        self.conv_flow3 = nn.Sequential(
            nn.Conv2d(4 + nf * 4, nf, [1, 1], 1, [0, 0]),
            nn.ReLU(),
            nn.Conv2d(nf, 2 * nf, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(2 * nf, 4 * nf, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(4 * nf, 2 * nf, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(2 * nf, nf, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(nf, 4, [3, 3], 1, [1, 1]),
        )
        
        self.refine_unet = RefineUNet(nf, scale, img_ch)

    def bwarp(self, x, flo):
        B, C, H, W = x.size()
        device = x.device
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W).to(device)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W).to(device)
        grid = torch.cat((xx, yy), 1).float()
        vgrid = torch.autograd.Variable(grid) + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        return output * mask

    def forward(self, x, t_value, is_training=True):
        B, C, T, H, W = x.size()
        t_value = t_value.view(B, 1, 1, 1)
        
        flow_l = None
        feat_x = self.rec_ext_ds_module(x)
        feat_x_list = [feat_x]
        
        lowest_depth_level = self.S_trn if is_training else self.S_tst
        for level in range(1, lowest_depth_level + 1):
            feat_x = self.rec_ctx_ds(feat_x)
            feat_x_list.append(feat_x)
        
        # Testing mode
        for level in range(lowest_depth_level, 0, -1):
            l = 2 ** level
            x_l = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
            if level != 0:
                x_l = F.interpolate(x_l, scale_factor=(1.0 / l, 1.0 / l), mode='bicubic', align_corners=False)
            x_l = x_l.view(B, T, C, H // l, W // l).permute(0, 2, 1, 3, 4)
            
            B_l, C_l, T_l, H_l, W_l = x_l.size()
            feat0_l = feat_x_list[level][:, :, 0, :, :]
            feat1_l = feat_x_list[level][:, :, 1, :, :]
            
            if flow_l is None:
                flow_l_tmp = self.conv_flow_bottom(torch.cat((feat0_l, feat1_l), dim=1))
                flow_l = flow_l_tmp[:, :4, :, :]
            else:
                up_flow_l_prev = 2.0 * F.interpolate(flow_l.detach(), scale_factor=(2, 2), mode='bilinear', align_corners=False)
                warped_feat1_l = self.bwarp(feat1_l, up_flow_l_prev[:, :2, :, :])
                warped_feat0_l = self.bwarp(feat0_l, up_flow_l_prev[:, 2:4, :, :])
                flow_l_tmp = self.conv_flow2(torch.cat([
                    self.conv_flow1(torch.cat([feat0_l, warped_feat1_l], dim=1)),
                    self.conv_flow1(torch.cat([feat1_l, warped_feat0_l], dim=1)),
                    up_flow_l_prev
                ], dim=1))
                flow_l = flow_l_tmp[:, :4, :, :] + up_flow_l_prev
            
            if level != 1:
                continue
            
            # Final level
            flow_01_l = flow_l[:, :2, :, :]
            flow_10_l = flow_l[:, 2:4, :, :]
            z_01_l = torch.sigmoid(flow_l_tmp[:, 4:5, :, :])
            z_10_l = torch.sigmoid(flow_l_tmp[:, 5:6, :, :])
            
            # CFR
            flow_forward = z_01_l * t_value * flow_01_l
            flow_backward = z_10_l * (1 - t_value) * flow_10_l
            
            flow_t0_l = -(1 - t_value) * flow_forward + t_value * flow_backward
            flow_t1_l = (1 - t_value) * flow_forward - t_value * flow_backward
            
            warped0_l = self.bwarp(feat_x_list[0][:, :, 0, :, :], flow_t0_l)
            warped1_l = self.bwarp(feat_x_list[0][:, :, 1, :, :], flow_t1_l)
            
            flow_refine_l = torch.cat([feat_x_list[0][:, :, 0, :, :], warped0_l, warped1_l, feat_x_list[0][:, :, 1, :, :], flow_t0_l, flow_t1_l], dim=1)
            flow_refine_l = self.conv_flow3(flow_refine_l) + torch.cat([flow_t0_l, flow_t1_l], dim=1)
            flow_t0_l = flow_refine_l[:, :2, :, :]
            flow_t1_l = flow_refine_l[:, 2:4, :, :]
            
            flow_t0_l = self.scale * F.interpolate(flow_t0_l, scale_factor=(self.scale, self.scale), mode='bilinear', align_corners=False)
            flow_t1_l = self.scale * F.interpolate(flow_t1_l, scale_factor=(self.scale, self.scale), mode='bilinear', align_corners=False)
            
            x_0 = x[:, :, 0, :, :]
            x_1 = x[:, :, 1, :, :]
            warped_img0_l = self.bwarp(x_0, flow_t0_l)
            warped_img1_l = self.bwarp(x_1, flow_t1_l)
            
            refine_out = self.refine_unet(torch.cat([
                F.pixel_shuffle(torch.cat([feat_x_list[0][:, :, 0, :, :], feat_x_list[0][:, :, 1, :, :], warped0_l, warped1_l], dim=1), self.scale),
                x_0, x_1, warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l
            ], dim=1))
            
            occ_0_l = torch.sigmoid(refine_out[:, 0:1, :, :])
            occ_1_l = 1 - occ_0_l
            
            out_l = (1 - t_value) * occ_0_l * warped_img0_l + t_value * occ_1_l * warped_img1_l
            out_l = out_l / ((1 - t_value) * occ_0_l + t_value * occ_1_l) + refine_out[:, 1:4, :, :]
            
            return out_l
        
        return torch.zeros(B, 3, H, W, device=x.device)


class XVFInet(nn.Module):
    """XVFI main network."""
    def __init__(self, nf: int = 64, scale: int = 4, S_trn: int = 3, S_tst: int = 5, img_ch: int = 3):
        super().__init__()
        self.nf = nf
        self.scale = scale
        self.S_trn = S_trn
        self.S_tst = S_tst
        self.vfinet = VFInet(nf, scale, S_trn, S_tst, img_ch)
        self.lrelu = nn.ReLU()
        self.in_channels = img_ch
        self.channel_converter = nn.Sequential(
            nn.Conv3d(img_ch, nf, [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            nn.ReLU()
        )

    def forward(self, x, t_value, is_training=False):
        return self.vfinet(x, t_value, is_training=is_training)


class XVFIModel(PyTorchVFIModel):
    """XVFI model wrapper for VFI-gui."""
    
    MODEL_NAME = "xvfi"
    SUPPORTED_VERSIONS = ["x4k1000fps", "vimeo"]
    DEFAULT_VERSION = "x4k1000fps"
    MIN_INPUT_FRAMES = 2
    
    # Config for different checkpoints
    CKPT_CONFIGS = {
        "XVFInet_X4K1000FPS_exp1_latest.pt": {
            "module_scale_factor": 4,
            "S_trn": 3,
            "S_tst": 5,
        },
        "XVFInet_Vimeo_exp1_latest.pt": {
            "module_scale_factor": 2,
            "S_trn": 1,
            "S_tst": 1,
        },
        "xvfi.pth": {  # Default checkpoint
            "module_scale_factor": 4,
            "S_trn": 3,
            "S_tst": 5,
        },
    }
    
    def __init__(self, config: Optional[VFIConfig] = None):
        if config is None:
            config = VFIConfig(model_type=ModelType.XVFI)
        super().__init__(config)
        self._model_config = None
    
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """Load XVFI model weights."""
        if self._model is not None:
            return
        
        ckpt_name = Path(checkpoint_path).name
        config = self.CKPT_CONFIGS.get(ckpt_name, self.CKPT_CONFIGS["xvfi.pth"])
        self._model_config = config
        
        # Create model
        self._model = XVFInet(
            nf=64,
            scale=config["module_scale_factor"],
            S_trn=config["S_trn"],
            S_tst=config["S_tst"],
            img_ch=3,
        )
        
        # Load weights
        if Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'state_dict_Model' in state_dict:
                state_dict = state_dict['state_dict_Model']
            self._model.load_state_dict(state_dict, strict=False)
        
        self._model = self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True
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
        
        # Calculate padding
        if self._model_config:
            divide = 2 ** self._model_config["S_tst"] * self._model_config["module_scale_factor"] * 4
        else:
            divide = 128
        
        _, _, H, W = frame0.shape
        H_padding = (divide - H % divide) % divide
        W_padding = (divide - W % divide) % divide
        
        if H_padding != 0 or W_padding != 0:
            frame0 = F.pad(frame0, (0, W_padding, 0, H_padding), "constant")
            frame1 = F.pad(frame1, (0, W_padding, 0, H_padding), "constant")
        
        # Stack frames
        x = torch.stack([frame0, frame1], dim=0)
        x = x.permute(1, 2, 0, 3, 4)  # [T, B, C, H, W] -> [B, C, T, H, W]
        
        # Create timestep tensor
        t_tensor = torch.tensor([timestep], device=self.device, dtype=frame0.dtype).unsqueeze(1)
        
        assert self._model is not None
        with torch.no_grad():
            output = self._model(x, t_tensor, is_training=False)
        
        # Remove padding
        output = output[:, :, :H, :W]
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
