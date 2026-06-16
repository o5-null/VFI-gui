"""
ATM-VFI Network (base variant).

Reference: https://github.com/HP-NTNU-VFI/ATM
"""

import einops
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .flow_warp import flow_warp
from .attention import ATMFormer
from .attention import RefineBottleneck as SwinTransformer


def upsample_flow(flow, upsample_factor=2, mode="bilinear"):
    if mode == "nearest":
        return F.interpolate(flow, scale_factor=upsample_factor, mode=mode) * upsample_factor
    else:
        return F.interpolate(flow, scale_factor=upsample_factor, mode=mode, align_corners=True) * upsample_factor


def conv(in_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_dim),
    )


def deconv(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_dim),
    )


class CrossScaleFeatureFusion(nn.Module):
    def __init__(self, in_dims=(32, 64, 128, 256), fused_dim=None):
        super().__init__()
        layers = []
        for i in range(len(in_dims) - 1):
            for j in range(2**i):
                layers.append(
                    nn.Conv2d(in_dims[-2 - i], in_dims[-2 - i], kernel_size=3, stride=2 ** (i + 1), padding=1 + j, dilation=1 + j, bias=True)
                )
        self.layers = nn.ModuleList(layers)
        concat_dim = sum([2 ** (len(in_dims) - 2 - i) * in_dims[i] for i in range(len(in_dims) - 1)]) + in_dims[-1]
        fused_dim = fused_dim or concat_dim
        self.proj = nn.Conv2d(concat_dim, fused_dim, 1, 1)
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, xs):
        ys = []
        k = 0
        for i in range(len(xs) - 1):
            for _ in range(2**i):
                ys.append(self.layers[k](xs[-2 - i]))
                k += 1
        ys.append(xs[-1])
        x = self.proj(torch.cat(ys, dim=1))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Network(nn.Module):
    """ATM-VFI base network with 4 pyramid levels and hidden dims [24, 48, 96, 192]."""

    def __init__(self, global_motion=True, ensemble_global_motion=False):
        super().__init__()
        self.pyramid_level = 4
        self.hidden_dims = [24, 48, 96, 192]
        self.global_motion = global_motion
        self.ensemble_global_motion = ensemble_global_motion

        # Pyramid feature extraction
        self.feat_extracts = nn.ModuleList([])
        for i in range(self.pyramid_level):
            if i == 0:
                self.feat_extracts.append(
                    nn.Sequential(conv(3, self.hidden_dims[i], kernel_size=3, stride=1, padding=1), conv(self.hidden_dims[i], self.hidden_dims[i], kernel_size=3, stride=1, padding=1))
                )
            else:
                self.feat_extracts.append(
                    nn.Sequential(conv(self.hidden_dims[i - 1], self.hidden_dims[i], kernel_size=3, stride=2, padding=1), conv(self.hidden_dims[i], self.hidden_dims[i], kernel_size=3, stride=1, padding=1))
                )

        # Local motion
        concat_dim = self.hidden_dims[-1] + self.hidden_dims[-2] + 2 * self.hidden_dims[-3]
        fused_dim = concat_dim
        self.cross_scale_feature_fusion = CrossScaleFeatureFusion(in_dims=self.hidden_dims[1:], fused_dim=fused_dim)

        self.local_motion_args = {"window_size": 8, "num_heads": 8, "patch_size": 1, "dim": fused_dim, "enhance_window": 8}

        self.feat_enhance_transformer = nn.ModuleList([
            SwinTransformer(dim=self.local_motion_args["dim"], window_size=self.local_motion_args["enhance_window"], shift_size=0, patch_size=self.local_motion_args["patch_size"], num_heads=self.local_motion_args["num_heads"]),
            SwinTransformer(dim=self.local_motion_args["dim"], window_size=self.local_motion_args["enhance_window"], shift_size=self.local_motion_args["enhance_window"] // 2, patch_size=self.local_motion_args["patch_size"], num_heads=self.local_motion_args["num_heads"]),
        ])

        self.local_motion_atmformer = nn.ModuleList([
            ATMFormer(dim=self.local_motion_args["dim"], window_size=self.local_motion_args["window_size"], shift_size=0, patch_size=self.local_motion_args["patch_size"], num_heads=self.local_motion_args["num_heads"]),
            ATMFormer(dim=self.local_motion_args["dim"], window_size=self.local_motion_args["window_size"], shift_size=self.local_motion_args["window_size"] // 2, patch_size=self.local_motion_args["patch_size"], num_heads=self.local_motion_args["num_heads"]),
        ])

        self.fused_dim = fused_dim * 2
        self.motion_out_dim = 5
        motion_mlp_hidden_dim = int(self.fused_dim * 0.75)
        self.local_motion_mlp = nn.Sequential(
            conv(self.fused_dim + self.local_motion_args["num_heads"], motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
            conv(motion_mlp_hidden_dim, motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(motion_mlp_hidden_dim, self.motion_out_dim, kernel_size=1, stride=1, padding=0),
        )

        # Global motion
        last_feat_dim = self.hidden_dims[-1] + 96
        self.last_feat_extract = nn.Sequential(
            conv(self.hidden_dims[-1], last_feat_dim, kernel_size=3, stride=2, padding=1),
            conv(last_feat_dim, last_feat_dim, kernel_size=3, stride=1, padding=1),
        )

        concat_dim = last_feat_dim + self.hidden_dims[-1] + 2 * self.hidden_dims[-2]
        self.global_feature_fusion = CrossScaleFeatureFusion(in_dims=[self.hidden_dims[-2], self.hidden_dims[-1], last_feat_dim], fused_dim=concat_dim)

        self.global_motion_args = {"window_size": 12, "num_heads": 8, "patch_size": 1, "dim": concat_dim}

        self.global_motion_atmformer = nn.ModuleList([
            ATMFormer(dim=self.global_motion_args["dim"], window_size=self.global_motion_args["window_size"], shift_size=0, patch_size=self.global_motion_args["patch_size"], num_heads=self.global_motion_args["num_heads"]),
            ATMFormer(dim=self.global_motion_args["dim"], window_size=self.global_motion_args["window_size"], shift_size=self.global_motion_args["window_size"] // 2, patch_size=self.global_motion_args["patch_size"], num_heads=self.global_motion_args["num_heads"]),
        ])

        motion_mlp_hidden_dim = 768
        self.global_motion_mlp = nn.Sequential(
            conv(concat_dim * 2 + self.global_motion_args["num_heads"], motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
            conv(motion_mlp_hidden_dim, motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(motion_mlp_hidden_dim, self.motion_out_dim, kernel_size=1, stride=1, padding=0),
        )

        # Upsample pyramid + residual refinement
        self.fused_dim1 = self.fused_dim // 2
        self.fused_dim2 = self.fused_dim // 4
        self.fused_dim3 = self.fused_dim // 8
        self.fused_dims = [self.fused_dim1, self.fused_dim2, self.fused_dim3, 2 * self.fused_dim1]
        deconv_args = {"kernel_size": 2, "stride": 2, "padding": 0}
        self.upsample_pyramid = nn.ModuleList([
            nn.Sequential(
                deconv(self.fused_dim + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, **deconv_args),
                conv(self.fused_dim1 + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.fused_dim1 + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                nn.PReLU(self.fused_dim1 + self.motion_out_dim),
                deconv(self.fused_dim1 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, **deconv_args),
                conv(self.fused_dim2 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.fused_dim2 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                nn.PReLU(self.fused_dim2 + self.motion_out_dim),
                deconv(self.fused_dim2 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, **deconv_args),
                conv(self.fused_dim3 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.fused_dim3 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
            ),
        ])

        # Residual refinement network
        in_chan = self.fused_dim3 + self.motion_out_dim + 15
        hidden_dim = 64
        self.proj_ref = conv(in_chan, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Sequential(conv(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1))
        self.down2 = nn.Sequential(
            conv(self.fused_dim2 + hidden_dim, 2 * hidden_dim, kernel_size=3, stride=2, padding=1),
            conv(2 * hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.down3 = nn.Sequential(
            conv(self.fused_dim1 + 2 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=2, padding=1),
            conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
            conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.up1 = nn.Sequential(
            deconv(4 * hidden_dim, 2 * hidden_dim, kernel_size=2, stride=2, padding=0),
            conv(2 * hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.up2 = nn.Sequential(
            deconv(4 * hidden_dim, 2 * hidden_dim, kernel_size=2, stride=2, padding=0),
            conv(2 * hidden_dim, 1 * hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.up3 = nn.Sequential(
            deconv(2 * hidden_dim, 1 * hidden_dim, kernel_size=2, stride=2, padding=0),
        )
        self.refine_head = nn.Sequential(
            conv(2 * hidden_dim, 1 * hidden_dim, kernel_size=3, stride=1, padding=1),
            conv(1 * hidden_dim, 3, kernel_size=3, stride=1, padding=1),
        )

    def __set_local_window_size__(self, window_size):
        self.local_motion_args["window_size"] = window_size
        self.local_motion_atmformer[0]._set_window_size_(window_size, 0)
        self.local_motion_atmformer[1]._set_window_size_(window_size, window_size // 2)

    def __set_global_window_size__(self, window_size):
        self.global_motion_args["window_size"] = window_size
        self.global_motion_atmformer[0]._set_window_size_(window_size, 0)
        self.global_motion_atmformer[1]._set_window_size_(window_size, window_size // 2)

    def forward(self, im0, im1):
        if not self.ensemble_global_motion:
            return self._forward_normal(im0, im1)
        else:
            return self._forward_global_ensemble(im0, im1)

    def _shared_feat_extraction(self, x):
        feat_scale_level = []
        for scale in range(self.pyramid_level):
            x = self.feat_extracts[scale](x)
            if scale != 0:
                feat_scale_level.append(x)
        return x, feat_scale_level

    def _shared_feat_enhancement(self, x, h, w):
        x = einops.rearrange(x, "B (H W) C -> B H W C", H=h, W=w)
        for k, blk in enumerate(self.feat_enhance_transformer):
            x = blk(x)
            if k % 2 == 0:
                x = einops.rearrange(x, "B (H W) C -> B H W C", H=h)
        return x

    def _estimate_local_motion(self, feat):
        motion = []
        for k, blk in enumerate(self.local_motion_atmformer):
            B, h, w, _ = feat.size()
            feat, x_motion = blk(feat, h, w, B // 2)
            if k == 0:
                feat = einops.rearrange(feat, "B (H W) C -> B H W C", H=h)
            x_motion = einops.rearrange(x_motion, "(N B) L K -> B L (N K)", N=2)
            motion.append(x_motion)
        feat_concat = einops.rearrange(feat, "(N B) (H W) C -> B (N C) H W", N=2, H=h)
        motion = torch.cat(motion, dim=2)
        motion = einops.rearrange(motion, "B (H W) C -> B C H W", H=h)
        out = self.local_motion_mlp(torch.cat([motion, feat_concat], dim=1))
        opt_flow_0 = out[:, :2]
        opt_flow_1 = out[:, 2:4]
        occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
        return opt_flow_0, opt_flow_1, occ_mask1, feat, out

    def _estimate_global_motion(self, x, feat_scale_level):
        feat_ = self.last_feat_extract(x)
        feat_scale_level.append(feat_)
        feat_scale_level.pop(0)
        feat_, h_, w_ = self.global_feature_fusion(feat_scale_level)
        feat_ = einops.rearrange(feat_, "B (H W) C -> B H W C", H=h_)

        motion = []
        for k, blk in enumerate(self.global_motion_atmformer):
            B, h_, w_, _ = feat_.size()
            feat_, x_motion = blk(feat_, h_, w_, B // 2)
            if k == 0:
                feat_ = einops.rearrange(feat_, "B (H W) C -> B H W C", H=h_)
            x_motion = einops.rearrange(x_motion, "(N B) L K -> B L (N K)", N=2)
            motion.append(x_motion)
        feat_ = einops.rearrange(feat_, "(N B) (H W) C -> B (N C) H W", N=2, H=h_)
        motion = torch.cat(motion, dim=2)
        motion = einops.rearrange(motion, "B (H W) C -> B C H W", H=h_)
        out = self.global_motion_mlp(torch.cat([motion, feat_], dim=1))
        return out[:, :2], out[:, 2:4], torch.sigmoid(out[:, 4].unsqueeze(1))

    def _residual_refinement(self, feat, im0, I_t_0, im1, I_t_1, I_t, backbone_decoder_feats):
        feat0 = torch.cat([feat, im0, I_t_0, im1, I_t_1, I_t], dim=1)
        feat0 = self.proj_ref(feat0)
        feat1 = self.down1(feat0)
        feat2 = self.down2(torch.cat([feat1, backbone_decoder_feats.pop()], dim=1))
        feat3 = self.down3(torch.cat([feat2, backbone_decoder_feats.pop()], dim=1))
        feat2_ = self.up1(feat3)
        feat1_ = self.up2(torch.cat([feat2_, feat2], dim=1))
        feat0_ = self.up3(torch.cat([feat1_, feat1], dim=1))
        I_t_residual = self.refine_head(torch.cat([feat0_, feat0], dim=1))
        I_t_residual = 2 * torch.sigmoid(I_t_residual) - 1
        return I_t_residual

    def _forward_normal(self, im0, im1):
        B, _, H, W = im0.size()
        im0_list = [im0]
        im1_list = [im1]
        for scale in range(self.pyramid_level - 1):
            im0_list.append(F.interpolate(im0_list[-1], scale_factor=0.5, mode="bilinear", align_corners=True))
            im1_list.append(F.interpolate(im1_list[-1], scale_factor=0.5, mode="bilinear", align_corners=True))

        feat_ = torch.cat([im0, im1], dim=0)
        feat_, feat_scale_level = self._shared_feat_extraction(feat_)
        feat, h, w = self.cross_scale_feature_fusion(feat_scale_level)

        if self.global_motion:
            opt_flow_0, opt_flow_1, occ_mask1 = self._estimate_global_motion(feat_, feat_scale_level)
            occ_mask2 = 1 - occ_mask1
            im0_down = F.interpolate(im0_list[-1], scale_factor=0.5, mode="bilinear", align_corners=True)
            im1_down = F.interpolate(im1_list[-1], scale_factor=0.5, mode="bilinear", align_corners=True)
            opt_flow_0_up = upsample_flow(opt_flow_0, upsample_factor=2, mode="bilinear")
            opt_flow_1_up = upsample_flow(opt_flow_1, upsample_factor=2, mode="bilinear")
            feat = einops.rearrange(feat, "B (H W) C -> B C H W", H=h)
            feat0 = flow_warp(feat[:B], flow=opt_flow_0_up)
            feat1 = flow_warp(feat[B:], flow=opt_flow_1_up)
            feat = torch.cat([feat0, feat1], dim=0)
            feat = einops.rearrange(feat, "B C H W -> B H W C", H=h)
            for i in reversed(range(self.pyramid_level)):
                im0_list[i] = flow_warp(im0_list[i], flow=opt_flow_0_up)
                im1_list[i] = flow_warp(im1_list[i], flow=opt_flow_1_up)
                if i != 0:
                    opt_flow_0_up = upsample_flow(opt_flow_0_up, upsample_factor=2, mode="bilinear")
                    opt_flow_1_up = upsample_flow(opt_flow_1_up, upsample_factor=2, mode="bilinear")
        else:
            feat = einops.rearrange(feat, "B (H W) C -> B H W C", H=h)

        opt_flow_0, opt_flow_1, occ_mask1, feat, out = self._estimate_local_motion(feat)
        occ_mask2 = 1 - occ_mask1
        feat = self._shared_feat_enhancement(feat, h, w)
        feat = einops.rearrange(feat, "(N B) (H W) C -> B (N C) H W", N=2, H=h)

        I_t_0 = flow_warp(im0_list[-1], opt_flow_0)
        I_t_1 = flow_warp(im1_list[-1], opt_flow_1)
        I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1

        feat1 = flow_warp(feat[:, : self.fused_dims[0]], opt_flow_0)
        feat2 = flow_warp(feat[:, self.fused_dims[0] : self.fused_dims[-1]], opt_flow_1)
        feat = torch.cat([feat1, feat2, out], dim=1)

        backbone_decoder_feats = []
        for i, scale in enumerate(reversed(range(self.pyramid_level - 1))):
            feat = self.upsample_pyramid[i](feat)
            out = feat[:, -self.motion_out_dim :]
            opt_flow_0 = out[:, :2]
            opt_flow_1 = out[:, 2:4]
            occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
            occ_mask2 = 1 - occ_mask1
            if scale != 0:
                backbone_decoder_feats.append(feat[:, : -self.motion_out_dim])
            I_t_0 = flow_warp(im0_list[scale], opt_flow_0)
            I_t_1 = flow_warp(im1_list[scale], opt_flow_1)
            I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1

        I_t_residual = self._residual_refinement(feat, im0, I_t_0, im1, I_t_1, I_t, backbone_decoder_feats)
        I_t = torch.clamp(I_t + I_t_residual, 0, 1)
        return {"I_t": I_t}

    def _forward_global_ensemble(self, im0, im1):
        B, _, H, W = im0.size()
        im0_list = [im0]
        im1_list = [im1]
        for scale in range(self.pyramid_level - 1):
            im0_list.append(F.interpolate(im0_list[-1], scale_factor=0.5, mode="bilinear", align_corners=True))
            im1_list.append(F.interpolate(im1_list[-1], scale_factor=0.5, mode="bilinear", align_corners=True))

        feat_ = torch.cat([im0, im1], dim=0)
        feat_, feat_scale_level = self._shared_feat_extraction(feat_)
        feat, h, w = self.cross_scale_feature_fusion(feat_scale_level)

        if self.global_motion:
            opt_flow_0, opt_flow_1 = self._multiscale_global_motion_ensemble(im0, im1)
            opt_flow_0_up = upsample_flow(opt_flow_0, upsample_factor=2, mode="bilinear")
            opt_flow_1_up = upsample_flow(opt_flow_1, upsample_factor=2, mode="bilinear")
            feat = einops.rearrange(feat, "B (H W) C -> B C H W", H=h)
            feat0 = flow_warp(feat[:B], flow=opt_flow_0_up)
            feat1 = flow_warp(feat[B:], flow=opt_flow_1_up)
            feat = torch.cat([feat0, feat1], dim=0)
            feat = einops.rearrange(feat, "B C H W -> B H W C", H=h)
            for i in reversed(range(self.pyramid_level)):
                im0_list[i] = flow_warp(im0_list[i], flow=opt_flow_0_up)
                im1_list[i] = flow_warp(im1_list[i], flow=opt_flow_1_up)
                if i != 0:
                    opt_flow_0_up = upsample_flow(opt_flow_0_up, upsample_factor=2, mode="bilinear")
                    opt_flow_1_up = upsample_flow(opt_flow_1_up, upsample_factor=2, mode="bilinear")
        else:
            feat = einops.rearrange(feat, "B (H W) C -> B H W C", H=h)

        opt_flow_0, opt_flow_1, occ_mask1, feat, out = self._estimate_local_motion(feat)
        occ_mask2 = 1 - occ_mask1
        feat = self._shared_feat_enhancement(feat, h, w)
        feat = einops.rearrange(feat, "(N B) (H W) C -> B (N C) H W", N=2, H=h)

        I_t_0 = flow_warp(im0_list[-1], opt_flow_0)
        I_t_1 = flow_warp(im1_list[-1], opt_flow_1)
        I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1

        feat1 = flow_warp(feat[:, : self.fused_dims[0]], opt_flow_0)
        feat2 = flow_warp(feat[:, self.fused_dims[0] : self.fused_dims[-1]], opt_flow_1)
        feat = torch.cat([feat1, feat2, out], dim=1)

        backbone_decoder_feats = []
        for i, scale in enumerate(reversed(range(self.pyramid_level - 1))):
            feat = self.upsample_pyramid[i](feat)
            out = feat[:, -self.motion_out_dim :]
            opt_flow_0 = out[:, :2]
            opt_flow_1 = out[:, 2:4]
            occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
            if scale != 0:
                backbone_decoder_feats.append(feat[:, : -self.motion_out_dim])
            I_t_0 = flow_warp(im0_list[scale], opt_flow_0)
            I_t_1 = flow_warp(im1_list[scale], opt_flow_1)
            I_t = occ_mask1 * I_t_0 + (1 - occ_mask1) * I_t_1

        I_t_residual = self._residual_refinement(feat, im0, I_t_0, im1, I_t_1, I_t, backbone_decoder_feats)
        I_t = torch.clamp(I_t + I_t_residual, 0, 1)
        return {"I_t": I_t}

    def _multiscale_global_motion_ensemble(self, im0, im1):
        B, _, _, _ = im0.size()
        im = torch.cat([im0, im1], dim=0)

        feat_, feat_scale_level = self._shared_feat_extraction(im)
        level0 = self._estimate_global_motion(feat_, feat_scale_level)
        loss0 = self._global_alignmentness(level0, im0, im1)

        im = F.interpolate(im, scale_factor=0.5, mode="bilinear", align_corners=True)
        feat_, feat_scale_level = self._shared_feat_extraction(im)
        level1 = self._estimate_global_motion(feat_, feat_scale_level)
        loss1 = self._global_alignmentness(level1, im0, im1)

        im = F.interpolate(im, scale_factor=0.5, mode="bilinear", align_corners=True)
        feat_, feat_scale_level = self._shared_feat_extraction(im)
        level2 = self._estimate_global_motion(feat_, feat_scale_level)
        loss2 = self._global_alignmentness(level2, im0, im1)

        opt_flow0 = torch.zeros_like(level0[0])
        opt_flow1 = torch.zeros_like(level0[1])
        for i in range(B):
            losses = [loss0[i], loss1[i], loss2[i]]
            scale_idx = losses.index(min(losses))
            if scale_idx == 0:
                opt_flow0[i] = level0[0][i]
                opt_flow1[i] = level0[1][i]
            elif scale_idx == 1:
                opt_flow0[i] = upsample_flow(level1[0][i, None], upsample_factor=2, mode="bilinear")
                opt_flow1[i] = upsample_flow(level1[1][i, None], upsample_factor=2, mode="bilinear")
            else:
                opt_flow0[i] = upsample_flow(level2[0][i, None], upsample_factor=4, mode="bilinear")
                opt_flow1[i] = upsample_flow(level2[1][i, None], upsample_factor=4, mode="bilinear")
        return opt_flow0, opt_flow1

    def _global_alignmentness(self, x, im0, im1):
        opt_flow0, opt_flow1 = x[0], x[1]
        _, _, H1, W1 = opt_flow0.size()
        _, _, H0, W0 = im0.size()
        factor = H0 // H1
        opt_flow0 = upsample_flow(opt_flow0, factor, mode="bilinear")
        opt_flow1 = upsample_flow(opt_flow1, factor, mode="bilinear")
        im0 = flow_warp(im0, opt_flow0)
        im1 = flow_warp(im1, opt_flow1)
        loss = F.l1_loss(im0, im1, reduction="none").mean(dim=[1, 2, 3])
        return loss
