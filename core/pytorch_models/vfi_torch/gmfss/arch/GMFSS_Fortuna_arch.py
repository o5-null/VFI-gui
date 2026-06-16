"""
GMFSS Fortuna Basic Architecture (no IFNet guidance).

Port from: https://github.com/98mxr/GMFSS_Fortuna

Shares all sub-networks with the union variant except IFNet.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Import shared architecture classes from the union variant
from .GMFSS_Fortuna_union_arch import (
    GMFlow,
    MetricNet,
    FeatureNet,
    GridNet,
    softsplat,
)


class Model:
    """GMFSS Fortuna Basic (without IFNet guidance)."""

    def __init__(self):
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()
        self.version = 3.9

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def to(self, device):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)
        return self

    def half(self):
        self.flownet = self.flownet.half()
        self.metricnet = self.metricnet.half()
        self.feat_ext = self.feat_ext.half()
        self.fusionnet = self.fusionnet.half()
        return self

    def load_model(self, path_dict):
        self.flownet.load_state_dict(torch.load(path_dict["flownet"], map_location="cpu", weights_only=False))
        self.metricnet.load_state_dict(torch.load(path_dict["metricnet"], map_location="cpu", weights_only=False))
        self.feat_ext.load_state_dict(torch.load(path_dict["feat_ext"], map_location="cpu", weights_only=False))
        self.fusionnet.load_state_dict(torch.load(path_dict["fusionnet"], map_location="cpu", weights_only=False))

    def reuse(self, img0, img1, scale):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)

        img0 = F.interpolate(
            img0, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        img1 = F.interpolate(
            img1, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        if scale != 1.0:
            imgf0 = F.interpolate(
                img0, scale_factor=scale, mode="bilinear", align_corners=False
            )
            imgf1 = F.interpolate(
                img1, scale_factor=scale, mode="bilinear", align_corners=False
            )
        else:
            imgf0 = img0
            imgf1 = img1
        flow01 = self.flownet(imgf0, imgf1, pred_bidir_flow=True)
        flow10 = self.flownet(imgf1, imgf0, pred_bidir_flow=True)
        if scale != 1.0:
            flow01 = (
                F.interpolate(
                    flow01,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                / scale
            )
            flow10 = (
                F.interpolate(
                    flow10,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                / scale
            )

        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        return (
            flow01,
            flow10,
            metric0,
            metric1,
            feat11,
            feat12,
            feat13,
            feat21,
            feat22,
            feat23,
        )

    def inference(
        self,
        img0,
        img1,
        flow01,
        flow10,
        metric0,
        metric1,
        feat11,
        feat12,
        feat13,
        feat21,
        feat22,
        feat23,
        timestep,
    ):
        F1t = timestep * flow01
        F2t = (1 - timestep) * flow10

        Z1t = timestep * metric0
        Z2t = (1 - timestep) * metric1

        img0_half = F.interpolate(
            img0, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        I1t = softsplat(img0_half, F1t, Z1t, strMode="soft")
        img1_half = F.interpolate(
            img1, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        I2t = softsplat(img1_half, F2t, Z2t, strMode="soft")

        feat1t1 = softsplat(feat11, F1t, Z1t, strMode="soft")
        feat2t1 = softsplat(feat21, F2t, Z2t, strMode="soft")

        F1td = (
            F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat1t2 = softsplat(feat12, F1td, Z1d, strMode="soft")
        F2td = (
            F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat2t2 = softsplat(feat22, F2td, Z2d, strMode="soft")

        F1tdd = (
            F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False)
            * 0.25
        )
        Z1dd = F.interpolate(
            Z1t, scale_factor=0.25, mode="bilinear", align_corners=False
        )
        feat1t3 = softsplat(feat13, F1tdd, Z1dd, strMode="soft")
        F2tdd = (
            F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False)
            * 0.25
        )
        Z2dd = F.interpolate(
            Z2t, scale_factor=0.25, mode="bilinear", align_corners=False
        )
        feat2t3 = softsplat(feat23, F2tdd, Z2dd, strMode="soft")

        # Basic (non-union): concat img0_half and img1_half instead of using IFNet
        out = self.fusionnet(
            torch.cat([img0_half, I1t, I2t, img1_half], dim=1),
            torch.cat([feat1t1, feat2t1], dim=1),
            torch.cat([feat1t2, feat2t2], dim=1),
            torch.cat([feat1t3, feat2t3], dim=1),
        )

        return torch.clamp(out, 0, 1)
