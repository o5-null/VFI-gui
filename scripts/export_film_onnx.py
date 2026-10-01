#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FILM ONNX 导出脚本 / FILM ONNX export script.

背景 / Background
-----------------
`models/film/film_net_fp32.pt` 是 TorchScript 模型（`interpolator.Interpolator`，
来自 dajes/frame-interpolation-pytorch）。直接用 ``torch.onnx.export`` 导出
TorchScript 会失败，因为其图里 ``F.interpolate`` 的 ``size`` 来自
``aten::slice(aten::size(x), 2, None)`` 这种动态 int[] 值，legacy exporter 会报
``Unsupported: ONNX export of operator interpolate (with a scalar output_size)``。

解决方案 / Solution
-------------------
用纯 PyTorch 忠实重建同一套 FILM 架构（与本文件内嵌代码一致），加载 TorchScript 的
``state_dict``（键名完全匹配、严格加载），再 trace + export。纯 Python 模块在静态
256x256 输入下 trace 时，``.shape`` 返回具体整数，``interpolate`` 的 size 变成常量，
从而绕开上述问题，且输出与原 TorchScript 逐位一致。

关于 timestep / About timestep
------------------------------
原 TorchScript 内部把中间时刻硬编码为 0.5（``torch.full_like(batch_dt, .5)``），
**完全不使用** batch_dt 的值，因此原始模型只能输出 t=0.5 的中间帧。为满足统一 ONNX
接口（必须带 timestep 输入），本脚本把 timestep 真正接入光流缩放：
  中间帧 = t 时刻。t=0.5 时与原始 TorchScript 输出逐位一致。

用法 / Usage
-----------
    D:\\code\\VFI\\runtime\\cuda\\Scripts\\python.exe scripts/export_film_onnx.py
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# FILM 架构（内嵌自 dajes/frame-interpolation-pytorch，保持忠实实现）
# FILM architecture (embedded from dajes/frame-interpolation-pytorch)
# ---------------------------------------------------------------------------

def conv(in_channels, out_channels, size, activation="relu"):
    """Conv2d + 可选 LeakyReLU（activation=None 时返回裸 Conv2d）。"""
    _conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=size,
        padding="same",
    )
    if activation is None:
        return _conv
    assert activation == "relu"
    return nn.Sequential(_conv, nn.LeakyReLU(0.2))


class SubTreeExtractor(nn.Module):
    """级联特征金字塔的子提取器（每层两个 3x3 卷积，层间平均池化）。"""

    def __init__(self, in_channels=3, channels=64, n_layers=4):
        super().__init__()
        convs = []
        for i in range(n_layers):
            convs.append(
                nn.Sequential(
                    conv(in_channels, (channels << i), 3),
                    conv((channels << i), (channels << i), 3),
                )
            )
            in_channels = channels << i
        self.convs = nn.ModuleList(convs)

    def forward(self, image: torch.Tensor, n: int) -> List[torch.Tensor]:
        head = image
        pyramid = []
        for i, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if i < n - 1:
                head = F.avg_pool2d(head, kernel_size=2, stride=2)
        return pyramid


class FeatureExtractor(nn.Module):
    """级联多尺度特征提取器（Multi-view Image Fusion 风格）。"""

    def __init__(self, in_channels=3, channels=64, sub_levels=4):
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(in_channels, channels, sub_levels)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        sub_pyramids: List[List[torch.Tensor]] = []
        for i in range(len(image_pyramid)):
            capped = min(len(image_pyramid) - i, self.sub_levels)
            sub_pyramids.append(self.extract_sublevels(image_pyramid[i], capped))

        feature_pyramid: List[torch.Tensor] = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.sub_levels):
                if j <= i:
                    features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
        return feature_pyramid


def get_channels_at_level(level, filters):
    """计算 Fusion 某一层的输入通道数。"""
    n_images = 2
    channels = 3
    flows = 2
    return (sum(filters << i for i in range(level)) + channels + flows) * n_images


class Fusion(nn.Module):
    """U-Net 解码器：从粗到细融合对齐后的图像/特征/光流金字塔，输出 RGB。"""

    def __init__(self, n_layers=4, specialized_layers=3, filters=64):
        super().__init__()
        self.output_conv = nn.Conv2d(filters, 3, kernel_size=1)
        self.convs = nn.ModuleList()

        in_channels = get_channels_at_level(n_layers, filters)
        increase = 0
        for i in range(n_layers)[::-1]:
            num_filters = (
                (filters << i) if i < specialized_layers else (filters << specialized_layers)
            )
            layers = nn.ModuleList(
                [
                    conv(in_channels, num_filters, size=2, activation=None),
                    conv(in_channels + (increase or num_filters), num_filters, size=3),
                    conv(num_filters, num_filters, size=3),
                ]
            )
            self.convs.append(layers)
            in_channels = num_filters
            increase = get_channels_at_level(i, filters) - num_filters // 2

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        net = pyramid[-1]
        for k, layers in enumerate(self.convs):
            i = len(self.convs) - 1 - k
            level_size = pyramid[i].shape[2:4]
            net = F.interpolate(net, size=level_size, mode="nearest")
            net = layers[0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = layers[1](net)
            net = layers[2](net)
        return self.output_conv(net)


class FlowEstimator(nn.Module):
    """单层残差光流预测器。"""

    def __init__(self, in_channels: int, num_convs: int, num_filters: int):
        super().__init__()
        self._convs = nn.ModuleList()
        for _ in range(num_convs):
            self._convs.append(conv(in_channels=in_channels, out_channels=num_filters, size=3))
            in_channels = num_filters
        self._convs.append(conv(in_channels, num_filters // 2, size=1))
        self._convs.append(conv(num_filters // 2, 2, size=1, activation=None))

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        net = torch.cat([features_a, features_b], dim=1)
        for c in self._convs:
            net = c(net)
        return net


class PyramidFlowEstimator(nn.Module):
    """由粗到细的金字塔光流残差估计。"""

    def __init__(self, filters=64, flow_convs=(3, 3, 3, 3), flow_filters=(32, 64, 128, 256)):
        super().__init__()
        in_channels = filters << 1
        predictors = []
        for i in range(len(flow_convs)):
            predictors.append(
                FlowEstimator(
                    in_channels=in_channels,
                    num_convs=flow_convs[i],
                    num_filters=flow_filters[i],
                )
            )
            in_channels += filters << (i + 2)
        self._predictor = predictors[-1]
        self._predictors = nn.ModuleList(predictors[:-1][::-1])

    def forward(self, feature_pyramid_a, feature_pyramid_b):
        levels = len(feature_pyramid_a)
        v = self._predictor(feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        for i in range(levels - 2, len(self._predictors) - 1, -1):
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode="bilinear")
            warped = warp(feature_pyramid_b[i], v)
            v_residual = self._predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v

        for k, predictor in enumerate(self._predictors):
            i = len(self._predictors) - 1 - k
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode="bilinear")
            warped = warp(feature_pyramid_b[i], v)
            v_residual = predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v
        return residuals


class Interpolator(nn.Module):
    """FILM 主模型：特征提取 -> 金字塔光流 -> 扭曲 -> 融合。

    与原实现的唯一区别：使用传入的 ``batch_dt`` 作为中间时刻用于光流缩放，
    这样 timestep 成为真正的 ONNX 输入（原实现硬编码 0.5）。
    """

    def __init__(
        self,
        pyramid_levels=7,
        fusion_pyramid_levels=5,
        specialized_levels=3,
        sub_levels=4,
        filters=64,
        flow_convs=(3, 3, 3, 3),
        flow_filters=(32, 64, 128, 256),
    ):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels
        self.extract = FeatureExtractor(3, filters, sub_levels)
        self.predict_flow = PyramidFlowEstimator(filters, flow_convs, flow_filters)
        self.fuse = Fusion(sub_levels, specialized_levels, filters)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, batch_dt: torch.Tensor) -> torch.Tensor:
        image_pyramids = [
            build_image_pyramid(x0, self.pyramid_levels),
            build_image_pyramid(x1, self.pyramid_levels),
        ]
        feature_pyramids = [self.extract(image_pyramids[0]), self.extract(image_pyramids[1])]

        forward_residual = self.predict_flow(feature_pyramids[0], feature_pyramids[1])
        backward_residual = self.predict_flow(feature_pyramids[1], feature_pyramids[0])

        forward_flow_pyramid = flow_pyramid_synthesis(forward_residual)[: self.fusion_pyramid_levels]
        backward_flow_pyramid = flow_pyramid_synthesis(backward_residual)[: self.fusion_pyramid_levels]

        # 使用传入的中间时刻 t（原实现为固定 0.5）
        t = batch_dt.reshape(-1)
        backward_flow = multiply_pyramid(backward_flow_pyramid, t)
        forward_flow = multiply_pyramid(forward_flow_pyramid, 1 - t)

        pyramids_to_warp = [
            concatenate_pyramids(
                image_pyramids[0][: self.fusion_pyramid_levels],
                feature_pyramids[0][: self.fusion_pyramid_levels],
            ),
            concatenate_pyramids(
                image_pyramids[1][: self.fusion_pyramid_levels],
                feature_pyramids[1][: self.fusion_pyramid_levels],
            ),
        ]
        forward_warped = pyramid_warp(pyramids_to_warp[0], backward_flow)
        backward_warped = pyramid_warp(pyramids_to_warp[1], forward_flow)

        aligned = concatenate_pyramids(forward_warped, backward_warped)
        aligned = concatenate_pyramids(aligned, backward_flow)
        aligned = concatenate_pyramids(aligned, forward_flow)
        return self.fuse(aligned)


# ---------------------------------------------------------------------------
# 工具函数（同样内嵌自 dajes 实现）
# ---------------------------------------------------------------------------

def build_image_pyramid(image: torch.Tensor, pyramid_levels: int = 3) -> List[torch.Tensor]:
    """构建图像金字塔：最细层为原图，逐层平均池化减半。"""
    pyramid = []
    for i in range(pyramid_levels):
        pyramid.append(image)
        if i < pyramid_levels - 1:
            image = F.avg_pool2d(image, 2, 2)
    return pyramid


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """用给定光流对图像做反向双线性扭曲（BCHW 输入输出）。"""
    flow = -flow.flip(1)
    dtype = flow.dtype
    device = flow.device

    ls1 = 1 - 1 / flow.shape[3]
    ls2 = 1 - 1 / flow.shape[2]
    normalized_flow2 = flow.permute(0, 2, 3, 1) / torch.tensor(
        [flow.shape[2] * 0.5, flow.shape[3] * 0.5], dtype=dtype, device=device
    )[None, None, None]
    normalized_flow2 = torch.stack(
        [
            torch.linspace(-ls1, ls1, flow.shape[3], dtype=dtype, device=device)[None, None, :]
            - normalized_flow2[..., 1],
            torch.linspace(-ls2, ls2, flow.shape[2], dtype=dtype, device=device)[None, :, None]
            - normalized_flow2[..., 0],
        ],
        dim=3,
    )
    warped = F.grid_sample(
        input=image,
        grid=normalized_flow2,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped.reshape(image.shape)


def multiply_pyramid(pyramid: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
    """金字塔每层乘以逐样本标量。"""
    return [image * scalar for image in pyramid]


def flow_pyramid_synthesis(residual_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """把残差光流金字塔合成为光流金字塔。"""
    flow = residual_pyramid[-1]
    flow_pyramid: List[torch.Tensor] = [flow]
    for residual_flow in residual_pyramid[:-1][::-1]:
        level_size = residual_flow.shape[2:4]
        flow = F.interpolate(2 * flow, size=level_size, mode="bilinear")
        flow = residual_flow + flow
        flow_pyramid.insert(0, flow)
    return flow_pyramid


def pyramid_warp(feature_pyramid: List[torch.Tensor], flow_pyramid: List[torch.Tensor]):
    """按金字塔逐层扭曲特征。"""
    return [warp(features, flow) for features, flow in zip(feature_pyramid, flow_pyramid)]


def concatenate_pyramids(pyramid1: List[torch.Tensor], pyramid2: List[torch.Tensor]):
    """逐层在通道维拼接两个金字塔。"""
    return [torch.cat([f1, f2], dim=1) for f1, f2 in zip(pyramid1, pyramid2)]


# ---------------------------------------------------------------------------
# 导出封装：统一 ONNX 接口 img0 / img1 / timestep -> output
# ---------------------------------------------------------------------------

class ExportWrapper(nn.Module):
    """把 timestep [1] 适配为内部 Interpolator 所需的 [1,1]。"""

    def __init__(self, net: Interpolator):
        super().__init__()
        self.net = net

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: torch.Tensor):
        return self.net(img0, img1, timestep.reshape(1, 1))


def build_model(checkpoint_path: str) -> nn.Module:
    """加载 TorchScript 权重到重建的纯 PyTorch FILM 模型。"""
    ref = torch.jit.load(checkpoint_path, map_location="cpu").eval()
    model = Interpolator().eval()
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict 不匹配: missing={missing}, unexpected={unexpected}")
    return model, ref


def export_onnx(model: nn.Module, out_path: str, size: int = 256, opset: int = 17) -> None:
    """静态 256x256 trace + legacy ONNX 导出。"""
    wrapper = ExportWrapper(model).eval()
    img0 = torch.rand(1, 3, size, size)
    img1 = torch.rand(1, 3, size, size)
    timestep = torch.tensor([0.5])

    torch.onnx.export(
        wrapper,
        (img0, img1, timestep),
        out_path,
        input_names=["img0", "img1", "timestep"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def validate(out_path: str, ts_model: nn.Module, size: int = 256) -> None:
    """用 onnxruntime 加载并做数值校验，打印报告。"""
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    inputs = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]

    torch.manual_seed(0)
    img0 = torch.rand(1, 3, size, size)
    img1 = torch.rand(1, 3, size, size)
    for t in (0.5, 0.25, 0.75):
        ts = torch.tensor([t])
        y = sess.run(
            None,
            {"img0": img0.numpy(), "img1": img1.numpy(), "timestep": ts.numpy()},
        )[0]
        with torch.no_grad():
            y_ref = ts_model(img0, img1, ts.reshape(1, 1)).numpy()
        mae = float(np.abs(y - y_ref).mean())
        print(
            f"  t={t}: shape={y.shape} dtype={y.dtype} finite={np.isfinite(y).all()} "
            f"MAE(vs TorchScript@0.5)={mae:.3e}"
        )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n产物: {out_path}  ({size_mb:.1f} MB)")
    print(f"输入: {inputs}")
    print(f"输出: {outputs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 FILM 为 ONNX")
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--checkpoint",
        default=str(root / "models" / "film" / "film_net_fp32.pt"),
        help="TorchScript 权重路径",
    )
    parser.add_argument(
        "--output",
        default=str(root / "models" / "film" / "film_net_fp32.onnx"),
        help="ONNX 输出路径",
    )
    parser.add_argument("--size", type=int, default=256, help="静态输入边长(64 的倍数)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    args = parser.parse_args()

    assert args.size % 64 == 0, "FILM 要求边长是 64 的倍数"

    print(f"加载权重: {args.checkpoint}")
    model, ts_model = build_model(args.checkpoint)
    print("state_dict 严格加载成功（与 TorchScript 键名完全匹配）")

    # 先做一次 PyTorch 侧一致性检查（t=0.5 应与原 TorchScript 逐位一致）
    img0 = torch.rand(1, 3, args.size, args.size)
    img1 = torch.rand(1, 3, args.size, args.size)
    with torch.no_grad():
        y_new = model(img0, img1, torch.tensor([0.5]))
        y_ref = ts_model(img0, img1, torch.tensor([[0.5]]))
    print(f"重建模型 vs TorchScript @t=0.5  max abs diff = {float((y_new - y_ref).abs().max()):.3e}")

    print(f"\n导出 ONNX: {args.output}")
    export_onnx(model, args.output, size=args.size, opset=args.opset)

    print("\n验证:")
    validate(args.output, ts_model, size=args.size)


if __name__ == "__main__":
    # cv2 只是可选依赖，这里不使用，保持脚本轻量
    main()
