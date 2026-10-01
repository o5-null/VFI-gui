"""3D ResNet encoder for FLAVR.

Adapted from:
    https://github.com/pytorch/vision/tree/master/torchvision/models/video
    https://github.com/tarun005/FLAVR/blob/main/model/resnet_3D.py

Provides unet_18 (3D ResNet-18) as an encoder with SEGating.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["unet_18", "unet_34"]

useBias = False


class Identity(nn.Module):
    def forward(self, x):
        return x


class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super().__init__(in_planes, out_planes, kernel_size=(3, 3, 3),
                         stride=stride, padding=padding, bias=useBias)

    @staticmethod
    def get_downsample_stride(stride, temporal_stride):
        if temporal_stride:
            return (temporal_stride, stride, stride)
        return (stride, stride, stride)


class BasicStem(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=useBias),
            nn.BatchNorm3d(64) if batchnorm is not None else Identity(),
            nn.ReLU(inplace=False),
        )


class SEGating(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn_layer(self.pool(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes) if batchnorm is not None else Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes) if batchnorm is not None else Identity(),
        )
        self.fg = SEGating(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class VideoResNet(nn.Module):
    def __init__(self, block, conv_makers, layers, stem, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2, temporal_stride=1)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2, temporal_stride=1)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=1, temporal_stride=1)
        self._initialize_weights()

    def forward(self, x):
        x_0 = self.stem(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_0, x_1, x_2, x_3, x_4

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, temporal_stride=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride, temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion) if batchnorm is not None else Identity(),
            )
            stride = ds_stride

        layers_ = [block(self.inplanes, planes, conv_builder, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_.append(block(self.inplanes, planes, conv_builder))
        return nn.Sequential(*layers_)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)
    if pretrained:
        # FLAVR doesn't use pre-trained weights from kinetics
        pass
    return model


def unet_18(pretrained=False, progress=True, **kwargs):
    """Construct 18-layer 3D ResNet encoder for FLAVR UNet."""
    global batchnorm
    batchnorm = nn.BatchNorm3d

    return _video_resnet(
        "r3d_18", pretrained, progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def unet_34(pretrained=False, progress=True, **kwargs):
    """Construct 34-layer 3D ResNet encoder for FLAVR UNet."""
    global batchnorm
    batchnorm = nn.BatchNorm3d

    return _video_resnet(
        "r3d_34", pretrained, progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[3, 4, 6, 3],
        stem=BasicStem,
        **kwargs,
    )


batchnorm = nn.BatchNorm3d
