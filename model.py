import torch
from torch import nn
from typing import Literal


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init


def get_act(act: str) -> nn.Module:
    if act == "relu":
        return nn.ReLU()
    elif act == "celu":
        return nn.CELU()
    else:
        raise NotImplementedError


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        act: str = "celu",
    ):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

        self.stride = stride
        self.act = get_act(act)

        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2.weight)
        init.kaiming_normal_(self.conv3.weight)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.batch_norm1(self.conv1(x)))
        out = self.act(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))
        out += identity
        out = self.act(out)
        return out


class Block(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        act: str = "celu",
    ):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,  # Set stride always 1 here
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,  # Set stride argument here
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.stride = stride
        self.act = get_act(act)
        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2.weight)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.act(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))

        out += identity
        out = self.act(out)
        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        ResBlock,
        layer_list: list[int],
        num_classes: int,
        num_channels: int = 3,
        act: str = "celu",
    ):
        super(_ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            num_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.act = get_act(act)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()

        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.fc.weight)

    def forward(self, x: torch.Tensor):
        x = self.act(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        layers = []

        layers.append(ResBlock(self.in_channels, planes, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet18(num_classes: int):
    return _ResNet(Block, [2, 2, 2, 2], num_classes)


def ResNet50(num_classes: int):
    return _ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet152(num_classes):
    return _ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def ResNetFactory(
    kind: Literal["18", "50", "152"],
    num_classes: int = 2,
) -> torch.nn.Module:
    if kind == "18":
        return ResNet18(num_classes)
    if kind == "50":
        return ResNet50(num_classes)
    if kind == "152":
        return ResNet152(num_classes)


if __name__ == "__main__":
    net = ResNet50(2).cuda()
    x = torch.randn(5, 3, 64, 64).cuda()
    y = net(x)
    print(y.size())
