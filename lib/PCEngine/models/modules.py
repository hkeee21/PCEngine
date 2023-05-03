from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..script.sptensor import spTensor
from ..script.spconv import conv3d
from ..script.batchnorm import BatchNorm
from ..script.activation import ReLU


##### Sparse 3D Modules ######
class SparseConvBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: int = 1) -> None:
        super().__init__(
            conv3d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride, 
                    padding=padding
                    ),
            BatchNorm(out_channels),
            ReLU(True),
        )


class SparseConvTransposeBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1) -> None:
        super().__init__(
            conv3d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    transposed=True),
            BatchNorm(out_channels),
            ReLU(True),
        )


class SparseResBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            conv3d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride),
            BatchNorm(out_channels),
            ReLU(True),
            conv3d(in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size),
            BatchNorm(out_channels),
        )

        if in_channels != out_channels or np.prod(stride) != 1:
            self.shortcut = nn.Sequential(
                conv3d(in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=1, 
                        stride=stride),
                BatchNorm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = ReLU(True)

    def forward(self, x: spTensor) -> spTensor:
        x = self.relu(self.main(x) + self.shortcut(x))
        return x


##### 2D Modules ######
class SCConv2d(nn.Module):
    """
    Jiang-Jiang Liu, Qibin Hou, Ming-Ming Cheng, Changhu Wang and Jiashi Feng,
    Improving Convolutional Networks with Self-Calibrated Convolutions.
    In CVPR 2020.

    The same implementation as paper but slightly different from official code.
    
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        paths = []
        for i in range(4):
            paths.append(
                nn.Conv2d(
                    in_channels // 2,
                    out_channels // 2, 
                    kernel_size=kernel_size,
                    stride=1 if i not in [0, 3] else stride,
                    padding=(kernel_size-1)//2,
                    bias=False
                )
            )
        self.paths = nn.ModuleList(paths)
        self.pool = nn.AvgPool2d(4, 4)

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        x1, x2 = inputs[:, :C//2, :, :], inputs[:, C//2:, :, :]
        # B, C//2, H, W
        y2 = self.paths[0](x2)
        # B, C//2, H, W
        y1_d = F.interpolate(
            self.paths[1](self.pool(x1)),
            (H, W)
        )
        y1_d = torch.sigmoid(x1 + y1_d)
        # B, C//2, H, W
        y1 = self.paths[2](x1)
        # B, C//2, H, W
        y1 = y1 * y1_d
        # B, C//2, H, W
        y1 = self.paths[3](y1)
        # B, C, H, W
        y = torch.cat([y1, y2], dim=1)
        return y


class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 momentum: float = 0.1,
                 eps: float = 1e-05) -> None:
        super().__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels,
                      eps=eps,
                      momentum=momentum),
            nn.ReLU(True),
        )


class SCConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 momentum: float = 0.1,
                 eps: float = 1e-05) -> None:
        super().__init__(
            SCConv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels,
                      eps=eps,
                      momentum=momentum),
            nn.ReLU(True),
        )


class ConvTransposeBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

