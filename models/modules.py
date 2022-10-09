from typing import List, Tuple, Union

import numpy as np
from torch import nn

from script.sptensor import spTensor
from script.spconv import conv3d
from script.batchnorm import BatchNorm
from script.activation import ReLU


class SparseConvBlock(nn.Sequential):

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
                    stride=stride
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
