from typing import List, Tuple, Union

import numpy as np
from torch import nn
import torch

import spconv.pytorch as spconv

def cat(inputs: List[spconv.SparseConvTensor]) -> spconv.SparseConvTensor:
    features = torch.cat([input.features for input in inputs], dim=1)
    output = inputs[0].replace_feature(features)
    return output

def add(input1: spconv.SparseConvTensor, input2: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    features = input1.features + input2.features
    output = input1.replace_feature(features)
    return output

class BatchNorm(nn.BatchNorm1d):

    def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        features = super().forward(input.features)
        output = input.replace_feature(features)
        return output
    
class ReLU(nn.ReLU):

    def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        features = super().forward(input.features)
        output = input.replace_feature(features)
        return output


'''class SparseConvBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 indice_key: str, 
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1) -> None:
        if np.prod(stride) > 1:
            super().__init__(
                spconv.SparseConv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        indice_key=indice_key),
                BatchNorm(out_channels),
                ReLU(True),
            )
        else:
            super().__init__(
                spconv.SubMConv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        indice_key=indice_key),
                BatchNorm(out_channels),
                ReLU(True),
            )'''


class SparseConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 indice_key: str, 
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, list, tuple] = 0,
                 dilation: int = 1) -> None:
        super().__init__()
        if np.prod(stride) > 1:
            self.net = nn.Sequential(
                spconv.SparseConv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        indice_key=indice_key),
                BatchNorm(out_channels),
                ReLU(True),
            )
        else:
            self.net = nn.Sequential(
                spconv.SubMConv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                        indice_key=indice_key),
                BatchNorm(out_channels),
                ReLU(True),
            )
    
    def forward(self, x):
        return self.net(x)


'''class SparseConvTransposeBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 indice_key: str, 
                 kernel_size: Union[int, List[int], Tuple[int, ...]]
                ) -> None:
        super().__init__(
            spconv.SparseInverseConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                indice_key=indice_key
            ),
            BatchNorm(out_channels),
            ReLU(True),
        )'''


class SparseConvTransposeBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 indice_key: str, 
                 kernel_size: Union[int, List[int], Tuple[int, ...]]
                ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spconv.SparseInverseConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                indice_key=indice_key
            ),
            BatchNorm(out_channels),
            ReLU(True),
        )
    
    def forward(self, x):
        return self.net(x)


class SparseResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 indice_key: str,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1, 
                 ) -> None:
        super().__init__()
        if np.prod(stride) == 1:
            self.main = nn.Sequential(
                spconv.SubMConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    indice_key=indice_key
                ),
                BatchNorm(out_channels),
                ReLU(True),
                spconv.SubMConv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    indice_key=indice_key
                ),
                BatchNorm(out_channels),
            )
        else:
            self.main = nn.Sequential(
                spconv.SparseConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride, 
                    indice_key=indice_key + '1'
                ),
                BatchNorm(out_channels),
                ReLU(True),
                spconv.SubMConv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    indice_key=indice_key + '2'
                ),
                BatchNorm(out_channels),
            )
        if in_channels != out_channels or np.prod(stride) != 1:
            if np.prod(stride) != 1:
                self.shortcut = nn.Sequential(
                    spconv.SparseConv3d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=1, 
                        stride=stride, 
                        indice_key='indentity' + indice_key
                    ),
                    BatchNorm(out_channels),
                )
            else:
                self.shortcut = nn.Sequential(
                    spconv.SubMConv3d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=1, 
                        indice_key='indentity' + indice_key
                    ),
                    BatchNorm(out_channels),
                )
        else:
            self.shortcut = nn.Identity()

        self.relu = ReLU(True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.relu(add(self.main(x), self.shortcut(x)))
        return x


class SparseResUNet(nn.Module):

    def __init__(
        self,
        stem_channels: int,
        encoder_channels: List[int],
        decoder_channels: List[int],
        *,
        in_channels: int = 4,
        width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.stem_channels = stem_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier

        num_channels = [stem_channels] + encoder_channels + decoder_channels
        num_channels = [int(width_multiplier * nc) for nc in num_channels]

        self.stem = nn.Sequential(
            spconv.SubMConv3d(
                in_channels=in_channels, out_channels=num_channels[0], kernel_size=3, indice_key='subm0'
            ),
            BatchNorm(num_channels[0]),
            ReLU(True),
            spconv.SubMConv3d(
                in_channels=num_channels[0], out_channels=num_channels[0], kernel_size=3, indice_key='subm0'
            ),
            BatchNorm(num_channels[0]),
            ReLU(True),
        )

        # TODO(Zhijian): the current implementation of encoder and decoder
        # is hard-coded for 4 encoder stages and 4 decoder stages. We should
        # work on a more generic implementation in the future.

        self.encoders = nn.ModuleList()
        for k in range(4):
            key = str(k)
            self.encoders.append(
                nn.Sequential(
                    SparseConvBlock(
                        in_channels=num_channels[k],
                        out_channels=num_channels[k],
                        kernel_size=2,
                        stride=2,
                        padding=1,
                        indice_key=key
                    ),
                    SparseResBlock(in_channels=num_channels[k], 
                        out_channels=num_channels[k + 1], 
                        kernel_size=3,
                        indice_key='res0' + key),
                    SparseResBlock(in_channels=num_channels[k + 1], 
                        out_channels=num_channels[k + 1], 
                        kernel_size=3, 
                        indice_key='res1' + key),
                ))

        self.decoders = nn.ModuleList()
        for k in range(4):
            key = str(3-k)
            self.decoders.append(
                nn.ModuleDict({
                    'upsample':
                        SparseConvTransposeBlock(
                            in_channels=num_channels[k + 4],
                            out_channels=num_channels[k + 5],
                            kernel_size=2,
                            indice_key=key
                        ),
                    'fuse':
                        nn.Sequential(
                            SparseResBlock(
                                in_channels=num_channels[k + 5] + num_channels[3 - k],
                                out_channels=num_channels[k + 5],
                                kernel_size=3,
                                indice_key='res2' + key
                            ),
                            SparseResBlock(
                                in_channels=num_channels[k + 5],
                                out_channels=num_channels[k + 5],
                                kernel_size=3,
                                indice_key='res3' + key
                            ),
                        )
                }))

    def _unet_forward(
        self,
        x: spconv.SparseConvTensor,
        encoders: nn.ModuleList,
        decoders: nn.ModuleList,
    ) -> List[spconv.SparseConvTensor]:
        if not encoders and not decoders:
            return [x]

        # downsample
        xd = encoders[0](x)

        # inner recursion
        outputs = self._unet_forward(xd, encoders[1:], decoders[:-1])
        yd = outputs[-1]

        # upsample and fuse
        u = decoders[-1]['upsample'](yd)
        y = decoders[-1]['fuse'](cat([u, x]))

        return [x] + outputs + [y]

    def forward(self, x: spconv.SparseConvTensor) -> List[spconv.SparseConvTensor]:
        return self._unet_forward(self.stem(x), self.encoders, self.decoders)


class SparseResUNet42(SparseResUNet):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            stem_channels=32,
            encoder_channels=[32, 64, 128, 256],
            decoder_channels=[256, 128, 96, 96],
            **kwargs,
        )


class SparseResNet(nn.ModuleList):

    def __init__(
        self,
        blocks: List[Tuple[int, int, Union[int, Tuple[int, ...]],
                           Union[int, Tuple[int, ...]], Union[int, Tuple[int, ...]]]],
        *,
        in_channels: int = 4,
        width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier

        block_count = 0
        for num_blocks, out_channels, kernel_size, stride, padding in blocks:
            out_channels = int(out_channels * width_multiplier)
            blocks = []
            for index in range(num_blocks):
                if index == 0:
                    block_count += 1
                    blocks.append(
                        SparseConvBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding, 
                            indice_key=str(block_count)
                        ))
                    block_count += 1
                else:
                    blocks.append(
                        SparseResBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            indice_key=str(block_count)
                        ))
                in_channels = out_channels
            self.append(nn.Sequential(*blocks))

    def forward(self, x: spconv.SparseConvTensor) -> List[spconv.SparseConvTensor]:
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs


class SparseResNet21D(SparseResNet):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            blocks=[
                (3, 16, 3, 1, 1),
                (3, 32, 3, 2, 1),
                (3, 64, 3, 2, 1),
                (3, 128, 3, 2, [1, 0, 1]),
                (1, 128, [1, 3, 1], [1, 2, 1], 0),
            ],
            **kwargs,
        )
