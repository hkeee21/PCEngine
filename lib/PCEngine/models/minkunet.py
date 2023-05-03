from typing import List

from torch import nn

from ..script.sptensor import spTensor
from ..script.spconv import conv3d
from ..script.batchnorm import BatchNorm
from ..script.activation import ReLU
from ..script.utils import cat

from .modules import SparseConvBlock, SparseConvTransposeBlock, SparseResBlock


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
            conv3d(in_channels=in_channels, out_channels=num_channels[0], kernel_size=3),
            BatchNorm(num_channels[0]),
            ReLU(True),
            conv3d(in_channels=num_channels[0], out_channels=num_channels[0], kernel_size=3),
            BatchNorm(num_channels[0]),
            ReLU(True),
        )

        # TODO(Zhijian): the current implementation of encoder and decoder
        # is hard-coded for 4 encoder stages and 4 decoder stages. We should
        # work on a more generic implementation in the future.

        self.encoders = nn.ModuleList()
        for k in range(4):
            self.encoders.append(
                nn.Sequential(
                    SparseConvBlock(
                        num_channels[k],
                        num_channels[k],
                        2,
                        stride=2,
                    ),
                    SparseResBlock(num_channels[k], num_channels[k + 1], 3),
                    SparseResBlock(num_channels[k + 1], num_channels[k + 1], 3),
                ))

        self.decoders = nn.ModuleList()
        for k in range(4):
            self.decoders.append(
                nn.ModuleDict({
                    'upsample':
                        SparseConvTransposeBlock(
                            num_channels[k + 4],
                            num_channels[k + 5],
                            2,
                            stride=2,
                        ),
                    'fuse':
                        nn.Sequential(
                            SparseResBlock(
                                num_channels[k + 5] + num_channels[3 - k],
                                num_channels[k + 5],
                                3,
                            ),
                            SparseResBlock(
                                num_channels[k + 5],
                                num_channels[k + 5],
                                3,
                            ),
                        )
                }))

    def _unet_forward(
        self,
        x: spTensor,
        encoders: nn.ModuleList,
        decoders: nn.ModuleList,
    ) -> List[spTensor]:
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

    def forward(self, x: spTensor) -> List[spTensor]:
        return self._unet_forward(self.stem(x), self.encoders, self.decoders)


class SparseResUNet42(SparseResUNet):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            stem_channels=32,
            encoder_channels=[32, 64, 128, 256],
            decoder_channels=[256, 128, 96, 96],
            **kwargs,
        )
