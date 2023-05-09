from torch import nn
# from torchsparse import nn as spnn
from ..modules import SparseConvBlock, SparseResBlock
# import core.models.modules.wrapper as wrapper
from .backbone3d_template import Backbone3DTemplate

__all__ = ['SparseResNet']

### modified to fit PCEngine
class SparseResNet(Backbone3DTemplate):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()

        self.num_channels = num_channels = [16, 32, 64, 128]
        self.in_channels = in_channels
        self.out_channels = num_channels[-1]
        # self.backend = backend = kwargs.get('backend', 'torchsparse')
        # self.kmap_mode = kwargs.get('kmap_mode', 'hashmap')
        # self.wrapper = wrapper.Wrapper(backend=self.backend)

        self.stem = nn.Sequential(
            SparseConvBlock(in_channels, num_channels[0], 3, stride=1, padding=1),
            SparseResBlock(num_channels[0], num_channels[0], 3),
            SparseResBlock(num_channels[0], num_channels[0], 3)
        )
        self.stage1 = nn.Sequential(
            SparseConvBlock(num_channels[0], num_channels[1], 3, stride=[2, 2, 2], padding=1),
            SparseResBlock(num_channels[1], num_channels[1], 3),
            SparseResBlock(num_channels[1], num_channels[1], 3),
        )
        self.stage2 = nn.Sequential(
            SparseConvBlock(num_channels[1], num_channels[2], 3, stride=[2, 2, 2], padding=1),
            SparseResBlock(num_channels[2], num_channels[2], 3),
            SparseResBlock(num_channels[2], num_channels[2], 3),
        )
        self.stage3 = nn.Sequential(
            SparseConvBlock(num_channels[2], num_channels[3], 3, stride=[2, 2, 2], padding=[1, 0, 1]),
            SparseResBlock(num_channels[3], num_channels[3], 3),
            SparseResBlock(num_channels[3], num_channels[3], 3),
        )

        self.stage4 = SparseConvBlock(
            num_channels[3], num_channels[3], kernel_size=[1, 3, 1], stride=[1, 2, 1]
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # print(x0.feats.shape, x1.feats.shape, x2.feats.shape, x3.feats.shape, x4.feats.shape)
        return [x0, x1, x2, x3, x4]
