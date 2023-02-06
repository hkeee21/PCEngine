import numpy as np
import torch
from torch import nn
from ..modules import ConvBlock, SCConvBlock, ConvTransposeBlock
from .backbone2d_template import Backbone2DTemplate


__all__ = ['DenseRPNHead']


class DenseRPNHead(Backbone2DTemplate):
    def __init__(self, classes, in_channels, 
                 loc_min, loc_max, proposal_stride, 
                 need_tobev=False, sep_heads=False, **kwargs) -> None:
        super().__init__()
        
        _num_channels = kwargs.get('num_channels', [64, 128, 256])
        _up_num_channels = kwargs.get('up_num_channels', [128, 128, 128])
        stride = kwargs.get('stride', [2, 2, 2])
        layer_nums = kwargs.get('layer_nums', [3, 5, 5])
        up_stride = kwargs.get('up_stride', [1, 2, 4])
        use_scconv = kwargs.get('use_scconv', False)

        self.classes = classes
        self.num_channels = num_channels = [in_channels] + _num_channels
        self.stride = stride
        self.layer_nums = layer_nums
        self.up_num_channels = up_num_channels = _up_num_channels
        self.up_stride = up_stride
        self.sep_heads = sep_heads
        
        blocks = []
        for i, layer_num in enumerate(self.layer_nums):
            block = [
                ConvBlock(
                    num_channels[i], num_channels[i+1], 3, 
                    stride=self.stride[i], padding=1
                )
            ]
            _blk = ConvBlock if not use_scconv else SCConvBlock
            for j in range(layer_num):
                block.append(
                    _blk(
                        num_channels[i+1], num_channels[i+1], 3, 
                        stride=1, padding=1
                    )
                )
            block = nn.Sequential(*block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        
        upsample_blocks = []
        for i, c in enumerate(up_num_channels):
            if self.up_stride[i] >= 1:
                upsample_blocks.append(
                    ConvTransposeBlock(
                        num_channels[i+1], up_num_channels[i], self.up_stride[i], 
                        stride=self.up_stride[i], padding=0
                    )
                )
            else:
                upsample_blocks.append(
                    ConvBlock(
                        num_channels[i+1], up_num_channels[i], 
                        int(1 // self.up_stride[i]), 
                        stride=int(1 // self.up_stride[i]), padding=0
                    )
                )
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        
        upsample_outs = [self.upsample_blocks[i](x) for i, x in enumerate(outs)]
        out = torch.cat(upsample_outs, dim=1)
        return out
