
import torch.nn as nn
# import torchsparse.nn as spnn
# import core.models.modules.wrapper as wrapper
from script.spconv import conv3d
from script.batchnorm import BatchNorm
from script.activation import ReLU
from script.utils import cat
from ..modules import SparseConvBlock, SparseConvTransposeBlock, SparseResBlock

class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        ic = kwargs.get('in_channels', 4)
        cr = kwargs.get('cr', 1.0)
        cs = [64, 64, 64, 128, 256, 256, 128, 64, 64]
        cs = [int(cr * x) for x in cs]
        # make sure #channels is even
        for i, x in enumerate(cs):
            if x % 2 != 0:
                cs[i] = x + 1

        # self.backend = backend
        # self.wrapper = wrapper.Wrapper(backend=self.backend)
        
        '''self.stem = self.wrapper.sequential(
            self.wrapper.conv3d(ic, cs[0], kernel_size=3, stride=1, indice_key="pre"),
            self.wrapper.bn(cs[0]), self.wrapper.relu(True),
            self.wrapper.conv3d(cs[0], cs[0], kernel_size=3, stride=1, indice_key="pre"),
            self.wrapper.bn(cs[0]), self.wrapper.relu(True))'''
        
        self.stem = nn.Sequential(
            conv3d(in_channels=ic, out_channels=cs[0], kernel_size=3),
            BatchNorm(cs[0]), ReLU(True), 
            conv3d(in_channels=cs[0], out_channels=cs[0], kernel_size=3),
            BatchNorm(cs[0]), ReLU(True)
        )
        
        self.stage1 = nn.Sequential(
            SparseConvBlock(cs[0], cs[0], kernel_size=2, stride=[2, 2, 2]),
            SparseResBlock(cs[0], cs[1], kernel_size=3),
            SparseResBlock(cs[1], cs[1], kernel_size=3),
        )

        self.stage2 = nn.Sequential(
            SparseConvBlock(cs[1], cs[1], kernel_size=2, stride=[2, 2, 2]),
            SparseResBlock(cs[1], cs[2], kernel_size=3),
            SparseResBlock(cs[2], cs[2], kernel_size=3))

        self.stage3 = nn.Sequential(
            SparseConvBlock(cs[2], cs[2], kernel_size=2, stride=[2, 2, 2]),
            SparseResBlock(cs[2], cs[3], kernel_size=3),
            SparseResBlock(cs[3], cs[3], kernel_size=3),
        )

        self.stage4 = nn.Sequential(
            SparseConvBlock(cs[3], cs[3], kernel_size=2, stride=[2, 2, 2]),
            SparseResBlock(cs[3], cs[4], kernel_size=3),
            SparseResBlock(cs[4], cs[4], kernel_size=3),
        )

        self.up1 = nn.ModuleList([
            SparseConvTransposeBlock(cs[4], cs[5], kernel_size=2, stride=[2, 2, 2]),
            nn.Sequential(
                SparseResBlock(cs[5] + cs[3], cs[5], kernel_size=3),
                SparseResBlock(cs[5], cs[5], kernel_size=3),
            )
        ])

        self.up2 = nn.ModuleList([
            SparseConvTransposeBlock(cs[5], cs[6], kernel_size=2, stride=[2, 2, 2]),
            nn.Sequential(
                SparseResBlock(cs[6] + cs[2], cs[6], kernel_size=3),
                SparseResBlock(cs[6], cs[6], kernel_size=3),
            )
        ])

        self.up3 = nn.ModuleList([
            SparseConvTransposeBlock(cs[6], cs[7], kernel_size=2, stride=[2, 2, 2]),
            nn.Sequential(
                SparseResBlock(cs[7] + cs[1], cs[7], kernel_size=3),
                SparseResBlock(cs[7], cs[7], kernel_size=3),
            )
        ])

        self.up4 = nn.ModuleList([
            SparseConvTransposeBlock(cs[7], cs[8], kernel_size=2, stride=[2, 2, 2]),
            nn.Sequential(
                SparseResBlock(cs[8] + cs[0], cs[8], kernel_size=3),
                SparseResBlock(cs[8], cs[8], kernel_size=3),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8],
                                                  kwargs['num_classes']))

        self.weight_initialization()
        #self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x['pts_input']

        # print(x.feats.shape)

        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # print(x4.coords.shape[0])
        
        y1 = self.up1[0](x4)
        y1 = cat([y1, x3])
        y1 = self.up1[1](y1)

        # print(y1.coords.shape[0])
        
        y2 = self.up2[0](y1)
        y2 = cat([y2, x2])
        y2 = self.up2[1](y2)
    
        # print(y2.coords.shape[0])

        y3 = self.up3[0](y2)
        y3 = cat([y3, x1])
        y3 = self.up3[1](y3)

        # print(y3.coords.shape[0])

        y4 = self.up4[0](y3)
        y4 = cat([y4, x0])
        y4 = self.up4[1](y4)

        # print(y4.coords.shape[0])
        # print(y1.coords.shape[0], y2.coords.shape[0], y3.coords.shape[0], y4.coords.shape[0])
        
        out = self.classifier(y4.F)

        return out


