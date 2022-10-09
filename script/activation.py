from torch import nn

from .sptensor import spTensor
from .utils import fapply

class ReLU(nn.ReLU):

    def forward(self, input: spTensor) -> spTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):

    def forward(self, input: spTensor) -> spTensor:
        return fapply(input, super().forward)
