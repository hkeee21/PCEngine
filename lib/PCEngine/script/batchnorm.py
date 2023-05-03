import torch
from torch import nn

from .sptensor import spTensor
from .utils import fapply

class BatchNorm(nn.BatchNorm1d):

    def forward(self, input: spTensor) -> spTensor:
        return fapply(input, super().forward)

