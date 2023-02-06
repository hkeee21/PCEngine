import torch
import torch.nn as nn

__all__ = ['Backbone3DTemplate']

class Backbone3DTemplate(nn.Module):
    def __init__(self):
        super().__init__()

    def reset_bn_params(self, momentum=0.1, eps=1e-5):
        for name, module in self.named_modules():
            for x in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                if isinstance(module, x):
                    module.momentum = momentum
                    module.eps = eps
                    break

