import torch.nn as nn
import torch
from torch.utils.cpp_extension import load
import math

backend = load(name="conv_fwd_cuda",
                   sources=["backend/pybind_cuda.cpp", 
                   "backend/spconv.cu"],
                   verbose=True)


class conv3d(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_volume = self.kernel_size ** 3

        self.kernel = nn.Parameter(
                torch.rand((self.kernel_volume, in_channels, out_channels), dtype=torch.float))
        self.reset_parameters()
    

    def forward(self, coords: torch.IntTensor, feats: torch.FloatTensor, 
                maps: torch.IntTensor, mappat: list) -> torch.FloatTensor:
        
        return conv_func(coords=coords,
                        feats=feats,
                        maps=maps,
                        mappat=mappat,
                        out_channel=self.out_channels,
                        kernel=self.kernel,
                        kernel_size=self.kernel_size,
                        )
    

    def reset_parameters(self):
        with torch.no_grad():
            n = self.out_channels * self.kernel_volume
            stdv = 1.0 / math.sqrt(n)
            self.kernel.data.uniform_(-stdv, stdv)
    


def conv_func(coords, feats, maps, mappat, out_channel, kernel, kernel_size):
    
    input_size = coords.size(0)

    output = torch.zeros((input_size, out_channel), dtype=torch.float, device=feats.device)

    # hashgemm-v2
    # remap = True

    # hashgemm-v3
    
    remap = False
    if kernel_size not in mappat:
        remap = True
        mappat.append(kernel_size)

    backend.conv_fwd_cuda(
                coords,
                feats,
                kernel,
                kernel_size,
                maps,
                output,
                remap
                )
    
    return output
