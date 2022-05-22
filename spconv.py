import torch.nn as nn
import torch
from torch.utils.cpp_extension import load
import math
from sptensor import spTensor

hash_module = load(name="mapping_cuda",
                   sources=["backend/pybind_hash.cpp", 
                   "backend/hash.cu"],
                   verbose=True)

conv_module = load(name="conv_fwd_cuda",
                   sources=["backend/pybind_conv.cpp", 
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
    

    def forward(self, input: spTensor) -> torch.FloatTensor:
        
        return conv_func(input=input,
                        out_channel=self.out_channels,
                        kernel=self.kernel,
                        kernel_size=self.kernel_size,
                        )
    

    def reset_parameters(self):
        with torch.no_grad():
            n = self.out_channels * self.kernel_volume
            stdv = 1.0 / math.sqrt(n)
            self.kernel.data.uniform_(-stdv, stdv)
    


def conv_func(input, out_channel, kernel, kernel_size):
    
    input_size = input.coords.size(0)

    output_feats = torch.zeros((input_size, out_channel), dtype=torch.float, device=input.coords.device)

    if kernel_size not in input.mappat:
        
        input.mappat.append(kernel_size)
        input.maps[kernel_size] = torch.zeros((input_size * (kernel_size ** 3)), dtype=torch.int, device=input.coords.device)
        input.knnz[kernel_size] = torch.zeros((kernel_size ** 3), dtype=torch.int, device=input.coords.device)

        input.kidx[kernel_size] = hash_module.mapping_cuda(
            input.coords,
            kernel_size,
            input.maps[kernel_size],
            input.knnz[kernel_size]
        )

    conv_module.conv_fwd_cuda(
        input.coords,
        input.feats,
        kernel,
        kernel_size,
        input.maps[kernel_size],
        output_feats,
        input.knnz[kernel_size],
        input.kidx[kernel_size]
    )

    output = spTensor(output_feats, input.coords)
    output.mappat = input.mappat
    output.maps[kernel_size] = input.maps[kernel_size]
    output.knnz[kernel_size] = input.knnz[kernel_size]
    output.kidx[kernel_size] = input.kidx[kernel_size]
    
    return output
