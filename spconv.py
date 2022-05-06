import torch.nn as nn
import torch
import torch.Tensor as Tensor
from torch.utils.cpp_extension import load

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
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        
    def forward(self, coords: Tensor, feats: Tensor) -> Tensor:
        return conv_func(coords=coords,
                        feats=feats,
                        in_channel=self.in_channels,
                        out_channel=self.out_channels,
                        kernel=self.kernel,
                        kernel_size=self.kernel_size,
                        )
    

def conv_func(coords, feats, in_channel, out_channel, kernel, kernel_size):
    assert(coords.size(0) == feats.size(0), "Coords and Feats mismatch!")
    input_size = coords.size(0)
    assert(feats.size(1) == in_channel, "The dimension of input feature is wrong!")

    output = torch.zeros((input_size, out_channel), dtype=torch.float)
    map = torch.zeros((input_size, 1), dtype=torch.int)

    backend.conv_fwd_cuda(
                coords,
                feats,
                kernel,
                kernel_size,
                map,
                output
                )
    
    return output
