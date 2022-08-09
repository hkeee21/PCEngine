import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch
from torch.utils.cpp_extension import load
import math
from sptensor import spTensor
from dgCloud.backend import mapping_cuda, conv_fwd_cuda, conv_bwd_cuda


class conv3d(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 tc_mode_16f: bool = 0
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.tensorcore16F = tc_mode_16f
        self.kernel_volume = self.kernel_size ** 3

        self.kernel = nn.Parameter(
                torch.rand((self.kernel_volume, in_channels, out_channels), \
                    dtype=torch.float))
        self.reset_parameters()
    

    def forward(self, input: spTensor) -> spTensor:
        
        return conv_func(input=input,
                        in_channel=self.in_channels, 
                        out_channel=self.out_channels,
                        kernel=self.kernel,
                        kernel_size=self.kernel_size,
                        tensorcore16F = self.tensorcore16F
                        )
    

    def reset_parameters(self):
        with torch.no_grad():
            n = self.out_channels * self.kernel_volume
            stdv = 1.0 / math.sqrt(n)
            self.kernel.data.uniform_(-stdv, stdv)
    

class convF(Function):
    
    @staticmethod
    @custom_fwd
    def forward(ctx,
                in_feats: torch.Tensor,
                knnz: torch.Tensor, 
                kpos: torch.Tensor, 
                imap: torch.Tensor, 
                omap: torch.Tensor,
                in_channel: int,
                out_channel: int, 
                kernel: torch.Tensor, 
                kernel_size: int, 
                in_buffer: torch.Tensor, 
                out_buffer: torch.Tensor, 
                tensorcore16F: bool) -> torch.Tensor:
        
        input = input.contiguous()
        kernel = kernel.contiguous()
        imap = imap.contiguous()
        omap = omap.contiguous()
        in_buffer = in_buffer.contiguous()
        out_buffer = out_buffer.contiguous()

        output_feats = torch.zeros((in_feats.size(0), out_channel), \
            dtype=torch.float, device=in_feats.device)

        conv_fwd_cuda(
        in_feats,
        kernel,
        kernel_size,
        output_feats,
        knnz,
        kpos, 
        imap,
        omap,
        in_buffer, 
        out_buffer, 
        tensorcore16F
        )
    
        # TODO: replace in_feats with gathered features in in_buffer
        ctx.for_backwards = (in_feats, kernel, kernel_size, knnz, kpos, imap, omap, \
            in_buffer, out_buffer, tensorcore16F)

    
    @staticmethod
    @custom_bwd
    def backward(ctx, out_feats_grad: torch.Tensor):

        in_feats, kernel, kernel_size, knnz, kpos, imap, omap, \
            in_buffer, out_buffer, tensorcore16F = ctx.for_backwards
        
        input_size = in_feats.size(0)
        input_channel = in_feats.size(1)
        output_channel = kernel.size(-1)
        kernel_volume = kernel.size(0)

        in_feats_grad = torch.zeros((input_size, input_channel), dtype=torch.float, \
            device=in_feats.device)
        weight_grad = torch.zeros((kernel_volume, input_channel, output_channel), \
            dtype=torch.float, device=in_feats.device)

        conv_bwd_cuda(
            out_feats_grad, 
            in_feats,
            kernel,
            kernel_size,
            in_feats_grad, 
            weight_grad, 
            knnz,
            kpos, 
            imap,
            omap,
            in_buffer, 
            out_buffer,  
            tensorcore16F
        )

        return in_feats_grad, None, None, None, None, None, None, \
            weight_grad, None, None, None, None


def conv_func(input, in_channel, out_channel, kernel, kernel_size, tensorcore16F):
    
    input_size = input.coords.size(0)

    if kernel_size not in input.mappat:
        
        input.mappat.append(kernel_size)
        
        input.knnz[kernel_size] = torch.zeros((kernel_size ** 3 - 1), dtype=torch.int, \
            device=input.coords.device)
        input.imap[kernel_size] = - torch.ones((input_size * (kernel_size ** 3 - 1)), \
            dtype=torch.int, device=input.coords.device)
        input.omap[kernel_size] = - torch.ones((input_size * (kernel_size ** 3 - 1)), \
            dtype=torch.int, device=input.coords.device)
        
        mapping_cuda(
            input.coords,
            kernel_size,
            input.imap[kernel_size],
            input.omap[kernel_size],  
            input.knnz[kernel_size]
        )

        input.kpos[kernel_size] = torch.zeros((kernel_size ** 3 - 1), dtype=torch.int, \
            device=input.coords.device)
        for k in range(kernel_size ** 3 - 2):
            input.kpos[kernel_size][k + 1] = input.kpos[kernel_size][k] + input.knnz[kernel_size][k]
        
        sum_nnz = input.knnz[kernel_size].sum()
        input.gbuf[kernel_size] = torch.zeros((sum_nnz, in_channel), dtype=torch.float, \
            device=input.feats.device)
        input.sbuf[kernel_size] = torch.zeros((sum_nnz, out_channel), dtype=torch.float, \
            device=input.feats.device)
        
        # nonzero_idx = torch.nonzero(map != -1)
        # input.imap[kernel_size] = map[nonzero_idx]
        # input.omap[kernel_size] = (nonzero_idx % input_size).int()
        
    output_feats = convF.apply(
        input.feats,
        input.knnz[kernel_size], 
        input.kpos[kernel_size],
        input.imap[kernel_size],
        input.omap[kernel_size],
        in_channel, 
        out_channel,
        kernel, 
        kernel_size, 
        input.gbuf[kernel_size], 
        input.sbuf[kernel_size], 
        tensorcore16F
    )

    output = spTensor(output_feats, input.coords)
    output.mappat = input.mappat
    output.imap[kernel_size] = input.imap[kernel_size]
    output.omap[kernel_size] = input.omap[kernel_size]
    output.knnz[kernel_size] = input.knnz[kernel_size]
    output.kpos[kernel_size] = input.kpos[kernel_size]
    output.gbuf[kernel_size] = input.gbuf[kernel_size]
    output.sbuf[kernel_size] = input.sbuf[kernel_size]
    
    return output
