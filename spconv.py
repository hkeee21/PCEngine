import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch
from torch.utils.cpp_extension import load
import math
from sptensor import spTensor
from dgCloud.backend import mapping_cuda, conv_fwd_cuda, conv_bwd_cuda
import time


class conv3d(nn.Module):
    
    def __init__(self,
                 in_channels: int = 16,
                 out_channels: int = 32,
                 buffer: torch.Tensor = None, 
                 kernel_size: int = 3,
                 tc_mode_16f: bool = 0
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.buffer = buffer
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
                        buffer=self.buffer, 
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
                icsr: torch.Tensor,
                ocsr: torch.Tensor, 
                in_channel: int,
                out_channel: int, 
                kernel: torch.Tensor, 
                kernel_size: int, 
                sum_nnz: int, 
                buffer: torch.Tensor, 
                tensorcore16F: bool) -> torch.Tensor:
        
        in_feats = in_feats.contiguous()
        kernel = kernel.contiguous()
        imap = imap.contiguous()
        omap = omap.contiguous()
        icsr = icsr.contiguous()
        ocsr = ocsr.contiguous()
        buffer = buffer.contiguous()

        output_feats = torch.zeros((in_feats.size(0), out_channel), \
            dtype=torch.float, device=in_feats.device)

        conv_fwd_cuda(
            in_feats,
            kernel,
            kernel_size,
            sum_nnz, 
            output_feats,
            knnz,
            kpos, 
            imap,
            omap,
            icsr,
            ocsr, 
            buffer, 
            tensorcore16F
        )
    
        # TODO: replace in_feats with gathered features in in_buffer
        ctx.for_backwards = (in_feats, kernel, kernel_size, knnz, imap, omap, \
            buffer, tensorcore16F)
        
        return output_feats

    
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

        return in_feats_grad, None, None, None, None, None, \
            weight_grad, None, None, None, None


def conv_func(input, in_channel, out_channel, buffer, 
    kernel, kernel_size, tensorcore16F):
    
    input_size = input.coords.size(0)

    if kernel_size not in input.mappat:
    # if True:
    
        input.mappat.append(kernel_size)
        
        input.knnz[kernel_size] = torch.zeros((kernel_size ** 3), dtype=torch.int, \
            device=input.coords.device)
        input.imap[kernel_size] = - torch.ones((input_size * (kernel_size ** 3 - 1)), \
            dtype=torch.int, device=input.coords.device)
        input.omap[kernel_size] = - torch.ones((input_size * (kernel_size ** 3 - 1)), \
            dtype=torch.int, device=input.coords.device)
        input.kpos[kernel_size] = torch.zeros((kernel_size ** 3 - 1), dtype=torch.int, \
            device=input.coords.device)
        input.icsr[kernel_size] = torch.zeros((input_size + 2), dtype=torch.int, \
            device=input.coords.device)
        input.ocsr[kernel_size] = torch.zeros((input_size + 2), dtype=torch.int, \
            device=input.coords.device)
        
        # torch.cuda.synchronize()
        # start=time.time()

        sum_nnz = mapping_cuda(
            input.coords,
            kernel_size,
            in_channel, 
            out_channel, 
            input.imap[kernel_size],
            input.omap[kernel_size], 
            input.icsr[kernel_size], 
            input.ocsr[kernel_size],  
            input.knnz[kernel_size],
            input.kpos[kernel_size]
        )

        # torch.cuda.synchronize()
        # end=time.time()
        # dur=(end-start) * 1000000
        # print(dur)   # us

        # print(input.buf.shape)

        # the loop is inefficient
        # input.kpos[kernel_size] = torch.zeros((kernel_size ** 3 - 1), dtype=torch.int, \
        #     device=input.coords.device)
        # for k in range(kernel_size ** 3 - 2):
        #     input.kpos[kernel_size][k + 1] = input.kpos[kernel_size][k] + input.knnz[kernel_size][k]
        # input.kpos[kernel_size] = torch.matmul(input.knnz[kernel_size].float(), torch.triu(torch.ones(kernel_size ** 3 - 1, \
        # kernel_size ** 3 - 1).to(input.coords.device), diagonal=1)).int()
        
        # inonzero = torch.nonzero(input.imap[kernel_size] != -1)
        # input.imap[kernel_size] = input.imap[kernel_size][inonzero]
        # ononzero = torch.nonzero(input.omap[kernel_size] != -1)
        # input.omap[kernel_size] = input.omap[kernel_size][ononzero]

        # sum_nnz = input.icsr[kernel_size][input_size].cpu()

        # print(sum_nnz)
        input.knnz[kernel_size] = input.knnz[kernel_size].cpu()
        input.snnz[kernel_size] = sum_nnz

        # print("sum_nnz: %d" % sum_nnz)
        if buffer.size(0) < sum_nnz * (in_channel + out_channel):
            print("Newly allocated buffer.")
            buffer = torch.zeros((sum_nnz, (in_channel + out_channel)), dtype=torch.float, \
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
        input.icsr[kernel_size],
        input.ocsr[kernel_size],
        in_channel, 
        out_channel,
        kernel, 
        kernel_size,
        input.snnz[kernel_size],  
        buffer,
        tensorcore16F
    )

    output = spTensor(output_feats, input.coords)
    output.mappat = input.mappat
    output.imap[kernel_size] = input.imap[kernel_size]
    output.omap[kernel_size] = input.omap[kernel_size]
    output.icsr[kernel_size] = input.icsr[kernel_size]
    output.ocsr[kernel_size] = input.ocsr[kernel_size]
    output.knnz[kernel_size] = input.knnz[kernel_size]
    output.kpos[kernel_size] = input.kpos[kernel_size]
    output.snnz[kernel_size] = input.snnz[kernel_size]
    
    return output
